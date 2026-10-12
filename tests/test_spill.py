"""Spill: an idle device takes a waiting call whose fallback it is while the call's own device is
busy with another job. One test per edge case of docs/GPU_SCHEDULING_CASES.zh-CN.md (E-numbers)."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gpu_runtime
from gpu_runtime import GIB, Device, load_config
from test_device_queues import LaneRuntime, QueueHarness
import harness as base
from pair_tenant import PAIR


class SpillRuntime(LaneRuntime):
    """Renders encode on gpu1 and sample on gpu0 (LaneRuntime). Server tenants live at home gpu1
    with fallback gpu0; `where` is the device each is loaded on, as Scheduler.located reports."""
    def __init__(self):
        super().__init__()
        self.tenants={'tr':{'spill':True,'fallback':'gpu0'},'nospill':{'fallback':'gpu0'},'solo':{'spill':True}}
        self.where={};self.server_on=self.where
    def validate_engine(self,req):pass
    def _slot(self,req):return self.tenants.get(req.get('engine'))
    def stage_devices(self,kind,req):
        if kind=='engine' and self._slot(req) is not None:
            fallback=self._slot(req).get('fallback');allowed=req.get('devices') or ('gpu1',fallback)
            return ('gpu1' if 'gpu1' in allowed else None),(fallback if fallback in allowed else None)
        return super().stage_devices(kind,req)
    def stages(self,kind,req):
        if kind=='engine' and self._slot(req) is not None:
            fallback=self._slot(req).get('fallback');down=lambda d:d in self.lent or d in self.blocked
            return [('engine',fallback if fallback and down('gpu1') and not down(fallback) else 'gpu1')]
        return super().stages(kind,req)
    def spills(self,kind,req):return kind=='engine' and bool((self._slot(req) or {}).get('spill'))
    def located(self,name):return self.where.get(name)
    def engine(self,req,out,lane='gpu1'):
        self.where[req['engine']]=lane;return super().engine(req,out,lane)


class SpillTests(QueueHarness):
    def setUp(self):
        super().setUp();self.s.close()
        self.rt=SpillRuntime();self.s=Service(ROOT,self.state,self.rt)
    def call_(self,path,engine='tr'):
        return self.call('engine_submit',engine=engine,path=path,method='POST',body='',content_type='x',request_key=path)
    def busy_gpu1(self):
        """gpu1 serving a long call of a tenant that has no fallback (an encode would also need gpu0)."""
        gate=self.rt.gate('engine:/busy');job=self.call('engine_submit',engine='audio',path='/busy',method='POST',body='',content_type='x',request_key='busy')
        self.rt.wait_started('engine:/busy');return gate,job

    def test_U3_a_call_runs_on_the_idle_fallback_while_its_device_encodes(self):
        gate,busy=self.busy_gpu1()
        c=self.call_('/c');self.assertEqual(self.wait(c)['status'],'completed','not waiting for the encode')
        self.assertEqual((self.rt.engine_on['/c'],self.job(c)['lane']),('gpu0','gpu0'))
        self.assertNotIn(('end','engine:/busy'),self.rt.log)
        gate.set();self.wait(busy)

    def test_E9_a_device_with_its_own_queue_is_not_idle(self):
        sampling=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        z=self.render('z');self.until(lambda:self.between(z),what='z waits for gpu0')
        gate,busy=self.busy_gpu1();c=self.call_('/c')
        time.sleep(.2);self.assertNotIn('/c',self.rt.engine_on,'gpu0 is sampling: nothing to spill onto')
        sampling.set();self.assertEqual(self.wait(c)['status'],'completed')
        self.assertTrue(self.order('sample:z','engine:/c'),'gpu0 serves its own queue first, then takes the call')
        gate.set()
        for job in (x,z,busy):self.wait(job)

    def test_a_call_allowed_only_its_device_waits_for_it(self):
        gate,busy=self.busy_gpu1()
        c=self.call('engine_submit',engine='tr',path='/c',method='POST',body='',content_type='x',request_key='/c',devices=['gpu1'])
        time.sleep(.2);self.assertNotIn('/c',self.rt.engine_on,'gpu0 is idle, but not allowed')
        gate.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c'],'gpu1');self.wait(busy)

    def test_E10_E18_a_tenant_without_spill_waits_for_its_device(self):
        gate,busy=self.busy_gpu1();c=self.call_('/c',engine='nospill')
        time.sleep(.2);self.assertNotIn('/c',self.rt.engine_on)
        gate.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c'],'gpu1')
        self.wait(busy)

    def test_a_tenant_without_a_fallback_has_nowhere_to_spill(self):
        gate,busy=self.busy_gpu1();c=self.call_('/c',engine='solo')
        time.sleep(.2);self.assertNotIn('/c',self.rt.engine_on)
        gate.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c'],'gpu1');self.wait(busy)

    def test_E2_a_tenant_loaded_on_its_device_waits_out_a_short_job_there(self):
        self.rt.where['tr']='gpu1'
        with patch.object(job_service,'SPILL_AFTER_S',5):
            gate,busy=self.busy_gpu1();c=self.call_('/c')
            time.sleep(.3);self.assertNotIn('/c',self.rt.engine_on,'moving it would cost a load a short job there would not')
            gate.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c'],'gpu1');self.wait(busy)

    def test_E2_but_not_a_long_one(self):
        """Seen in planning: an image on gpu1 next to a resident ASR held every ASR call for two minutes."""
        self.rt.where['tr']='gpu1'
        with patch.object(job_service,'SPILL_AFTER_S',.3):
            gate,busy=self.busy_gpu1();c=self.call_('/c');t=time.monotonic()
            self.assertEqual(self.wait(c)['status'],'completed');self.assertEqual(self.rt.engine_on['/c'],'gpu0')
            self.assertGreaterEqual(time.monotonic()-t,.2,'only once the other job had held gpu1 a while')
            gate.set();self.wait(busy)

    def test_E3_never_out_of_its_own_session(self):
        self.rt.where['tr']='gpu1';held=self.rt.gate('engine:/first')
        with patch.object(job_service,'SPILL_AFTER_S',.1):
            first=self.call_('/first');self.rt.wait_started('engine:/first');c=self.call_('/c')
            time.sleep(.4);self.assertNotIn('/c',self.rt.engine_on,'gpu1 is busy with the tenant itself')
            held.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c'],'gpu1');self.wait(first)

    def test_E6_a_tenant_already_on_the_fallback_is_taken_there(self):
        self.rt.where['tr']='gpu0'
        gate,busy=self.busy_gpu1();c=self.call_('/c')
        self.assertEqual(self.wait(c)['status'],'completed');self.assertEqual(self.rt.engine_on['/c'],'gpu0')
        gate.set();self.wait(busy)

    def test_an_idle_own_device_keeps_its_call(self):
        self.rt.where['tr']='gpu0';self.wait(self.call_('/c'));self.assertEqual(self.rt.engine_on['/c'],'gpu1')

    def test_E14_E15_never_onto_a_drained_lent_or_blocked_device(self):
        for how in ('drained','lent','blocked'):
            with self.subTest(how=how):
                if how=='drained':self.call('device_drain',device='gpu0')
                elif how=='lent':self.rt.lent['gpu0']={'owner':'lab'}
                else:self.rt.blocked['gpu0']='not ours'
                gate,busy=self.busy_gpu1();c=self.call_('/c-'+how)
                time.sleep(.2);self.assertNotIn('/c-'+how,self.rt.engine_on)
                gate.set();self.wait(c);self.assertEqual(self.rt.engine_on['/c-'+how],'gpu1');self.wait(busy)
                if how=='drained':self.call('device_drain',device='gpu0',drain=False)
                self.rt.lent.pop('gpu0',None);self.rt.blocked.pop('gpu0',None);self.rt.gates.pop('engine:/busy');self.rt.started.pop('engine:/busy')
                with self.s.db:self.s.db.execute("DELETE FROM jobs WHERE request_key='busy'")

    def test_E7_E8_a_spilled_call_is_not_moved_back(self):
        held=self.rt.gate('engine:/c');gate,busy=self.busy_gpu1();c=self.call_('/c');self.rt.wait_started('engine:/c')
        gate.set();self.wait(busy)   # gpu1 idle now, and a lend state change runs _relane
        self.rt.lent['gpu9']={};self.s.relaned=None
        with self.s.cv:self.s._relane()
        held.set();self.assertEqual(self.wait(c)['status'],'completed');self.assertEqual(self.job(c)['lane'],'gpu0')
        self.assertEqual(self.rt.engine_on['/c'],'gpu0')

    def test_E4_E5_a_spilled_tenant_lingers_on_the_fallback_but_jobs_still_run(self):
        self.rt.tenants['tr']['linger_s']=.6
        gate,busy=self.busy_gpu1();c=self.call_('/c');self.wait(c);self.rt.parked.clear()
        time.sleep(.3);self.assertNotIn('gpu0',self.rt.parked,'the DiT is not put back under the lingering call')
        x=self.render('x',);gate.set()
        self.assertEqual(self.wait(x)['status'],'completed','a render still samples on gpu0 meanwhile')
        self.wait(busy)


class SpillPlacementTests(base.SchedulerHarness):
    """E19, E20 on the real scheduler: an lru fallback card holding an undeclared resident plugin part."""
    def setUp(self):
        super().setUp()
        self.actual={'translator':19,'audio':3,'pair.back':50,'sd':15}
        def vram(device):
            if device.name!='gpu1':return (80*GIB,0)
            return (64*GIB,sum(self.actual.get(n,0) for n in self.s._occupants(device))*GIB)
        for p in (patch.object(Device,'vram',vram),patch.object(gpu_runtime,'VRAM_SETTLE_S',0)):p.start();self.addCleanup(p.stop)
        self.s=self.scheduler(base.v2(**{'tenants.pair':PAIR,'devices.gpu1.placement':'lru','tenants.translator.vram_gib':19,'tenants.audio.vram_gib':3,
                                         'tenants.sd.vram_gib':15,'tenants.translator.fallback':'gpu1','tenants.audio.fallback':'gpu1',
                                         'tenants.translator.api':'vulkan','tenants.audio.api':'vulkan'}))
    def on(self):return sorted(self.occupants(self.s)['gpu1'])

    def test_E20_nothing_is_placed_beside_an_undeclared_resident(self):
        self.s.plugins['pair'].start('back');self.s._server('audio','gpu1')
        self.assertEqual(self.on(),['audio'],'50 + 3 would fit at rest, but an undeclared part may grow to the card')

    def test_E19_E20_the_undeclared_goes_first_then_the_pair_shares_the_card(self):
        self.s._server('sd','gpu1');self.s.plugins['pair'].start('back')   # sd is older but declared, the part newer but undeclared
        self.assertEqual(self.on(),['pair.back'])
        self.s._server('translator','gpu1');self.s._server('audio','gpu1')
        self.assertEqual(self.on(),['audio','translator'],'the spilled two side by side once the undeclared part left')

    def test_spill_is_declared_per_tenant_and_never_on_a_plugins_stages(self):
        s=self.scheduler(base.v2(**{'tenants.audio.spill':True,'tenants.audio.fallback':'gpu1'}))
        self.assertTrue(s.spills('engine',{'engine':'audio'}));self.assertFalse(s.spills('engine',{'engine':'sd'}))
        self.assertFalse(s.spills('pair',{}))
        path=self.state/'gpu-config.json'
        for bad,message in ((base.v2(**{'tenants.audio.spill':'yes'}),'spill must be true or false'),):
            path.write_text(json.dumps(bad))
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):load_config(path)
        s._server('audio','gpu1');self.assertEqual(s.located('audio'),'gpu1');self.assertIsNone(s.located('sd'))


import job_service
from app_fixture import Service
from test_device_queues import ROOT

if __name__=='__main__':unittest.main()
