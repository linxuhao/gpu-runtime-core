"""Per-device modes: `placement` exclusive / lru (side by side while the card has room for what
the newcomer declares; visitors out before residents, least recently used first), `queue` priority /
fifo / affinity, drain, and a
tenant in use on one device not moved by another (invariant 2, E1)."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gpu_runtime
from gpu_runtime import GIB, Device, load_config
from app_fixture import Service
from test_device_queues import FAST, LaneRuntime, QueueHarness, ROOT
import harness as base
from pair_tenant import PAIR


class LruPlacementTests(base.SchedulerHarness):
    """A fake 24 GiB gpu0 whose used memory is what its resident tenants really take. Its idle
    residency is translator + audio (standing in for the translator + ASR pair); sd is a visitor."""
    def setUp(self):
        super().setUp()
        self.actual={'translator':19,'audio':3,'sd':15,'pair.front':20};self.foreign=0
        def vram(device):
            if device.name!='gpu0':return (80*GIB,0)
            return (24*GIB,(sum(self.actual.get(n,0) for n in self.s._occupants(device))+self.foreign)*GIB)
        for p in (patch.object(Device,'vram',vram),patch.object(gpu_runtime,'VRAM_SETTLE_S',0)):p.start();self.addCleanup(p.stop)
        self.s=self.scheduler(base.v2(**{'tenants.pair':PAIR,'devices.gpu0.placement':'lru','tenants.translator.vram_gib':19,
                                         'tenants.audio.vram_gib':3,'tenants.sd.vram_gib':15,'policy.idle.restore':['translator','audio','pair']}))
    def on(self):return sorted(self.occupants(self.s)['gpu0'])

    def test_tenants_that_fit_stay_side_by_side(self):
        translator=self.s._server('translator');self.s._server('audio')
        self.assertFalse(translator.stopped);self.assertEqual(self.on(),['audio','translator'])

    def test_a_newcomer_that_does_not_fit_evicts_the_least_recently_used_resident(self):
        translator=self.s._server('translator');audio=self.s._server('audio')
        self.s._server('sd')
        self.assertTrue(translator.stopped);self.assertFalse(audio.stopped,'room once the least recently used left');self.assertEqual(self.on(),['audio','sd'])

    def test_a_visitor_leaves_before_a_resident(self):
        """Seen live 2026-10-12: a TTS call between utterances, then the translator coming back,
        restarted the ASR (oldest placed) instead of sending the TTS (the visitor) away."""
        self.s._server('translator');audio=self.s._server('audio');sd=self.s._server('sd')
        self.s._server('translator')
        self.assertTrue(sd.stopped);self.assertFalse(audio.stopped,'the pair is kept, the visitor goes');self.assertEqual(self.on(),['audio','translator'])

    def test_use_not_placement_decides_among_residents(self):
        self.actual['sd']=2;self.s.tenants['sd']['vram_gib']=2   # 19 + 3 + 2 + margin is just over 24
        translator=self.s._server('translator');audio=self.s._server('audio')
        with self.s._using('translator','gpu0'):pass   # the translator was used after the ASR stand-in
        self.s._server('sd');self.assertTrue(audio.stopped);self.assertFalse(translator.stopped)

    def test_a_tenant_declaring_nothing_is_placed_as_on_an_exclusive_card(self):
        self.s._server('translator');self.s._server('audio');self.s.plugins['pair'].start('front')
        self.assertEqual(self.on(),['pair.front'],'a plugin part declaring nothing, likewise')

    def test_a_resident_counts_with_its_declared_peak_until_it_reaches_it(self):
        self.actual['translator']=5   # loaded, compute buffers not grown yet
        translator=self.s._server('translator');self.s._server('audio');self.s._server('sd')
        self.assertTrue(translator.stopped,'19 + 3 declared leave no room for 15, whatever is in use now')

    def test_what_does_not_fit_even_alone_is_refused_with_the_numbers(self):
        self.foreign=10
        with self.assertRaisesRegex(RuntimeError,r'gpu0: 14.0 GiB free, translator needs 19.0 GiB \(\+0.5 margin\)'):self.s._server('translator')

    def test_lend_room_counts_what_shared_lends_declare_or_use_whichever_is_more(self):
        self.foreign=10   # the shared lends on the card, as measured
        self.assertTrue(self.s.lend_room('gpu0',13,8),'24 - max(10 used, 8 declared) - 0.5 margin leaves 13.5')
        self.assertFalse(self.s.lend_room('gpu0',13,12),'12 declared leave 11.5')
        self.foreign=0;self.assertFalse(self.s.lend_room('gpu0',24,0),'the margin stays free')

    def test_an_exclusive_card_is_unchanged(self):
        s=self.scheduler(base.v2(**{'tenants.translator.vram_gib':19,'tenants.audio.vram_gib':3}))
        translator=s._server('translator');s._server('audio')
        self.assertTrue(translator.stopped);self.assertEqual(self.occupants(s)['gpu0'],['audio'])

    def test_config_names_the_mistake(self):
        path=self.state/'gpu-config.json'
        for bad,message in ((base.v2(**{'devices.gpu0.sharing':'shared'}),'"sharing" is replaced by "placement"'),
                            (base.v2(**{'devices.gpu0.placement':'fifo'}),'placement must be exclusive or lru'),
                            (base.v2(**{'devices.gpu0.queue':'random'}),'queue must be priority, fifo or affinity'),
                            (base.v2(**{'tenants.audio.vram_gib':0}),'vram_gib must be a number above 0'),
                            (base.v2(**{'tenants.pair':PAIR,'tenants.pair.back.vram_gib':'50'}),'back.vram_gib must be a number above 0')):
            path.write_text(json.dumps(bad))
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):load_config(path)


class QueueModeTests(unittest.TestCase):
    """_next under each queue mode, on rows written straight into the table."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.rt=LaneRuntime();self.loaded=set()
        self.rt.loaded_on=lambda lane:set(self.loaded)
        self.s=Service(ROOT,Path(self.temp.name),self.rt,start_worker=False);self.n=0
    def tearDown(self):self.s.close();self.temp.cleanup()
    def mode(self,queue):self.rt.devices['gpu1']=SimpleNamespace(queue=queue)
    def add(self,ident,kind,step,engine=None,urgent=0):
        self.n+=1;req=json.dumps({'engine':engine} if engine else {})
        with self.s.db:self.s.db.execute("INSERT INTO jobs(id,project,kind,status,request,created,updated,request_key,request_hash,lane,step,waiting,urgent)"
                                         " VALUES(?,'',?,'queued',?,?,?,?,'h','gpu1',?,1,?)",(ident,kind,req,self.n,self.n,ident,step,urgent))
    def take(self):
        row=self.s._next('gpu1')
        if row is None:return None
        with self.s.db:self.s.db.execute('UPDATE jobs SET waiting=0 WHERE id=?',(row['id'],))
        self.s.passed.pop(row['id'],None);return row['id']

    def test_fifo_is_creation_order_only(self):
        self.mode('fifo');self.add('r','render','encode');self.add('x','engine','engine','audio',urgent=1)
        self.assertEqual([self.take(),self.take()],['r','x'],'the engine call does not go first')

    def test_priority_is_unchanged(self):
        self.mode('priority');self.add('r','render','encode');self.add('x','engine','engine','audio',urgent=1);self.loaded={'translator'}
        self.add('t','engine','engine','translator',urgent=1)
        self.assertEqual([self.take(),self.take(),self.take()],['x','t','r'])

    def test_affinity_serves_what_is_loaded_first_within_its_tier(self):
        self.mode('affinity');self.loaded={'translator','asr'}
        self.add('a','engine','engine','audio',urgent=1);self.add('t','engine','engine','translator',urgent=1)
        self.add('s','engine','engine','asr',urgent=1);self.add('r','render','encode')
        self.assertEqual([self.take(),self.take(),self.take(),self.take()],['t','s','a','r'],'the render, a lower tier, is not pulled ahead')

    def test_affinity_passes_over_the_oldest_at_most_the_limit(self):
        self.mode('affinity');self.loaded={'translator'};self.add('a','engine','engine','audio',urgent=1)
        for i in range(5):self.add('t%d'%i,'engine','engine','translator',urgent=1)
        self.assertEqual([self.take() for _ in range(6)],['t0','t1','t2','a','t3','t4'])

    def test_affinity_never_jumps_a_lend(self):
        self.mode('affinity');self.loaded={'pair.front'};self.add('l','lend','lend');self.add('e','render','encode')
        self.assertEqual(self.take(),'l')


class DrainTests(QueueHarness):
    def test_a_drained_device_finishes_its_queue_takes_nothing_new_and_is_emptied(self):
        held=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        b=self.render('b');self.until(lambda:self.between(b),what='b waits for gpu0')
        self.assertTrue(self.call('device_drain',device='gpu0')['drained'])
        with self.assertRaisesRegex(ValueError,'gpu0 is drained for maintenance'):self.render('c')
        with self.assertRaisesRegex(ValueError,'gpu0 is drained'):self.lend('l',device='gpu0')
        self.assertEqual(self.wait(self.engine('e'))['status'],'completed','gpu1 is not drained')
        self.rt.parked.clear();held.set()
        for job in (a,b):self.assertEqual(self.wait(job)['status'],'completed','what was queued finishes')
        self.until(lambda:'gpu0' in self.rt.stopped,what='gpu0 emptied')
        time.sleep(.2);self.assertNotIn('gpu0',self.rt.parked,'no idle restore while drained')
        self.assertTrue(self.call('resource_status')['lanes']['gpu0']['drained'])
        self.call('device_drain',device='gpu0',drain=False)
        self.until(lambda:'gpu0' in self.rt.parked,what='idle residency back')
        self.assertEqual(self.wait(self.render('c'))['status'],'completed')

    def test_drained_stays_drained_across_a_restart(self):
        self.call('device_drain',device='gpu1')
        self.s.close();self.s=Service(ROOT,self.state,self.rt)
        with self.assertRaisesRegex(ValueError,'gpu1 is drained'):self.engine('e')
        with self.assertRaisesRegex(ValueError,'unknown device'):self.call('device_drain',device='gpu7')

    def test_new_work_with_a_fallback_goes_there(self):
        s=base.SchedulerHarness('run');s.setUp();self.addCleanup(s.tearDown)
        sched=s.scheduler(base.v2(**{'tenants.audio.fallback':'gpu1'}));sched.drained={'gpu0'}
        self.assertEqual(sched.stages('engine',{'engine':'audio'}),[('engine','gpu1')])


class InUseTests(base.SchedulerHarness):
    """Invariant 2 (E1): translator's home is gpu0, it lives on its fallback gpu1, and a call there
    is not cut short by gpu0's idle restore or by another queue placing it on gpu0."""
    def setUp(self):
        super().setUp()
        self.s=self.scheduler(base.v2(**{'tenants.translator.fallback':'gpu1','tenants.pair':PAIR,'tenants.pair.front.fallback':'gpu1'}))
        self.away=self.s._server('translator','gpu1')
    def session(self,hold):
        """A call to translator on gpu1 in another thread, in use until `hold` is done."""
        entered=threading.Event()
        def use():
            with self.s.engine_session('translator','gpu1'):entered.set();hold()
        thread=threading.Thread(target=use);thread.start();self.assertTrue(entered.wait(5));return thread

    def test_idle_restore_leaves_a_tenant_in_use_elsewhere_and_brings_it_home_later(self):
        done=threading.Event();thread=self.session(lambda:done.wait(5))
        self.s.park('gpu0')
        self.assertFalse(self.away.stopped,'the call on gpu1 goes on');self.assertEqual(self.s.server_on['translator'],'gpu1')
        self.assertEqual(self.s.status()['in_use'],['translator'])
        done.set();thread.join(5)
        self.assertEqual(self.s.status()['in_use'],[])
        self.s.park('gpu0')
        self.assertTrue(self.away.stopped);self.assertEqual(self.occupants(self.s),{'gpu0':['translator'],'gpu1':[]})

    def test_placing_it_on_another_device_waits_until_the_call_is_done(self):
        order=[]
        def hold():time.sleep(.2);order.append(('call done',self.away.stopped))
        thread=self.session(hold)
        moved=self.s._server('translator','gpu0');order.append(('moved',self.away.stopped));thread.join(5)
        self.assertEqual(order,[('call done',False),('moved',True)]);self.assertEqual(self.s.server_on['translator'],'gpu0')
        self.assertIsNot(moved,self.away)

    def test_the_wait_is_bounded_and_says_why(self):
        with patch.object(gpu_runtime,'TENANT_BUSY_WAIT_S',.1),self.s._using('translator','gpu1'):
            with self.assertRaisesRegex(RuntimeError,'translator is in use on gpu1; not moved to gpu0 after waiting'):self.s._server('translator','gpu0')
        self.assertFalse(self.away.stopped);self.assertEqual(self.s.server_on['translator'],'gpu1')

    def test_a_plugin_part_in_use_on_its_fallback_is_not_moved_either(self):
        pair=self.s.plugins['pair'];front=pair.start('front','gpu1')
        with patch.object(gpu_runtime,'TENANT_BUSY_WAIT_S',.1),self.s._using('pair.front','gpu1'):
            with self.assertRaisesRegex(RuntimeError,'pair.front is in use on gpu1'):pair.start('front','gpu0')
        self.assertFalse(front.stopped)
        pair.start('front','gpu0');self.assertTrue(front.stopped);self.assertEqual(self.s.located('pair.front'),'gpu0')

    def test_using_it_where_it_lives_never_waits(self):
        with patch.object(gpu_runtime,'TENANT_BUSY_WAIT_S',0):
            with self.s.engine_session('translator','gpu1') as (first,_):pass
            with self.s.engine_session('translator','gpu1') as (second,_):pass
            self.assertIs(first,self.away);self.assertIs(second,self.away)
            with self.s._using('translator','gpu1'):
                self.assertIs(self.s._server('translator','gpu1'),self.away)
                self.s._server('audio','gpu1')   # the device's own queue: its residents are evicted as before
            self.assertTrue(self.away.stopped);self.assertEqual(self.occupants(self.s)['gpu1'],['audio'])


if __name__=='__main__':unittest.main()


class StartRaceTests(base.SchedulerHarness):
    """E22: two device queues wanting the same stopped tenant at once start one process, not two."""
    def setUp(self):
        super().setUp();self.live=[];self.most=[0];self.fail=[False];lock=threading.Lock();test=self
        class SlowServer(base.FakeServer):
            def __init__(self,*a,**k):
                with lock:test.live.append(self);test.most[0]=max(test.most[0],len(test.live))
                time.sleep(.3)
                if test.fail[0]:
                    with lock:test.live.remove(self)
                    raise RuntimeError('engine exited')
                super().__init__(*a,**k)
            def stop(self):
                with lock:
                    if self in test.live:test.live.remove(self)
                super().stop()
        p=patch.object(base.Scheduler,'server_worker',SlowServer);p.start();self.addCleanup(p.stop)
        self.s=self.scheduler(base.v2(**{'tenants.translator.fallback':'gpu1'}))

    def test_two_queues_starting_the_same_tenant_start_it_once_at_a_time(self):
        ends={}
        def start(lane):self.s._server('translator',lane);ends[lane]=time.monotonic()
        threads=[threading.Thread(target=start,args=(lane,)) for lane in ('gpu0','gpu1')]
        for t in threads:t.start();time.sleep(.05)
        for t in threads:t.join(5)
        self.assertEqual(self.most[0],1,'never two translator processes at once')
        self.assertEqual(len(self.live),1);self.assertEqual(self.s.server_on['translator'],'gpu1','the second queue moved it once the first start ended')

    def test_the_idle_restore_skips_a_tenant_another_queue_is_starting(self):
        t=threading.Thread(target=self.s._server,args=('translator','gpu1'));t.start();time.sleep(.1)
        began=time.monotonic();self.assertIsNone(self.s._server('translator','gpu0',wait=False))
        self.assertLess(time.monotonic()-began,.2,'did not wait for the start');t.join(5)
        self.assertEqual(self.s.server_on['translator'],'gpu1')

    def test_a_failed_start_lets_the_next_one_go(self):
        self.fail[0]=True
        with self.assertRaisesRegex(RuntimeError,'engine exited'):self.s._server('translator','gpu0')
        self.assertEqual(self.s.loading,{})
        self.fail[0]=False;self.s._server('translator','gpu1');self.assertEqual(self.s.server_on['translator'],'gpu1')
