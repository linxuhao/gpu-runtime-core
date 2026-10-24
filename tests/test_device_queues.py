"""One queue per device: a job waits in one device's queue at a time, a render is two stages
(encode, then sample), nothing holds one device while it waits for another, and a device can be
lent to research in queued, shared or exclusive mode. Service behaviour runs against a two-device fake
runtime (renders: app_fixture), and the borrow CLI against a real service socket."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app_fixture import Service
import job_service

ROOT=Path(__file__).resolve().parents[1]
ORIGIN={'project':'project-fixture','shot':'shot-fixture','shot_revision':1}
FAST={'IDLE_COALESCE_S':.05,'BLOCKED_RECHECK_S':.05,'LEND_SWEEP_S':.05}


class LaneRuntime:
    """Renders encode on gpu1 and sample on gpu0, engines run on gpu1. A step named in `gates`
    waits for that event; one named in `failures` raises after it ran. With `fallback` set, encodes
    go there while gpu1 is lent or blocked (as Scheduler.encoder_lane); `encoded_on` records the
    device each encode was handed, `loaded` which stage last ran on each device."""
    ready=None
    python=sys.executable
    def __init__(self):
        self.devices={'gpu0':None,'gpu1':None};self.lanes=['gpu0','gpu1'];self.tenants={}
        self.blocked={};self.lent={};self.gates={};self.started={};self.failures={}
        self.lock=threading.Lock();self.log=[];self.live={'gpu0':0,'gpu1':0};self.max_live={'gpu0':0,'gpu1':0}
        self.stopped=[];self.parked=[];self.lends=[];self.reclaimed=[];self.foreign={}
        self.fallback=None;self.engine_fallback=None;self.encoded_on={};self.engine_on={};self.loaded={};self.restored=[]
        self.vram={'gpu0':64,'gpu1':24};self.measured=[]   # GiB free on an emptied card; lend_room calls
    def lane_for(self,fallback):
        down=lambda d:d in self.lent or d in self.blocked
        return fallback if fallback and down('gpu1') and not down(fallback) else 'gpu1'
    def encoder_lane(self):return self.lane_for(self.fallback)
    def stage_devices(self,kind,req):
        if kind=='render':return 'gpu1',self.fallback
        if kind=='engine':return 'gpu1',self.engine_fallback
        return req.get('device'),None
    def resident(self,device):return self.loaded.get(device)
    def stages(self,kind,req):
        if kind=='render':return [('encode',self.encoder_lane()),('sample','gpu0')]
        if kind=='engine':return [('engine',self.lane_for(self.engine_fallback))]
        if req.get('device') not in self.devices:raise ValueError('unknown device %r'%req.get('device'))
        return [('lend',req['device'])]
    def gate(self,name):
        self.gates[name]=threading.Event();return self.gates[name]
    def wait_started(self,name,timeout=5):
        deadline=time.monotonic()+timeout
        while name not in self.started:
            if time.monotonic()>deadline:raise AssertionError(name+' never started')
            time.sleep(.01)
    def _step(self,device,name):
        with self.lock:
            self.live[device]+=1;self.max_live[device]=max(self.max_live[device],self.live[device]);self.log.append(('start',name))
        self.started[name]=time.monotonic()
        try:
            if name in self.gates and not self.gates[name].wait(10):raise RuntimeError('fixture timeout')
        finally:
            with self.lock:self.live[device]-=1;self.log.append(('end',name))
        if name in self.failures:raise RuntimeError(self.failures[name])
    def render_encode(self,req,out,device):
        self.encoded_on[req['prompt']]=device;self.loaded[device]='encode'
        self._step(device,'encode:'+req['prompt']);out.mkdir(parents=True);(out/'encoded.json').write_text(json.dumps(req))
    def render_sample(self,out):
        req=json.loads((out/'encoded.json').read_text());self.loaded['gpu0']='sample';self._step('gpu0','sample:'+req['prompt'])
        p=out/'result.mp4';p.write_bytes(b'fixture-'+req['prompt'].encode());return {'output':str(p)}
    def run_stage(self,kind,stage,req,out,lane,cancelled=None):
        return self.render_encode(req,out,lane) if stage=='encode' else self.render_sample(out)
    def part_of(self,kind,stage):return {'encode':'pair.front','sample':'pair.back'}[stage]
    def engine(self,req,out,lane='gpu1'):
        self.engine_on[req['path']]=lane
        self._step(lane,'engine:'+req['path']);out.mkdir(parents=True);return {'status':200}
    def park(self,device=None):self.parked.append(device)
    def stop_device(self,device):self.stopped.append(device)
    def stop(self):self.stopped.append('all')
    def check_device(self,device):
        found=list(self.foreign.get(device,[]))
        if found:self.blocked[device]='not ours: '+', '.join(found)
        else:self.blocked.pop(device,None)
        return found
    def lend(self,device,record):
        self.log.append(('start','lend:'+device));self.lends.append((device,record));self.lent[device]=record
        return {'CUDA_VISIBLE_DEVICES':'GPU-'+device}
    def restore_lend(self,device,record):
        self.restored.append((device,record));self.lent[device]=record
        return {'CUDA_VISIBLE_DEVICES':'GPU-'+device}
    def lend_room(self,device,need,declared):
        self.measured.append((device,need,declared));return declared+need<=self.vram[device]
    def reclaim(self,device):
        self.lent.pop(device,None);self.reclaimed.append(device);return self.check_device(device)


class QueueHarness(unittest.TestCase):
    def setUp(self):
        self.patches=[patch.multiple(job_service,**FAST)]
        for p in self.patches:p.start()
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name);self.rt=LaneRuntime()
        self.s=Service(ROOT,self.state,self.rt)
    def tearDown(self):
        for gate in self.rt.gates.values():gate.set()
        for lend in [*self.s.lends.values(),*(l for lends in self.s.shared.values() for l in lends.values())]:lend['returned']=True
        with self.s.cv:self.s.cv.notify_all()
        self.s.close();self.temp.cleanup()
        for p in self.patches:p.stop()
    def call(self,action,**args):return self.s.call(action,args)
    def render(self,key,prompt=None,**kw):
        return self.call('render',request={'prompt':prompt or key,'origin':ORIGIN,**kw},request_key=key)
    def engine(self,key,path='/v1/audio/transcriptions'):
        return self.call('engine_submit',engine='audio',path=path,method='POST',body='',content_type='audio/wav',request_key=key)
    def lend(self,key,device='gpu0',mode='queued',hours=1,owner='lab',**extra):
        return self.call('lend_submit',device=device,mode=mode,hours=hours,owner=owner,request_key=key,**extra)
    def share(self,key,vram_gib,**extra):return self.lend(key,mode='shared',vram_gib=vram_gib,**extra)
    def job(self,j):return self.call('job_get',job_id=j['id'])
    def wait(self,j,timeout=5):return self.call('job_wait',job_id=j['id'],timeout=timeout)
    def until(self,predicate,timeout=5,what='condition'):
        deadline=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>deadline:self.fail(what+' never held')
            time.sleep(.01)
    def between(self,j):
        """Encoded, back in a queue, waiting for the DiT's device."""
        job=self.job(j);return (job['status'],job['step'],job['lane'],job['waiting'])==('running','sample','gpu0',1)
    def granted(self,j):
        return self.call('lend_get',job_id=j['id'])['granted']
    def order(self,a,b):
        names=[n for e,n in self.rt.log if e=='start']
        return names.index(a)<names.index(b)


class LaneTests(QueueHarness):
    def test_the_two_devices_work_at_the_same_time_and_each_alone(self):
        sampling=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        e1=self.engine('e1');e2=self.engine('e2',path='/health')
        self.assertEqual(self.wait(e1)['status'],'completed');self.assertEqual(self.wait(e2)['status'],'completed')
        running=self.job(a);self.assertEqual((running['status'],running['step']),('running','sample'),'the render still samples')
        sampling.set();self.assertEqual(self.wait(a)['status'],'completed')
        self.assertEqual(self.rt.max_live,{'gpu0':1,'gpu1':1},'one thing at a time on each device')

    def test_a_render_lets_go_of_the_encoder_device_while_it_waits_for_the_dit(self):
        held=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        b=self.render('b')
        self.until(lambda:('end','encode:b') in self.rt.log,what='b encoded')
        waiting=self.job(b)
        self.assertEqual((waiting['status'],waiting['step'],waiting['lane'],waiting['waiting']),('running','sample','gpu0',1))
        self.assertEqual(self.wait(self.engine('e'))['status'],'completed','gpu1 serves while b waits for gpu0')
        held.set();self.assertEqual(self.wait(b)['status'],'completed')
        self.assertTrue(self.order('sample:a','sample:b'))
        self.assertTrue(self.order('encode:b','engine:/v1/audio/transcriptions'),'b encoded before the engine call')

    def test_the_second_stage_keeps_the_place_its_job_had(self):
        x_held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        a_enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        later=self.lend('later')                       # queued on gpu0 while a was still encoding
        a_enc.set();self.until(lambda:self.between(a),what='a between stages')
        x_held.set()
        self.assertEqual(self.wait(a)['status'],'completed')
        self.until(lambda:self.granted(later) is not None,what='the lend granted')
        self.assertTrue(self.order('sample:a','lend:gpu0'),'a was created first, so its sampling went before the lend')
        self.call('lend_return',job_id=later['id']);self.assertEqual(self.wait(later)['status'],'completed')

    def test_engine_calls_still_go_first_within_their_device(self):
        enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        b=self.render('b');e=self.engine('e')
        enc.set();self.assertEqual(self.wait(b)['status'],'completed');self.assertEqual(self.wait(e)['status'],'completed')
        self.assertTrue(self.order('engine:/v1/audio/transcriptions','encode:b'),'the engine call precedes the queued encode')
        self.wait(a)

    def test_cancelled_queued_encoding_between_stages_and_sampling(self):
        busy=self.rt.gate('encode:busy');first=self.render('busy');self.rt.wait_started('encode:busy')
        queued=self.render('queued')
        self.assertEqual(self.call('job_cancel',job_id=queued['id'])['status'],'cancelled')
        busy.set();self.wait(first)
        enc=self.rt.gate('encode:c');c=self.render('c');self.rt.wait_started('encode:c')
        self.assertEqual(self.call('job_cancel',job_id=c['id'])['status'],'cancelling')
        enc.set();self.assertEqual(self.wait(c)['status'],'cancelled')
        self.assertIn('gpu1',self.rt.stopped)
        held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        d=self.render('d');self.until(lambda:self.between(d),what='d between stages')
        self.assertEqual(self.call('job_cancel',job_id=d['id'])['status'],'cancelled','between stages it runs nowhere: it ends at once')
        held.set();self.wait(x)
        sampling=self.rt.gate('sample:e');e=self.render('e');self.rt.wait_started('sample:e')
        self.call('job_cancel',job_id=e['id']);sampling.set()
        self.assertEqual(self.wait(e)['status'],'cancelled');self.assertIn('gpu0',self.rt.stopped)
        started=[n for kind,n in self.rt.log if kind=='start']
        for name in ('encode:queued','sample:queued','sample:c','sample:d'):self.assertNotIn(name,started)
        for job in (queued,c,d,e):self.assertEqual(self.job(job)['status'],'cancelled','not selectable')
        for job in (queued,c,d):self.assertIsNone(self.job(job)['result'],'never sampled')

    def test_a_failure_stops_only_its_own_device(self):
        held=self.rt.gate('sample:g');g=self.render('g');self.rt.wait_started('sample:g')
        self.rt.failures['encode:f']='encoder fell over';f=self.render('f')
        failed=self.wait(f);self.assertEqual(failed['status'],'failed');self.assertIn('encoder fell over',failed['error'])
        self.assertEqual(self.rt.stopped,['gpu1'])
        held.set();self.assertEqual(self.wait(g)['status'],'completed')

    def test_each_device_gets_its_idle_residency_back_on_its_own(self):
        held=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        self.rt.parked.clear();self.wait(self.engine('e'))
        self.until(lambda:'gpu1' in self.rt.parked,what='gpu1 restored')
        self.assertNotIn('gpu0',self.rt.parked,'gpu0 is still busy')
        held.set();self.wait(a);self.until(lambda:'gpu0' in self.rt.parked,what='gpu0 restored')

    def test_status_keeps_its_fields_and_adds_the_queues(self):
        held=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        status=self.call('resource_status')
        self.assertLessEqual({'mode','draining','active_job','workers','queued','cancellation'},set(status))
        self.assertEqual(status['active_job'],a['id'])
        self.assertEqual(status['lanes']['gpu0']['active'],a['id']);self.assertEqual(status['lanes']['gpu0']['step'],'sample')
        held.set();self.wait(a)

    def test_status_lists_who_runs_and_who_waits_without_request_bodies(self):
        sampling=self.rt.gate('sample:a');a=self.render('a');self.rt.wait_started('sample:a')
        encoding=self.rt.gate('encode:b');b=self.render('b');self.rt.wait_started('encode:b')
        c=self.render('c');e=self.engine('e');lend=self.lend('l',owner='lab-7',label=' kl check ')
        lanes=self.call('resource_status')['lanes']
        running=lanes['gpu0']['running']
        self.assertEqual((running['id'],running['kind'],running['step'],running['project'],running['shot']),(a['id'],'render','sample','project-fixture','shot-fixture'))
        self.assertNotIn('request',running)
        self.assertEqual([(j['id'],j.get('owner'),j.get('label'),j.get('mode')) for j in lanes['gpu0']['waiting']],[(lend['id'],'lab-7','kl check','queued')])
        self.assertEqual([j['id'] for j in lanes['gpu1']['waiting']],[e['id'],c['id']],'engine calls first, then by creation')
        self.assertEqual(lanes['gpu1']['waiting'][0]['engine'],'audio')
        self.assertEqual(lanes['gpu1']['running']['id'],b['id'])
        self.assertIsNone(lanes['gpu0']['device'],'the fake runtime declares no hardware')
        self.call('lend_return',job_id=lend['id']);sampling.set();encoding.set()
        for job in (a,b,c,e):self.assertEqual(self.wait(job)['status'],'completed')


class RestartTests(unittest.TestCase):
    def test_a_render_between_stages_is_interrupted_and_old_queued_rows_are_placed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)
            db=sqlite3.connect(state/Service.DB_NAME)       # the shape before queues per device
            db.executescript('''CREATE TABLE jobs(id TEXT PRIMARY KEY, project TEXT NOT NULL, shot TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
              request TEXT NOT NULL, result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL, request_key TEXT UNIQUE, request_hash TEXT NOT NULL);''')
            db.execute("INSERT INTO jobs VALUES('take-old','p',NULL,'render','queued','{\"prompt\":\"old\"}',NULL,NULL,1,1,'old','h')")
            db.commit();db.close()
            s=Service(ROOT,state,LaneRuntime(),start_worker=False)
            try:
                old=s.call('job_get',{'job_id':'take-old'})
                self.assertEqual((old['status'],old['lane'],old['step'],old['waiting']),('queued','gpu1','encode',1))
                with self.assertRaisesRegex(ValueError,'queued jobs'):s.call('resource_release',{})
                with s.db:s.db.execute("UPDATE jobs SET status='running',lane='gpu0',step='sample',waiting=1 WHERE id='take-old'")
            finally:s.close()
            s=Service(ROOT,state,LaneRuntime(),start_worker=False)
            try:
                old=s.call('job_get',{'job_id':'take-old'})
                self.assertEqual((old['status'],old['waiting']),('interrupted',0))
            finally:s.close()


class LendRestartTests(unittest.TestCase):
    """A granted lend survives a restart while its deadline is ahead: it is lent again and its
    queue resumes holding it. Past its deadline it is interrupted as before."""
    def setUp(self):
        self.patches=[patch.multiple(job_service,**FAST)]
        for p in self.patches:p.start()
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
        for p in self.patches:p.stop()
    def crashed_with(self,*lends):
        """The database a service leaves when it dies holding these lends (no clean return)."""
        s=Service(ROOT,self.state,LaneRuntime(),start_worker=False)
        try:
            for ident,device,mode,deadline in lends:
                record={'job_id':ident,'device':device,'owner':'trainer','mode':mode,'granted':time.time()-60,'deadline':deadline}
                with s.db:s.db.execute("INSERT INTO jobs(id,project,kind,status,request,created,updated,request_key,request_hash,lane,step,waiting,lent)"
                                       " VALUES(?,'','lend','running',?,1,1,?,'h',?,'lend',0,?)",
                                       (ident,json.dumps({'device':device,'mode':mode,'hours':1,'owner':'trainer'}),ident,device,json.dumps(record)))
        finally:s.close()
    def test_a_lend_within_its_deadline_is_restored_and_resumed(self):
        self.crashed_with(('lend-live','gpu1','exclusive',time.time()+3600),('lend-old','gpu0','queued',time.time()-1))
        rt=LaneRuntime();rt.fallback='gpu0';s=Service(ROOT,self.state,rt)
        try:
            live=s.call('lend_get',{'job_id':'lend-live'})
            self.assertEqual((live['job']['status'],live['granted']['owner'],live['granted']['env']),('running','trainer',{'CUDA_VISIBLE_DEVICES':'GPU-gpu1'}))
            self.assertEqual([d for d,_ in rt.restored],['gpu1']);self.assertIn('gpu1',rt.lent);self.assertEqual(rt.lends,[],'restored, not lent anew: nothing stopped')
            old=s.call('job_get',{'job_id':'lend-old'});self.assertEqual(old['status'],'interrupted');self.assertNotIn('gpu0',rt.lent)
            r=s.call('render',{'request':{'prompt':'r','origin':ORIGIN},'request_key':'r'})
            self.assertEqual(r['lane'],'gpu0','the reservation stands: the render encodes on the fallback')
            self.assertEqual(s.call('job_wait',{'job_id':r['id'],'timeout':5})['status'],'completed')
            rt.fallback=None
            with self.assertRaisesRegex(ValueError,'gpu1 is reserved for research by trainer'):
                s.call('render',{'request':{'prompt':'q','origin':ORIGIN},'request_key':'q'})
            s.call('lend_return',{'job_id':'lend-live'})
            done=s.call('job_wait',{'job_id':'lend-live','timeout':5})
            self.assertEqual((done['status'],done['result']['reason']),('completed','returned'),'the hold loop resumed')
            self.assertEqual(rt.reclaimed,['gpu1'])
        finally:s.close()
    def test_the_restored_lend_still_ends_at_its_deadline(self):
        self.crashed_with(('lend-short','gpu0','queued',time.time()+2))
        rt=LaneRuntime();s=Service(ROOT,self.state,rt)
        try:
            done=s.call('job_wait',{'job_id':'lend-short','timeout':10})
            self.assertEqual((done['status'],done['result']['reason']),('completed','deadline passed'))
        finally:s.close()
    def test_a_granted_lend_is_recorded_with_its_job(self):
        rt=LaneRuntime();s=Service(ROOT,self.state,rt)
        try:
            j=s.call('lend_submit',{'device':'gpu0','mode':'exclusive','hours':1,'owner':'trainer','label':'PR A/B','request_key':'k'})
            deadline=time.monotonic()+5
            while s.call('lend_get',{'job_id':j['id']})['granted'] is None:
                self.assertLess(time.monotonic(),deadline);time.sleep(.01)
            lent=s.call('job_get',{'job_id':j['id']})['lent']
            self.assertEqual((lent['device'],lent['mode'],lent['owner'],lent['label']),('gpu0','exclusive','trainer','PR A/B'));self.assertGreater(lent['deadline'],time.time())
            self.assertEqual(s.call('resource_status',{})['lanes']['gpu0']['lent']['label'],'PR A/B')
            with self.assertRaisesRegex(ValueError,'reserved for research by trainer \\(PR A/B\\)'):s.call('render',{'request':{'prompt':'x','origin':ORIGIN},'request_key':'r'})
            s.call('lend_return',{'job_id':j['id']});s.call('job_wait',{'job_id':j['id'],'timeout':5})
        finally:s.close()
    def test_a_granted_lend_survives_a_service_stop(self):
        # A stop (docker stop / compose up) closes the service; the granted lend is left as it is.
        rt=LaneRuntime();rt.fallback='gpu0';s=Service(ROOT,self.state,rt)
        try:
            j=s.call('lend_submit',{'device':'gpu1','mode':'exclusive','hours':24,'owner':'trainer','request_key':'k'})
            deadline=time.monotonic()+5
            while s.call('lend_get',{'job_id':j['id']})['granted'] is None:
                self.assertLess(time.monotonic(),deadline);time.sleep(.01)
            before=s.call('job_get',{'job_id':j['id']})['lent']
        finally:s.close()
        self.assertEqual(rt.reclaimed,[],'not reclaimed on the way down')
        rt2=LaneRuntime();rt2.fallback='gpu0';s=Service(ROOT,self.state,rt2)
        try:
            job=s.call('job_get',{'job_id':j['id']})
            self.assertEqual((job['status'],job['lent']),('running',before),'still running, granted as before')
            self.assertEqual(rt2.lent['gpu1']['deadline'],before['deadline']);self.assertEqual(rt2.lends,[])
            self.assertEqual(s.call('render',{'request':{'prompt':'r','origin':ORIGIN},'request_key':'r'})['lane'],'gpu0')
            s.call('lend_return',{'job_id':j['id']})
            done=s.call('job_wait',{'job_id':j['id'],'timeout':5})
            self.assertEqual((done['status'],done['result']['reason'],done['lent']),('completed','returned',None))
            self.assertEqual(rt2.reclaimed,['gpu1'])
        finally:s.close()
    def test_a_returned_lend_is_recorded_ended_before_its_device_is_reclaimed(self):
        rt=LaneRuntime();held=threading.Event();reclaim=rt.reclaim
        rt.reclaim=lambda device:(held.wait(10),reclaim(device))[1]
        s=Service(ROOT,self.state,rt)
        try:
            j=s.call('lend_submit',{'device':'gpu0','mode':'exclusive','hours':1,'owner':'trainer','request_key':'k'})
            deadline=time.monotonic()+5
            while s.call('lend_get',{'job_id':j['id']})['granted'] is None:
                self.assertLess(time.monotonic(),deadline);time.sleep(.01)
            s.call('lend_return',{'job_id':j['id']})
            job=s.call('job_get',{'job_id':j['id']})
            self.assertEqual((job['status'],job['lent']),('running',None),'a restart now finds nothing to restore')
            held.set();s.call('job_wait',{'job_id':j['id'],'timeout':5})
        finally:held.set();s.close()
    def test_a_lend_waiting_at_a_restart_counts_its_borrower_silence_from_the_start(self):
        s=Service(ROOT,self.state,LaneRuntime(),start_worker=False)
        try:
            old=time.time()-3600   # submitted an hour ago, its last heartbeat in the service that stopped
            with s.db:s.db.execute("INSERT INTO jobs(id,project,kind,status,request,created,updated,request_key,request_hash,lane,step,waiting)"
                                   " VALUES('lend-waiting','','lend','queued',?,?,?,'k','h','gpu0','lend',1)",
                                   (json.dumps({'device':'gpu0','mode':'queued','hours':1,'owner':'lab'}),old,old))
        finally:s.close()
        rt=LaneRuntime();s=Service(ROOT,self.state,rt)
        try:
            deadline=time.monotonic()+5
            while s.call('lend_get',{'job_id':'lend-waiting'})['granted'] is None:
                self.assertLess(time.monotonic(),deadline,s.call('job_get',{'job_id':'lend-waiting'}).get('error'));time.sleep(.01)
            s.call('lend_return',{'job_id':'lend-waiting'})
            self.assertEqual(s.call('job_wait',{'job_id':'lend-waiting','timeout':5})['result']['reason'],'returned')
        finally:s.close()

    def test_a_lend_of_a_device_no_longer_configured_is_interrupted(self):
        self.crashed_with(('lend-gone','gpu7','exclusive',time.time()+3600))
        rt=LaneRuntime();s=Service(ROOT,self.state,rt)
        try:
            job=s.call('job_get',{'job_id':'lend-gone'})
            self.assertEqual(job['status'],'interrupted');self.assertIn('gpu7 is no longer configured',job['error'])
            self.assertEqual((rt.lent,rt.restored,s.lends),({},[],{}))
        finally:s.close()


class BlockedTests(QueueHarness):
    def setUp(self):
        self.patches=[patch.multiple(job_service,**FAST)]
        for p in self.patches:p.start()
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name);self.rt=LaneRuntime()
        self.rt.foreign['gpu0']=['pid:4242'];self.rt.check_device('gpu0')
        self.s=Service(ROOT,self.state,self.rt)
    def test_a_blocked_device_waits_while_the_other_serves(self):
        r=self.render('r');self.until(lambda:self.between(r),what='r between stages')
        self.assertEqual(self.wait(self.engine('e'))['status'],'completed')
        self.assertEqual(self.call('resource_status')['lanes']['gpu0']['blocked'],'not ours: pid:4242')
        time.sleep(.2);self.assertNotIn(('start','sample:r'),self.rt.log)
        self.rt.foreign.clear()                            # the foreign process went away
        self.assertEqual(self.wait(r)['status'],'completed')
        self.until(lambda:'gpu0' in self.rt.parked,what='gpu0 restored once clear')


class LendTests(QueueHarness):
    def test_queued_mode_waits_its_turn_and_holds_the_device_until_returned(self):
        held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        borrow=self.lend('borrow');y=self.render('y')
        state=self.call('lend_get',job_id=borrow['id'])
        self.assertIsNone(state['granted']);self.assertGreaterEqual(state['ahead'],1)
        held.set();self.until(lambda:self.granted(borrow) is not None,what='granted')
        grant=self.granted(borrow)
        self.assertEqual((grant['device'],grant['owner'],grant['mode'],grant['env']),('gpu0','lab','queued',{'CUDA_VISIBLE_DEVICES':'GPU-gpu0'}))
        self.assertEqual(self.wait(self.engine('e'))['status'],'completed','the other device keeps serving')
        self.until(lambda:self.between(y),what='y between stages')
        self.assertNotIn(('start','sample:y'),self.rt.log)
        self.call('lend_heartbeat',job_id=borrow['id'])
        self.call('lend_return',job_id=borrow['id'])
        done=self.wait(borrow);self.assertEqual((done['status'],done['result']['reason']),('completed','returned'))
        self.assertEqual(self.rt.reclaimed,['gpu0'])
        self.assertEqual(self.wait(y)['status'],'completed');self.wait(x)
        self.until(lambda:'gpu0' in self.rt.parked,what='gpu0 restored')

    def test_a_silent_borrower_loses_the_device_or_never_gets_it(self):
        with patch.object(job_service,'LEND_HEARTBEAT_S',.3):
            gone=self.lend('gone')
            self.until(lambda:self.granted(gone) is not None,what='granted')
            done=self.wait(gone);self.assertEqual(done['result']['reason'],'borrower stopped answering')
            held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
            late=self.lend('late');time.sleep(.5);held.set()
            failed=self.wait(late);self.assertEqual(failed['status'],'failed');self.assertIn('stopped answering',failed['error'])
            self.assertEqual(len(self.rt.lends),1,'the second request was never lent anything');self.wait(x)

    def test_a_waiting_lend_whose_borrower_went_silent_fails_where_it_waits(self):
        with patch.object(job_service,'LEND_HEARTBEAT_S',.5):
            held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
            silent=self.lend('silent');kept=self.lend('kept');stop=threading.Event()
            def beat():
                while not stop.wait(.1):self.call('lend_heartbeat',job_id=kept['id'])
            threading.Thread(target=beat,daemon=True).start()
            try:
                failed=self.wait(silent);self.assertEqual(failed['status'],'failed');self.assertIn('stopped answering while it waited',failed['error'])
                self.assertEqual(self.job(kept)['status'],'queued','its borrower answers: it keeps its place')
                self.assertEqual(self.rt.lends,[],'failed while x still held the device')
                held.set();self.until(lambda:self.granted(kept) is not None,what='kept granted')
                self.call('lend_return',job_id=kept['id']);self.wait(kept)
            finally:stop.set()
        self.wait(x)

    def test_the_deadline_takes_the_device_back(self):
        short=self.lend('short',hours=.5/3600)
        done=self.wait(short);self.assertEqual((done['status'],done['result']['reason']),('completed','deadline passed'))
        self.assertEqual(self.rt.reclaimed,['gpu0'])

    def test_exclusive_mode_refuses_what_needs_the_device_and_keeps_what_was_queued(self):
        # What already waits for the device is kept and waits for its return. Since the encoder
        # fallback (2026-10-10) that holds for every stage except an encode with a usable fallback,
        # which moves there instead: see FallbackTests. Here gpu0 is lent and z waits to sample on it.
        held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        train=self.lend('train',mode='exclusive',owner='trainer');z=self.render('z')   # z: accepted, the reservation is not in force yet
        held.set();self.until(lambda:self.granted(train) is not None,what='granted')
        with self.assertRaisesRegex(ValueError,'reserved for research by trainer until'):self.render('refused')
        with self.assertRaisesRegex(ValueError,'reserved for research by trainer'):self.lend('another')
        self.assertEqual(self.wait(self.engine('e'))['status'],'completed','what does not need gpu0 is served')
        with self.assertRaisesRegex(ValueError,'return the lent devices first'):self.call('resource_release')
        self.until(lambda:self.between(z),what='z between stages');self.assertEqual(self.job(z)['status'],'running')
        self.call('lend_return',job_id=train['id'])
        self.assertEqual(self.wait(z)['status'],'completed');self.wait(x)
        self.assertEqual(self.render('accepted again')['status'],'queued')

    def test_what_the_borrower_left_behind_blocks_the_device_until_it_is_gone(self):
        borrow=self.lend('borrow');self.until(lambda:self.granted(borrow) is not None,what='granted')
        self.rt.foreign['gpu0']=['pid:77']                 # the borrower's process outlives its return
        self.call('lend_return',job_id=borrow['id'])
        done=self.wait(borrow);self.assertEqual(done['result']['left_behind'],['pid:77'])
        self.assertEqual(self.call('resource_status')['lanes']['gpu0']['blocked'],'not ours: pid:77')
        r=self.render('r');self.until(lambda:self.between(r),what='r between stages')
        time.sleep(.2);self.assertNotIn(('start','sample:r'),self.rt.log)
        self.rt.foreign.clear()
        self.assertEqual(self.wait(r)['status'],'completed','runs once the leftover is gone')
        self.assertFalse(self.rt.blocked)

    def test_cancelling_a_lend_queued_or_granted(self):
        held=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        queued=self.lend('queued');self.assertEqual(self.call('job_cancel',job_id=queued['id'])['status'],'cancelled')
        held.set();self.wait(x)
        granted=self.lend('granted');self.until(lambda:self.granted(granted) is not None,what='granted')
        self.call('job_cancel',job_id=granted['id'])
        self.assertEqual(self.wait(granted)['status'],'cancelled');self.assertEqual(self.rt.reclaimed,['gpu0'])
        self.assertEqual(len(self.rt.lends),1)

    def test_requests_are_checked(self):
        for args,message in (({'device':'gpu9'},'unknown device'),({'mode':'forever'},'mode must be'),
                             ({'hours':0},'hours must be'),({'hours':13},'at most 12 in queued'),
                             ({'mode':'exclusive','hours':24*15},'at most 336'),({'owner':' '},'owner'),
                             ({'label':' '},'label'),({'label':'x'*101},'label'),({'label':7},'label'),
                             ({'mode':'shared'},'declares the VRAM'),({'mode':'shared','vram_gib':0},'vram_gib above 0'),
                             ({'mode':'shared','vram_gib':True},'vram_gib'),({'vram_gib':8},'for shared lends')):
            with self.subTest(args=args),self.assertRaisesRegex(ValueError,message):
                self.call('lend_submit',**{'device':'gpu0','mode':'queued','hours':1,'owner':'lab','request_key':'bad',**args})
        self.assertEqual(self.lend('ok',mode='exclusive',hours=24*14)['kind'],'lend')


class SharedLendTests(QueueHarness):
    """Shared lends: side by side on one card while what they declare fits (gpu0: 64 GiB here)."""
    def test_shared_lends_that_fit_hold_the_device_together(self):
        a=self.share('a',24,label='seed 1');b=self.share('b',24)
        self.until(lambda:self.granted(a) and self.granted(b),what='both granted')
        self.assertEqual(self.granted(b)['env'],{'CUDA_VISIBLE_DEVICES':'GPU-gpu0'})
        self.assertEqual(len(self.rt.lends),1,'the device was emptied once');self.assertEqual(self.rt.lent['gpu0']['mode'],'shared')
        c=self.share('c',24);time.sleep(.2)
        self.assertIsNone(self.granted(c),'48 + 24 GiB do not fit in 64');self.assertEqual(self.job(c)['status'],'queued')
        lane=self.call('resource_status')['lanes']['gpu0']
        self.assertEqual([(l['job_id'],l['vram_gib'],l.get('label')) for l in lane['shared']],[(a['id'],24,'seed 1'),(b['id'],24,None)])
        self.assertEqual((lane['running']['id'],[j['id'] for j in lane['also_running']]),(a['id'],[b['id']]),'gpu_top lists both under run')
        with self.assertRaisesRegex(ValueError,'return the lent devices first: gpu0 to lab \\(seed 1\\)'):self.call('resource_release')
        self.call('lend_return',job_id=a['id'])
        done=self.wait(a);self.assertEqual((done['status'],done['result']['reason'],done['lent']),('completed','returned',None))
        self.until(lambda:self.granted(c) is not None,what='c joins once a has left')
        self.assertEqual(self.rt.reclaimed,[],'b and c still hold it')
        self.call('lend_return',job_id=b['id']);self.call('lend_return',job_id=c['id'])
        self.wait(b);self.wait(c);self.until(lambda:self.rt.reclaimed==['gpu0'],what='reclaimed once, after the last')
        self.assertEqual(len(self.rt.lends),1)

    def test_nothing_jumps_the_head_and_another_job_ends_the_joining(self):
        a=self.share('a',40);self.until(lambda:self.granted(a) is not None,what='a granted')
        big=self.share('big',40);small=self.share('small',8);time.sleep(.2)
        self.assertIsNone(self.granted(small),'small fits, but big is ahead of it')
        self.call('lend_return',job_id=a['id'])
        self.until(lambda:self.granted(big) and self.granted(small),what='big and small together')
        q=self.lend('q');late=self.share('late',8);time.sleep(.2)
        self.assertIsNone(self.granted(late),'a queued lend at the head lets nothing more join')
        self.call('lend_return',job_id=big['id']);self.call('lend_return',job_id=small['id'])
        self.until(lambda:self.granted(q) is not None,what='q granted after the shared ones')
        self.assertIsNone(self.granted(late))
        self.call('lend_return',job_id=q['id']);self.until(lambda:self.granted(late) is not None,what='late after q')
        self.call('lend_return',job_id=late['id']);self.wait(late)
        self.assertEqual(self.rt.reclaimed,['gpu0']*4,'after a, after big and small, after q, after late')

    def test_a_render_waits_for_the_last_shared_lend(self):
        a=self.share('a',8);b=self.share('b',8);self.until(lambda:self.granted(a) and self.granted(b),what='granted')
        r=self.render('r');self.until(lambda:self.between(r),what='r between stages')
        self.call('lend_return',job_id=a['id']);time.sleep(.2)
        self.assertNotIn(('start','sample:r'),self.rt.log,'b still holds gpu0')
        self.call('lend_return',job_id=b['id'])
        self.assertEqual(self.wait(r)['status'],'completed')

    def test_each_ends_on_its_own_deadline_heartbeat_or_cancel(self):
        with patch.object(job_service,'LEND_HEARTBEAT_S',.5):
            short=self.share('short',8,hours=.3/3600);silent=self.share('silent',8);kept=self.share('kept',8);gone=self.share('gone',8)
            self.until(lambda:all(self.granted(j) for j in (short,silent,kept,gone)),what='all granted')
            self.call('job_cancel',job_id=gone['id'])
            stop=threading.Event()
            def beat():
                while not stop.wait(.1):self.call('lend_heartbeat',job_id=kept['id'])
            threading.Thread(target=beat,daemon=True).start()
            try:
                self.assertEqual(self.wait(short)['result']['reason'],'deadline passed')
                self.assertEqual(self.wait(silent)['result']['reason'],'borrower stopped answering')
                self.assertEqual(self.wait(gone)['status'],'cancelled')
                self.assertIsNotNone(self.granted(kept));self.assertEqual(self.rt.reclaimed,[])
                self.call('lend_return',job_id=kept['id']);self.wait(kept)
            finally:stop.set()
        self.until(lambda:self.rt.reclaimed==['gpu0'],what='reclaimed')

    def test_one_bigger_than_the_card_fails(self):
        huge=self.share('huge',80)
        failed=self.wait(huge);self.assertEqual(failed['status'],'failed');self.assertIn('needs 80 GiB',failed['error'])
        self.until(lambda:self.rt.reclaimed==['gpu0'],what='given back')

    def test_shared_lends_survive_a_restart(self):
        a=self.share('a',8);b=self.share('b',8);self.until(lambda:self.granted(a) and self.granted(b),what='granted')
        self.s.close()
        rt=LaneRuntime();self.rt=rt;self.s=Service(ROOT,self.state,rt)
        self.assertEqual([d for d,_ in rt.restored],['gpu0'],'lent again once');self.assertEqual(rt.lends,[])
        self.assertEqual({self.granted(j)['job_id'] for j in (a,b)},{a['id'],b['id']})
        c=self.share('c',8);self.until(lambda:self.granted(c) is not None,what='c joins the restored ones')
        for j in (a,b,c):self.call('lend_return',job_id=j['id'])
        self.assertEqual([self.wait(j)['result']['reason'] for j in (a,b,c)],['returned']*3)
        self.until(lambda:rt.reclaimed==['gpu0'],what='reclaimed')


class FallbackTests(QueueHarness):
    """The encoder's device (gpu1 here) lent or blocked: encodes go to the fallback (gpu0)."""
    def test_each_encode_is_handed_the_device_whose_queue_it_waited_in(self):
        self.assertEqual(self.wait(self.render('a'))['status'],'completed')
        self.assertEqual(self.rt.encoded_on,{'a':'gpu1'})
        self.rt.fallback='gpu0'
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        self.until(lambda:self.granted(train) is not None,what='granted')
        b=self.render('b');self.assertEqual(self.job(b)['lane'],'gpu0','accepted and queued on the fallback, not refused')
        self.assertEqual(self.wait(b)['status'],'completed');self.assertEqual(self.rt.encoded_on['b'],'gpu0')
        self.call('lend_return',job_id=train['id']);self.wait(train)
        self.assertEqual(self.wait(self.render('c'))['status'],'completed');self.assertEqual(self.rt.encoded_on['c'],'gpu1','back once returned')

    def test_without_a_usable_fallback_the_refusal_stands(self):
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        self.until(lambda:self.granted(train) is not None,what='granted')
        with self.assertRaisesRegex(ValueError,'gpu1 is reserved for research by trainer'):self.render('no fallback')
        self.rt.fallback='gpu0';self.rt.foreign['gpu0']=['pid:9'];self.rt.check_device('gpu0')
        with self.assertRaisesRegex(ValueError,'gpu1 is reserved'):self.render('fallback blocked')
        self.rt.foreign.clear();self.rt.check_device('gpu0')
        self.assertEqual(self.render('fallback clear')['lane'],'gpu0')

    def test_an_exclusive_grant_moves_the_waiting_encodes_in_order_and_nothing_else(self):
        self.rt.fallback='gpu0'
        enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        b=self.render('b');c=self.render('c')            # queued on gpu1 behind the lend, before it is granted
        self.assertEqual((self.job(b)['lane'],self.job(c)['lane']),('gpu1','gpu1'))
        enc.set()                                         # a finishes encoding on gpu1 and samples on gpu0
        self.until(lambda:self.granted(train) is not None,what='granted')
        self.until(lambda:self.job(b)['lane']=='gpu0' and self.job(c)['lane']=='gpu0',what='b and c moved')
        self.assertEqual(self.wait(a)['status'],'completed');self.assertEqual(self.rt.encoded_on['a'],'gpu1','a ran where it was')
        for j in (b,c):self.assertEqual(self.wait(j)['status'],'completed')
        self.assertEqual((self.rt.encoded_on['b'],self.rt.encoded_on['c']),('gpu0','gpu0'))
        self.assertTrue(self.order('encode:b','encode:c'),'their order is kept')
        self.call('lend_return',job_id=train['id']);self.wait(train)

    def test_a_queued_lend_moves_nothing_already_waiting(self):
        self.rt.fallback='gpu0'
        enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        borrow=self.lend('borrow',device='gpu1');b=self.render('b')
        enc.set();self.until(lambda:self.granted(borrow) is not None,what='granted')
        time.sleep(.2);self.assertEqual(self.job(b)['lane'],'gpu1','a queued lend is short: b waits for it')
        self.call('lend_return',job_id=borrow['id']);self.assertEqual(self.wait(b)['status'],'completed')
        self.assertEqual(self.rt.encoded_on['b'],'gpu1');self.wait(a)

    def test_once_the_device_is_back_the_waiting_encodes_return_to_it(self):
        self.rt.fallback='gpu0'
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        self.until(lambda:self.granted(train) is not None,what='granted')
        enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        held=[self.rt.gate('encode:b'),self.rt.gate('encode:c')];b=self.render('b');c=self.render('c')
        self.assertEqual((self.job(b)['lane'],self.job(c)['lane']),('gpu0','gpu0'))
        self.call('lend_return',job_id=train['id']);self.wait(train)
        self.until(lambda:self.job(b)['lane']=='gpu1' and self.job(c)['lane']=='gpu1',what='b and c moved back')
        for gate in [enc,*held]:gate.set()
        for j in (a,b,c):self.assertEqual(self.wait(j)['status'],'completed')
        self.assertEqual([self.rt.encoded_on[k] for k in 'abc'],['gpu0','gpu1','gpu1'],'one switch, no back and forth')
        self.assertTrue(self.order('encode:b','encode:c'))

    def test_an_engine_call_falls_back_like_an_encode(self):
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        self.until(lambda:self.granted(train) is not None,what='granted')
        with self.assertRaisesRegex(ValueError,'gpu1 is reserved for research by trainer'):self.engine('no fallback')
        self.rt.engine_fallback='gpu0'
        e=self.engine('e1');self.assertEqual(self.job(e)['lane'],'gpu0','accepted and queued on the fallback, not refused')
        self.assertEqual(self.wait(e)['status'],'completed');self.assertEqual(self.rt.engine_on['/v1/audio/transcriptions'],'gpu0')
        self.call('lend_return',job_id=train['id']);self.wait(train)
        self.assertEqual(self.wait(self.engine('e2',path='/v1/audio/speech'))['status'],'completed')
        self.assertEqual(self.rt.engine_on['/v1/audio/speech'],'gpu1','back once returned')

    def test_waiting_engine_calls_return_to_their_device_once_it_is_back(self):
        self.rt.fallback=self.rt.engine_fallback='gpu0'
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer')
        self.until(lambda:self.granted(train) is not None,what='granted')
        hold=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')   # gpu0 busy
        e=self.engine('e1');self.assertEqual(self.job(e)['lane'],'gpu0')
        self.call('lend_return',job_id=train['id']);self.wait(train)
        self.until(lambda:self.job(e)['lane']=='gpu1',what='e1 moved back')
        self.assertEqual(self.wait(e)['status'],'completed');self.assertEqual(self.rt.engine_on['/v1/audio/transcriptions'],'gpu1')
        hold.set();self.assertEqual(self.wait(a)['status'],'completed')

    def test_a_fallback_that_clears_after_the_grant_still_takes_the_waiting_encodes(self):
        self.rt.fallback='gpu0';self.rt.foreign['gpu0']=['pid:9'];self.rt.check_device('gpu0')
        enc=self.rt.gate('encode:a');a=self.render('a');self.rt.wait_started('encode:a')
        train=self.lend('train',device='gpu1',mode='exclusive',owner='trainer');b=self.render('b')
        enc.set();self.until(lambda:self.granted(train) is not None,what='granted')
        time.sleep(.2);self.assertEqual(self.job(b)['lane'],'gpu1','the fallback is blocked: nowhere to go yet')
        self.rt.foreign.clear()
        self.until(lambda:self.job(b)['lane']=='gpu0',what='b moved once gpu0 cleared')
        for j in (a,b):self.assertEqual(self.wait(j)['status'],'completed')
        self.assertEqual(self.rt.encoded_on['b'],'gpu0')
        self.call('lend_return',job_id=train['id']);self.wait(train)


class BatchingTests(unittest.TestCase):
    """_next on a device serving both render stages: the loaded stage first, the oldest never
    passed over more than BATCH_PASS_LIMIT times, engine calls first as ever."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.rt=LaneRuntime()
        self.s=Service(ROOT,Path(self.temp.name),self.rt,start_worker=False);self.n=0
    def tearDown(self):self.s.close();self.temp.cleanup()
    def add(self,ident,step,kind='render',lane='gpu0'):
        self.n+=1
        with self.s.db:self.s.db.execute("INSERT INTO jobs(id,project,kind,status,request,created,updated,request_key,request_hash,lane,step,waiting,urgent)"
                                         " VALUES(?,'',?,'queued','{}',?,?,?,'h',?,?,1,?)",(ident,kind,self.n,self.n,ident,lane,step,int(kind=='engine')))
    def take(self,lane='gpu0'):
        """What the queue would run next, then mark it running as _work does."""
        row=self.s._next(lane)
        if row is None:return None
        with self.s.db:self.s.db.execute('UPDATE jobs SET waiting=0 WHERE id=?',(row['id'],))
        self.s.passed.pop(row['id'],None);return row['id']
    def test_the_loaded_stage_goes_first_and_fifo_within_it(self):
        for ident,step in (('s1','sample'),('e1','encode'),('s2','sample'),('e2','encode')):self.add(ident,step)
        self.rt.loaded['gpu0']='encode'
        self.assertEqual([self.take(),self.take()],['e1','e2'],'the encoder is loaded: both encodes, oldest first')
        self.rt.loaded['gpu0']='sample'
        self.assertEqual([self.take(),self.take(),self.take()],['s1','s2',None])
    def test_nothing_waits_behind_more_than_the_limit(self):
        self.add('s1','sample')
        for i in range(1,7):self.add('e%d'%i,'encode')
        self.rt.loaded['gpu0']='encode'
        order=[self.take() for _ in range(7)]
        self.assertEqual(job_service.BATCH_PASS_LIMIT,3)
        self.assertEqual(order,['e1','e2','e3','s1','e4','e5','e6'],'s1 is passed over three times, then served')
    def test_nothing_loaded_or_one_stage_only_is_plain_fifo_and_engines_lead(self):
        self.add('s1','sample');self.add('e1','encode');self.add('x','engine',kind='engine')
        self.assertEqual(self.take(),'x','engine calls keep their priority')
        self.assertEqual(self.take(),'s1','nothing loaded: creation order')
        self.add('s2','sample');self.rt.loaded['gpu0']='sample'
        self.assertEqual([self.take(),self.take()],['s2','e1'])
        self.add('l','lend',kind='lend');self.add('s3','sample');self.rt.loaded['gpu0']='sample'
        self.assertEqual(self.take(),'l','only the two render stages are reordered')
    def test_a_lend_behind_the_head_is_not_jumped(self):
        self.add('s1','sample');self.add('l','lend',kind='lend');self.add('e1','encode');self.add('e2','encode')
        self.rt.loaded['gpu0']='encode'
        self.assertEqual([self.take(),self.take(),self.take(),self.take()],['s1','l','e1','e2'],'renders created after the lend wait behind it')


class BorrowCliTests(unittest.TestCase):
    """runtime/gpu_borrow.py against a real service socket (fake runtime underneath)."""
    def setUp(self):
        self.patches=[patch.multiple(job_service,**FAST)]
        for p in self.patches:p.start()
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name);self.rt=LaneRuntime()
        self.s=Service(ROOT,self.state,self.rt)
        self.sock=str(self.state/'runtime.sock')
        self.server=job_service.Server(self.sock,job_service.Handler);self.server.service=self.s
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def tearDown(self):
        for lend in [*self.s.lends.values(),*(l for lends in self.s.shared.values() for l in lends.values())]:lend['returned']=True
        with self.s.cv:self.s.cv.notify_all()
        self.server.shutdown();self.server.server_close();self.s.close();self.temp.cleanup()
        for p in self.patches:p.stop()
    def borrow(self,*args,timeout=30):
        import subprocess
        return subprocess.run([sys.executable,str(ROOT/'runtime'/'gpu_borrow.py'),'--socket',self.sock,*args],capture_output=True,text=True,timeout=timeout)
    def lends(self):return [dict(r) for r in self.s.db.execute("SELECT id,status,result FROM jobs WHERE kind='lend'")]

    def test_run_sees_only_its_device_passes_the_exit_code_and_returns_the_device(self):
        done=self.borrow('run','--device','gpu0','--owner','lab','--','python3','-c',
                         'import os,sys;print(os.environ["CUDA_VISIBLE_DEVICES"]);sys.exit(3)')
        self.assertEqual(done.returncode,3,done.stderr);self.assertEqual(done.stdout.strip(),'GPU-gpu0')
        [row]=self.lends();self.assertEqual(row['status'],'completed');self.assertEqual(json.loads(row['result'])['reason'],'returned')

    def test_run_stops_the_command_at_the_deadline(self):
        started=time.monotonic()
        done=self.borrow('run','--device','gpu0','--hours',str(4/3600),'--','sleep','60')
        self.assertEqual(done.returncode,143,done.stderr);self.assertLess(time.monotonic()-started,20)
        self.assertIn('deadline reached',done.stderr)

    def test_interrupting_the_wait_withdraws_the_request(self):
        import signal,subprocess
        held=self.rt.gate('sample:x')
        x=self.s.call('render',{'request':{'prompt':'x','origin':ORIGIN},'request_key':'x'});self.rt.wait_started('sample:x')
        p=subprocess.Popen([sys.executable,str(ROOT/'runtime'/'gpu_borrow.py'),'--socket',self.sock,'run','--device','gpu0','--','true'],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+10
        while not self.lends():
            self.assertLess(time.monotonic(),deadline);time.sleep(.05)
        time.sleep(.3);p.send_signal(signal.SIGINT)
        self.assertEqual(p.wait(10),130);p.stdout.close();p.stderr.close()
        [row]=self.lends();self.assertEqual(row['status'],'cancelled')
        held.set();self.s.call('job_wait',{'job_id':x['id'],'timeout':5})
        self.assertEqual(self.rt.lends,[],'nothing was lent')

    def test_the_command_dies_with_the_tool(self):
        import os,signal,subprocess
        pidfile=self.state/'child.pid'
        p=subprocess.Popen([sys.executable,str(ROOT/'runtime'/'gpu_borrow.py'),'--socket',self.sock,'run','--device','gpu0','--',
                            'sh','-c','echo $$ > %s; exec sleep 60'%pidfile],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+15
        while not (pidfile.exists() and pidfile.read_text().strip()):
            self.assertLess(time.monotonic(),deadline);time.sleep(.05)
        child=int(pidfile.read_text());p.kill();p.wait()
        deadline=time.monotonic()+5
        while True:
            try:os.kill(child,0)
            except ProcessLookupError:break
            try:
                if Path('/proc',str(child),'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
            except OSError:break   # gone between the two looks
            self.assertLess(time.monotonic(),deadline,'the command outlived the tool');time.sleep(.05)

    def test_reserve_status_and_release(self):
        done=self.borrow('reserve','--device','gpu0','--owner','trainer','--label','PR A/B','--hours','24')
        self.assertEqual(done.returncode,0,done.stderr);self.assertIn('export CUDA_VISIBLE_DEVICES=GPU-gpu0',done.stdout)
        [row]=self.lends();self.assertEqual(row['status'],'running')
        status=self.borrow('status');self.assertIn('lent (exclusive) to trainer (PR A/B)',status.stdout)
        self.assertEqual(self.borrow('release',row['id']).stdout.strip(),'running')
        self.assertEqual(self.s.call('job_wait',{'job_id':row['id'],'timeout':5})['status'],'completed')

    def test_a_runtime_restart_does_not_end_a_waiting_borrow(self):
        import os,subprocess
        held=self.rt.gate('sample:x')
        x=self.s.call('render',{'request':{'prompt':'x','origin':ORIGIN},'request_key':'x'});self.rt.wait_started('sample:x')
        p=subprocess.Popen([sys.executable,str(ROOT/'runtime'/'gpu_borrow.py'),'--socket',self.sock,'run','--device','gpu0','--','true'],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+10
        while not self.lends():
            self.assertLess(time.monotonic(),deadline);time.sleep(.05)
        time.sleep(1.2)   # polling for its turn
        # The socket is gone for a moment, as while the runtime's container restarts.
        self.server.shutdown();self.server.server_close();os.unlink(self.sock);time.sleep(2)
        self.server=job_service.Server(self.sock,job_service.Handler);self.server.service=self.s
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        held.set()
        out,err=p.communicate(timeout=30);self.assertEqual(p.returncode,0,err)
        [row]=self.lends();self.assertEqual(row['status'],'completed')
        self.s.call('job_wait',{'job_id':x['id'],'timeout':5})

    def test_run_shares_the_device_with_vram_gib(self):
        import subprocess
        p=subprocess.Popen([sys.executable,str(ROOT/'runtime'/'gpu_borrow.py'),'--socket',self.sock,'run','--device','gpu0','--vram-gib','12',
                            '--label','seed 1','--','sh','-c','touch %s; exec sleep 30'%(self.state/'started')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+10
            while not (self.state/'started').exists():   # the command runs: the tool holds the grant
                self.assertLess(time.monotonic(),deadline);time.sleep(.05)
            self.assertIn('shared 12.0 GiB with',self.borrow('status').stdout)
        finally:p.terminate();p.wait(10)
        [row]=self.lends();self.assertEqual(self.s.call('job_wait',{'job_id':row['id'],'timeout':5})['status'],'completed')

if __name__=='__main__':unittest.main()
