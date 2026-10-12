"""Services: one thing callers name (/engines/<name>, or the model of a /v1 call), never a device.
gpu-config `services` expand into server tenants — `prefer` one device with a fallback launched its
own way, `spread` one tenant per device — and a tenant with `slots` takes that many calls at once.
Calls by model wait in creation order with renders; calls by tenant name still go first."""
import contextlib
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
import job_service
from app_fixture import Service
from gpu_runtime import Scheduler, expand_services, load_config
from test_device_queues import FAST, LaneRuntime, QueueHarness, ROOT
import harness as base

QWEN={'models':['qwen3','translator'],'slots':3}


class ServiceRuntime(LaneRuntime):
    """LaneRuntime plus server tenants on their own devices, sessions and per-call steps
    ('call:<path>', gated and failed like any step)."""
    def __init__(self):
        super().__init__()
        self.tenants={'audio':{},'qwen3@gpu1':dict(QWEN,device='gpu1')}
        self.home={'qwen3@gpu1':'gpu1','qwen3@gpu0':'gpu0'}
        self.server_on={};self.sessions=[];self.place_error=None
    def model_tenants(self,name):return [t for t,s in self.tenants.items() if name in s.get('models',[])]
    def validate_engine(self,req):pass
    def stages(self,kind,req):
        if kind=='engine' and req.get('engine') in self.home:
            if req.get('devices') and self.home[req['engine']] not in req['devices']:raise ValueError('not on those devices')
            return [('engine',self.home[req['engine']])]
        return super().stages(kind,req)
    def stage_devices(self,kind,req):
        if kind=='engine' and req.get('engine') in self.home:return self.home[req['engine']],None
        return super().stage_devices(kind,req)
    @contextlib.contextmanager
    def engine_session(self,name,lane):
        if self.place_error:raise RuntimeError(self.place_error)
        self.sessions.append((name,lane));self.server_on[name]=lane
        yield 'server:'+name,lane
    def engine_call(self,req,out,server,device):
        self.engine_on[req['path']]=device
        self._step(device,'call:'+req['path']);out.mkdir(parents=True);return {'status':200,'device':device}


class ServiceHarness(QueueHarness):
    def setUp(self):
        self.patches=[patch.multiple(job_service,**FAST)]
        for p in self.patches:p.start()
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name);self.rt=ServiceRuntime()
        self.s=Service(ROOT,self.state,self.rt)
    def ask(self,path,key=None,model='qwen3'):
        return self.call('engine_submit',model=model,path=path,method='POST',body='',content_type='application/json',request_key=key or path)
    def by_name(self,path,name='translator'):
        return self.call('engine_submit',engine=name,path=path,method='POST',body='',content_type='application/json',request_key=path)
    def started(self,name):return ('start',name) in self.rt.log
    def running(self):return sum(1 for e,n in self.rt.log if n.startswith('call:') and e=='start')-sum(1 for e,n in self.rt.log if n.startswith('call:') and e=='end')


class SlotTests(ServiceHarness):
    def test_calls_run_side_by_side_up_to_the_slots(self):
        gates={p:self.rt.gate('call:'+p) for p in ('/c1','/c2','/c3','/c4','/c5')}
        jobs=[self.ask(p) for p in gates]
        self.until(lambda:self.running()==3,what='three calls running')
        time.sleep(.1);self.assertFalse(self.started('call:/c4'),'the fourth waits for a slot')
        lanes=self.call('resource_status')['lanes']
        self.assertEqual(lanes['gpu1']['running']['id'],jobs[0]['id'])
        self.assertEqual([j['id'] for j in lanes['gpu1']['also_running']],[jobs[1]['id'],jobs[2]['id']])
        self.assertEqual(lanes['gpu1']['also_running'][0]['model'],'qwen3');self.assertNotIn('request',lanes['gpu1']['also_running'][0])
        gates['/c2'].set();self.until(lambda:self.started('call:/c4'),what='the fourth took the free slot')
        for gate in gates.values():gate.set()
        for job in jobs:self.assertEqual(self.wait(job)['status'],'completed')
        self.assertEqual(self.rt.max_live['gpu1'],3)
        self.assertEqual(self.rt.sessions,[('qwen3@gpu1','gpu1')],'the server was placed once for all five')
        self.until(lambda:self.call('resource_status')['lanes']['gpu1']['also_running']==[],what='the session over')

    def test_whatever_else_reaches_the_head_ends_the_session(self):
        first=self.rt.gate('call:/c1');c1=self.ask('/c1');self.rt.wait_started('call:/c1')
        r=self.render('r');c2=self.ask('/c2')
        time.sleep(.15);self.assertFalse(self.started('call:/c2'),'the render came first: c2 does not join')
        first.set()
        for job in (c1,r,c2):self.assertEqual(self.wait(job)['status'],'completed')
        self.assertTrue(self.order('encode:r','call:/c2'),'creation order with renders')
        self.assertEqual(len(self.rt.sessions),2)

    def test_a_call_by_name_goes_first_and_joins_the_running_session(self):
        held=self.rt.gate('call:/c1');c1=self.ask('/c1');self.rt.wait_started('call:/c1')
        r=self.render('r');urgent=self.by_name('/now')
        self.until(lambda:self.started('call:/now'),what='the call by name joined while c1 runs')
        self.assertFalse(self.started('encode:r'))
        held.set()
        for job in (c1,r,urgent):self.assertEqual(self.wait(job)['status'],'completed')
        self.assertEqual(self.job(urgent)['request']['engine'],'qwen3@gpu1');self.assertEqual(self.job(urgent)['request']['route'],'translator')
        self.assertNotIn('model',self.job(urgent)['request'])

    def test_a_failed_call_stops_the_device_once_the_session_is_over(self):
        held=self.rt.gate('call:/ok');ok=self.ask('/ok');self.rt.wait_started('call:/ok')
        self.rt.failures['call:/bad']='server fell over';bad=self.ask('/bad')
        self.assertEqual(self.wait(bad)['status'],'failed')
        self.assertNotIn('gpu1',self.rt.stopped,'not while another call still runs')
        held.set();self.assertEqual(self.wait(ok)['status'],'completed')
        self.until(lambda:'gpu1' in self.rt.stopped,what='gpu1 stopped')

    def test_a_server_that_cannot_be_placed_fails_its_call(self):
        self.rt.place_error='no room on gpu1'
        failed=self.wait(self.ask('/c'));self.assertEqual(failed['status'],'failed');self.assertIn('no room on gpu1',failed['error'])
        self.until(lambda:'gpu1' in self.rt.stopped,what='gpu1 stopped')
        self.rt.place_error=None;self.assertEqual(self.wait(self.ask('/again'))['status'],'completed','the queue goes on')


class RoutingTests(ServiceHarness):
    def setUp(self):
        super().setUp();self.rt.tenants['qwen3@gpu0']=dict(QWEN,device='gpu0')

    def test_a_model_goes_to_the_least_busy_device_and_never_to_a_reserved_one(self):
        held=self.rt.gate('call:/c1');self.ask('/c1');self.rt.wait_started('call:/c1')
        self.assertEqual(self.rt.engine_on['/c1'],'gpu1','both idle and neither loaded: the first declared')
        c2=self.ask('/c2');self.wait(c2);self.assertEqual(self.rt.engine_on['/c2'],'gpu0','gpu1 is busy')
        lend=self.lend('l',device='gpu0',mode='exclusive');self.until(lambda:self.granted(lend) is not None,what='gpu0 lent')
        c3=self.ask('/c3');self.assertEqual(self.job(c3)['request']['engine'],'qwen3@gpu1','gpu0 is reserved')
        held.set();self.wait(c3);self.call('lend_return',job_id=lend['id'])

    def test_a_retry_under_the_same_key_keeps_its_tenant(self):
        held=self.rt.gate('call:/c1');c1=self.ask('/c1');self.rt.wait_started('call:/c1')
        self.assertEqual(self.ask('/c1')['id'],c1['id'],'gpu1 is now the busier, yet the retry is the same call')
        held.set();self.wait(c1)

    def test_waiting_calls_leave_a_device_reserved_after_they_were_routed(self):
        sampling=self.rt.gate('sample:x');x=self.render('x');self.rt.wait_started('sample:x')
        lend=self.lend('l',device='gpu0',mode='exclusive')            # queued behind the sampling
        gates=[self.rt.gate('call:/b%d'%i) for i in range(3)];busy=[self.ask('/b%d'%i) for i in range(3)]
        self.until(lambda:self.running()==3,what='gpu1 full')
        c=self.ask('/c');self.assertEqual(self.job(c)['lane'],'gpu0','gpu1 is busier')
        sampling.set();self.until(lambda:self.granted(lend) is not None,what='gpu0 lent')
        moved=self.job(c);self.assertEqual((moved['lane'],moved['request']['engine']),('gpu1','qwen3@gpu1'))
        for gate in gates:gate.set()
        self.assertEqual(self.wait(c)['status'],'completed');self.assertEqual(self.rt.engine_on['/c'],'gpu1')
        self.assertEqual(self.ask('/c',key='/c')['id'],c['id'],'its fingerprint followed the move')
        for job in busy+[x]:self.wait(job)
        self.call('lend_return',job_id=lend['id'])

    def test_an_unknown_model_is_refused_with_what_is_served(self):
        with self.assertRaisesRegex(ValueError,r"unknown model 'nope'; this host serves: qwen3, translator"):self.ask('/c',model='nope')
        self.assertEqual(self.call('models'),[{'id':'qwen3','max_parallel':6,'devices':['gpu0','gpu1'],'backends':[]},
                                              {'id':'translator','max_parallel':6,'devices':['gpu0','gpu1'],'backends':[]}],'two tenants of 3 slots each')


class DevicesTests(ServiceHarness):
    """A call's `devices`: where it may run, e.g. only where earlier calls ran, for identical output."""
    def setUp(self):
        super().setUp();self.rt.tenants['qwen3@gpu0']=dict(QWEN,device='gpu0')
    def ask_on(self,path,devices,model='qwen3'):
        return self.call('engine_submit',model=model,path=path,method='POST',body='',content_type='application/json',request_key=path,devices=devices)

    def test_a_call_goes_only_where_it_allows(self):
        c=self.ask_on('/c',['gpu0']);self.wait(c)
        self.assertEqual((self.rt.engine_on['/c'],self.job(c)['request']['devices']),('gpu0',['gpu0']),'gpu1, first declared, is idle too')
        held=self.rt.gate('call:/b1');b1=self.ask_on('/b1',['gpu1']);self.rt.wait_started('call:/b1')
        self.assertEqual(self.job(self.ask('/c2'))['request']['engine'],'qwen3@gpu0','gpu1 is the busier')
        c3=self.ask_on('/c3',['gpu1']);self.assertEqual(self.job(c3)['request']['engine'],'qwen3@gpu1','busier, but the only one allowed')
        held.set()
        for job in (b1,c3):self.wait(job)
        self.assertEqual(self.rt.engine_on['/c3'],'gpu1')

    def test_a_reserved_device_refuses_it_rather_than_sending_it_elsewhere(self):
        lend=self.lend('l',device='gpu1',mode='exclusive');self.until(lambda:self.granted(lend) is not None,what='gpu1 lent')
        with self.assertRaisesRegex(ValueError,'gpu1 is reserved for research'):self.ask_on('/c',['gpu1'])
        self.assertEqual(self.job(self.ask('/d'))['request']['engine'],'qwen3@gpu0','without devices it goes to gpu0')
        self.call('lend_return',job_id=lend['id'])

    def test_mistakes_are_refused(self):
        for devices,message in (('gpu1','devices must be a list'),([],'devices must be a list'),(['gpu1','gpu1'],'distinct'),
                                (['gpu9'],'unknown device gpu9; this host has gpu0, gpu1')):
            with self.subTest(devices=devices),self.assertRaisesRegex(ValueError,message):self.ask_on('/x',devices)
        del self.rt.tenants['qwen3@gpu0']
        with self.assertRaisesRegex(ValueError,'qwen3 runs on gpu1, not on gpu0'):self.ask_on('/y',['gpu0'])


class LingerTests(ServiceHarness):
    def test_a_lingering_tenant_keeps_its_device_before_the_idle_restore(self):
        self.rt.tenants['qwen3@gpu1']['linger_s']=.6
        self.wait(self.ask('/c1'));done=time.monotonic();self.rt.parked.clear()
        time.sleep(.3);self.assertNotIn('gpu1',self.rt.parked,'still lingering')
        self.until(lambda:'gpu1' in self.rt.parked,what='restored after the linger')
        self.assertGreaterEqual(time.monotonic()-done,.55)

    def test_a_job_that_arrives_meanwhile_is_not_held_back(self):
        self.rt.tenants['qwen3@gpu1']['linger_s']=30
        self.wait(self.ask('/c1'))
        self.assertEqual(self.wait(self.render('r'))['status'],'completed','renders run while it lingers')

    def test_nothing_lingers_once_the_tenant_left_the_device(self):
        self.rt.tenants['qwen3@gpu1']['linger_s']=30
        self.wait(self.ask('/c1'));self.rt.server_on.clear();self.rt.parked.clear()
        self.wait(self.by_name('/x',name='audio'))
        self.until(lambda:'gpu1' in self.rt.parked,what='restored without waiting 30 s')


class KeepingTests(ServiceHarness):
    def test_calls_by_model_are_deleted_a_day_after_they_end(self):
        model=self.wait(self.ask('/m'));named=self.wait(self.call('engine_submit',engine='audio',path='/a',method='POST',body='',content_type='x',request_key='a'))
        self.assertTrue((self.state/'jobs'/model['id']).is_dir())
        with self.s.db:self.s.db.execute('UPDATE jobs SET updated=? WHERE id IN (?,?)',(time.time()-job_service.MODEL_CALL_KEEP_S-1,model['id'],named['id']))
        self.s.pruned=0;self.ask('/fresh')
        with self.assertRaisesRegex(ValueError,'unknown jobs id'):self.job(model)
        self.assertFalse((self.state/'jobs'/model['id']).exists())
        self.assertEqual(self.job(named)['status'],'completed','calls by tenant name are kept')

    def test_brief_leaves_the_request_out(self):
        job=self.wait(self.ask('/m'))
        self.assertNotIn('request',self.call('job_get',job_id=job['id'],brief=True))
        self.assertNotIn('request',self.call('job_wait',job_id=job['id'],timeout=0,brief=True))
        self.assertIn('request',self.job(job))


class MigrationTests(unittest.TestCase):
    def test_engine_calls_from_before_keep_going_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=sqlite3.connect(Path(tmp)/Service.DB_NAME)
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY, project TEXT NOT NULL, shot TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,'
                       ' request TEXT NOT NULL, result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL, request_key TEXT UNIQUE, request_hash TEXT NOT NULL)')
            for ident,kind in (('e','engine'),('r','render')):
                db.execute("INSERT INTO jobs VALUES(?,'',NULL,?,'completed','{}',NULL,NULL,1,1,?,'h')",(ident,kind,ident))
            db.commit();db.close()
            s=Service(ROOT,tmp,LaneRuntime(),start_worker=False)
            self.assertEqual(dict(s.db.execute('SELECT id,urgent FROM jobs').fetchall()),{'e':1,'r':0})
            s.close()


LLAMA={'device':'gpu0','api':'vulkan','command':['llama-server'],'url':'http://127.0.0.1:18000','cwd':'/x'}
VLLM={'device':'gpu1','api':'cuda','command':['vllm','serve'],'url':'http://127.0.0.1:18200'}

class ServiceConfigTests(base.SchedulerHarness):
    def load(self,restore=('audio',),**services):
        cfg=base.v2();del cfg['tenants']['translator'];cfg['services']=services;cfg['policy']['idle']['restore']=list(restore)
        path=self.state/'gpu-config.json';path.write_text(json.dumps(cfg));return load_config(path)

    def test_prefer_runs_on_the_first_backend_and_its_fallback_launches_its_own_way(self):
        cfg=self.load(translator={'names':['translator','qwen3'],'backends':[LLAMA,VLLM],'slots':4})
        spec=cfg['tenants']['translator']
        self.assertEqual((spec['device'],spec['fallback'],spec['slots'],spec['models']),('gpu0','gpu1',4,['translator','qwen3']))
        self.assertEqual(spec['routes'],['/health','/v1/models','/v1/chat/completions','/v1/completions','/v1/embeddings'],'an OpenAI-style server unless it declares routes')
        self.assertEqual(self.load(translator={'backends':[LLAMA],'routes':['/health']})['tenants']['translator']['routes'],['/health'])
        s=self.scheduler(cfg)
        self.assertEqual(s.model_tenants('qwen3'),['translator'])
        s._server('translator');home=base.FakeServer.started[-1]
        self.assertEqual((home.command,home.url,home.env['GGML_VK_VISIBLE_DEVICES']),(['llama-server'],'http://127.0.0.1:18000','0'))
        s.lend('gpu0',{'owner':'lab','job_id':'lend-1'})
        self.assertEqual(s.stages('engine',{'engine':'translator'}),[('engine','gpu1')])
        s._server('translator','gpu1');away=base.FakeServer.started[-1]
        self.assertEqual((away.command,away.url,away.env['CUDA_VISIBLE_DEVICES']),(['vllm','serve'],'http://127.0.0.1:18200','GPU-1111'))
        self.assertNotIn('GGML_VK_VISIBLE_DEVICES',away.env)

    def test_each_device_says_how_it_runs_the_service(self):
        s=self.scheduler(self.load(translator={'names':['translator'],'backends':[dict(LLAMA,command=['/b/llama-b1/llama-server','-m','/m/q4.gguf']),VLLM]}))
        self.assertEqual(s.backends('translator'),[
            {'device':'gpu0','card':s.devices['gpu0'].display,'api':'vulkan','framework':'llama.cpp','build':'llama-b1','weights':'q4.gguf'},
            {'device':'gpu1','card':s.devices['gpu1'].display,'api':'cuda','framework':'vllm','build':None,'weights':None}])

    def test_a_server_response_names_the_device_that_ran_it(self):
        import http.server,threading as th
        class Reply(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']));self.send_response(200)
                self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{}')
            def log_message(self,*_):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Reply);th.Thread(target=server.serve_forever,daemon=True).start()
        try:
            s=self.scheduler(self.load(translator={'names':['translator'],'backends':[LLAMA]}))
            out=self.state/'call';handle=type('Server',(),{'url':'http://127.0.0.1:%d'%server.server_port,'backend':'0123abcd-42-7'})()
            s.engine_call({'engine':'translator','path':'/v1/chat/completions','method':'POST','body':'e30='},out,handle,'gpu0')
            self.assertEqual(json.loads((out/'response.json').read_text()),
                             {'status':200,'content_type':'application/json','device':'gpu0','backend':'0123abcd-42-7'},'the process that answered')
        finally:server.shutdown()

    def test_a_calls_devices_choose_between_its_device_and_its_fallback(self):
        s=self.scheduler(self.load(translator={'names':['translator'],'backends':[LLAMA,VLLM]}))
        lane=lambda **req:s.stages('engine',{'engine':'translator',**req})[0][1]
        self.assertEqual((lane(),lane(devices=['gpu0']),lane(devices=['gpu1']),lane(devices=['gpu1','gpu0'])),('gpu0','gpu0','gpu1','gpu0'))
        self.assertEqual(s.stage_devices('engine',{'engine':'translator','devices':['gpu0']}),('gpu0',None),'no fallback to move or spill to')
        self.assertEqual(s.stage_devices('engine',{'engine':'translator','devices':['gpu1']}),(None,'gpu1'))
        s.lend('gpu0',{'owner':'lab','job_id':'lend-1'})
        self.assertEqual((lane(),lane(devices=['gpu0'])),('gpu1','gpu0'),'lent: to the fallback, unless the call allows only gpu0 (refused there)')
        with self.assertRaisesRegex(ValueError,'translator runs on gpu0 or gpu1, not on gpu2'):lane(devices=['gpu2'])

    def test_spread_is_one_tenant_per_device_answering_to_the_same_names(self):
        cfg=self.load(restore=('chat','audio'),chat={'names':['qwen3'],'placement':'spread','backends':[VLLM,LLAMA]})
        self.assertEqual(cfg['policy']['idle']['restore'],['chat@gpu1','audio'],'idle restore means the first backend')
        s=self.scheduler(cfg)
        self.assertEqual(s.model_tenants('qwen3'),['chat@gpu1','chat@gpu0'])
        self.assertEqual([s.stages('engine',{'engine':t})[0][1] for t in ('chat@gpu1','chat@gpu0')],['gpu1','gpu0'])
        self.assertEqual(cfg['tenants']['chat@gpu0']['routes'],list(base.gpu_runtime.SERVICE_ROUTES))
        self.assertNotIn('fallback',cfg['tenants']['chat@gpu0'])

    def test_validation_names_the_mistake(self):
        for services,message in (({'x':{'backends':[]}},'backends must be a list'),
                                 ({'x':{'backends':[LLAMA,LLAMA]}},'one backend per device'),
                                 ({'x':{'backends':[dict(LLAMA,port=1)]}},'unknown fields port'),
                                 ({'x':{'backends':[{'device':'gpu0','api':'vulkan'}]}},'needs command, url'),
                                 ({'x':{'placement':'random','backends':[LLAMA]}},'placement must be'),
                                 ({'x':{'backends':[LLAMA],'slots':0}},'slots must be an integer'),
                                 ({'x':{'backends':[LLAMA,dict(VLLM,device='gpu0')]}},'one backend per device'),
                                 ({'x':{'backends':[dict(VLLM,device='gpu1'),dict(LLAMA,api='cuda')]}},'gpu0 cannot run cuda'),
                                 ({'audio':{'backends':[LLAMA]}},'tenant audio is already declared'),
                                 ({'x':{'names':['audio'],'backends':[LLAMA]}},'the name audio of x is another tenant')):
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):self.load(**services)

    def test_a_slow_server_gets_its_startup_time_and_its_device_lingers(self):
        cfg=self.load(fp8={'backends':[dict(VLLM,startup_s=900)],'linger_s':600,'slots':16})
        self.assertEqual((cfg['tenants']['fp8']['linger_s'],cfg['tenants']['fp8']['startup_s']),(600,900))
        s=self.scheduler(cfg);s._server('fp8');self.assertEqual(base.FakeServer.started[-1].startup_s,900)
        s._server('audio');self.assertEqual(base.FakeServer.started[-1].startup_s,240,'others keep the default')
        for bad in ({'linger_s':0},{'linger_s':'10'},{'linger_s':10**6}):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'linger_s must be a number above 0'):self.load(x={'backends':[LLAMA],**bad})
        with self.assertRaisesRegex(ValueError,'gpu1.startup_s must be'):self.load(x={'backends':[LLAMA,dict(VLLM,startup_s=-1)]})

    def test_a_server_child_holding_the_device_is_ours(self):
        import subprocess
        s=self.scheduler(self.load(fp8={'backends':[VLLM]}));s._server('fp8')
        child=subprocess.Popen(['sh','-c','sleep 30 & wait'])
        try:
            s.servers['fp8'].process=child;time.sleep(.2)
            grandchild=[p for p in base.gpu_runtime.descendants(child.pid) if p!=str(child.pid)]
            self.assertTrue(grandchild);self.assertLessEqual(set(grandchild),s._pids(),'the engine core below the server is not foreign')
        finally:
            child.kill();child.wait()

    def test_a_config_without_services_is_untouched(self):
        cfg=base.v2();self.assertIs(expand_services(cfg),cfg)

    def test_a_session_holds_the_device_while_its_calls_run_side_by_side(self):
        cfg=self.load(translator={'backends':[LLAMA],'slots':2});s=self.scheduler(cfg)
        taken=[]
        with s.engine_session('translator','gpu0') as (server,device):
            self.assertEqual(device,'gpu0');self.assertIs(server,s.servers['translator'])
            other=threading.Thread(target=lambda:taken.append(s.device_locks['gpu0'].acquire(blocking=False)));other.start();other.join()
        self.assertEqual(taken,[False],'nothing else is placed on the device meanwhile')
        self.assertEqual(s.server_on['translator'],'gpu0')


if __name__=='__main__':unittest.main()
