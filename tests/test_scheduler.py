"""The real scheduler with fake workers: batch tenants, device environments and foreign processes,
async engine calls (`poll`), and a plugin tenant's parts and stages end to end (pair_tenant)."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import harness
from harness import SD_POLL, SD_ROUTES, icds
from gpu_runtime import Device, Scheduler, load_config
from pair_tenant import PAIR
from app_fixture import Service

ROOT=Path(__file__).resolve().parents[1]

FAKE_ENHANCER="""import json,os,subprocess,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);mode=request['mode']
child=subprocess.Popen(['sleep','60'])
(out/'pids').write_text(json.dumps([os.getpid(),child.pid]))
(out/'started').write_text(os.environ.get('CUDA_VISIBLE_DEVICES',''))
if mode=='hang':time.sleep(60)
if mode=='fail':sys.exit(3)
while not (out/'go').exists():time.sleep(.05)
child.kill()
print('progress noise on stdout')
print(json.dumps({'output':str(out/'result.mp4'),'sha256':'x','mode':mode}))
"""

def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False

class BatchTests(harness.SchedulerHarness):
    config=staticmethod(lambda:harness.v2(**{'tenants.pair':PAIR,'policy.idle.restore':['translator','pair']}))
    def setUp(self):
        super().setUp()
        self.script=self.state/'fake_enhance.py';self.script.write_text(FAKE_ENHANCER)
    def scheduler_with_batch(self,**extra):
        spec={'kind':'batch','device':'gpu1','api':'cuda','python':sys.executable,'script':str(self.script),'settings':{'model_dir':'/x'},**extra}
        return self.scheduler(harness.override(self.config(),**{'tenants.upscale':spec}))
    def wait_for(self,path,timeout=10):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if time.monotonic()>deadline:self.fail('%s never appeared'%path)
            time.sleep(.02)
    def run_job(self,s,out,mode,**kw):
        result={}
        def target():
            try:result['value']=s.batch('batch',{'tenant':'upscale','mode':mode},out,**kw)
            except Exception as exc:result['error']=exc
        thread=threading.Thread(target=target);thread.start();return thread,result

    def test_a_batch_job_evicts_only_its_own_device_and_leaves_nothing_behind(self):
        s=self.scheduler_with_batch();pair=s.plugins['pair'];pair.start('front');pair.start('back');s._server('translator')
        front_seen=pair.part('front')
        out=self.state/'job';thread,result=self.run_job(s,out,'ok')
        self.wait_for(out/'started')
        self.assertEqual((out/'started').read_text(),'GPU-1111','the job lands on its declared NVIDIA card')
        self.assertEqual(self.occupants(s),{'gpu0':['translator'],'gpu1':['upscale']},'pair.back left gpu1; gpu0 untouched')
        self.assertIn(str(json.loads((out/'pids').read_text())[1]),s._pids(),'its children count as ours in conflict detection')
        (out/'go').write_text('');thread.join(10)
        self.assertEqual(result['value']['mode'],'ok','the last stdout line is the result')
        self.assertEqual((self.occupants(s),s.batches),({'gpu0':['translator'],'gpu1':[]},{}))
        self.assertIsNone(front_seen and pair.part('front'),'the front part had already made room for translator')
        s.park();self.assertEqual(self.occupants(s)['gpu1'],['pair.back'],'the idle policy puts the plugin\'s resident part back')

    def test_cancelling_a_running_job_stops_its_whole_process_group(self):
        s=self.scheduler_with_batch();out=self.state/'job';stop=threading.Event()
        thread,result=self.run_job(s,out,'hang',cancelled=stop.is_set)
        self.wait_for(out/'pids');pids=json.loads((out/'pids').read_text())
        stop.set();thread.join(20)
        self.assertRegex(str(result['error']),'cancelled')
        self.assertFalse(any(alive(pid) for pid in pids),'the tenant and the process it started are gone')
        self.assertEqual(s.batches,{})

    def test_timeout_and_failure_are_errors_that_name_the_log(self):
        s=self.scheduler_with_batch(timeout_s=1);out=self.state/'slow'
        thread,result=self.run_job(s,out,'hang');thread.join(20)
        self.assertRegex(str(result['error']),'exceeded 1s')
        s.tenants['upscale']['timeout_s']=60;started=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'exited 3; see .*upscale.log'):s.batch('batch',{'tenant':'upscale','mode':'fail'},self.state/'failed')
        # The failed tenant left `sleep 60` behind holding the result pipe: it is killed, not waited for.
        self.assertLess(time.monotonic()-started,10)
        self.assertFalse(any(alive(pid) for pid in json.loads((self.state/'failed'/'pids').read_text())))
        with self.assertRaisesRegex(ValueError,'no batch tenant'):s.batch('batch',{'tenant':'translator'},self.state/'wrong')

class DeviceTests(unittest.TestCase):
    def test_environment_per_vendor_and_api(self):
        amd=Device('gpu0',{'vendor':'amd','pci':'0000:03:00.0'});nv=Device('gpu1',{'vendor':'nvidia','uuid':'GPU-1','vulkan_index':1})
        with icds('/usr/share/vulkan/icd.d/radeon_icd.json'):self.assertEqual(amd.env('vulkan')['DRI_PRIME'],'pci-0000_03_00_0!')
        self.assertEqual(nv.env('cuda')['CUDA_VISIBLE_DEVICES'],'GPU-1')
        with icds('/usr/share/vulkan/icd.d/nvidia_icd.json'):
            self.assertEqual((nv.env('vulkan')['VK_DRIVER_FILES'],nv.env('vulkan')['GGML_VK_VISIBLE_DEVICES']),('/usr/share/vulkan/icd.d/nvidia_icd.json','1'))
        with self.assertRaisesRegex(RuntimeError,'not supported'):amd.env('cuda')
    def test_the_icd_manifest_is_declared_or_the_first_that_exists(self):
        nv=Device('gpu1',{'vendor':'nvidia','uuid':'GPU-1'})
        with icds('/etc/vulkan/icd.d/nvidia_icd.json'):
            self.assertEqual(nv.env('vulkan')['VK_DRIVER_FILES'],'/etc/vulkan/icd.d/nvidia_icd.json','inside the container NVIDIA\'s is under /etc')
        with icds('/usr/share/vulkan/icd.d/nvidia_icd.json','/etc/vulkan/icd.d/nvidia_icd.json'):
            self.assertEqual(nv.icd_path(),'/usr/share/vulkan/icd.d/nvidia_icd.json','/usr/share first')
        with icds():
            with self.assertRaisesRegex(RuntimeError,'gpu1: no Vulkan ICD manifest nvidia_icd.json in /usr/share/vulkan/icd.d, /etc/vulkan/icd.d'):nv.env('vulkan')
            self.assertEqual(nv.icd_path(strict=False),'/usr/share/vulkan/icd.d/nvidia_icd.json','a lend\'s env: the first, as before')
        declared=Device('gpu1',{'vendor':'nvidia','uuid':'GPU-1','vulkan_icd_path':'/opt/icd/nv.json'})
        with icds('/etc/vulkan/icd.d/nvidia_icd.json'):self.assertEqual(declared.env('vulkan')['VK_DRIVER_FILES'],'/opt/icd/nv.json')
    def test_nvidia_foreign_processes_are_filtered_by_uuid_and_ownership(self):
        nv=Device('gpu1',{'vendor':'nvidia','uuid':'GPU-1'})
        with patch('gpu_runtime.subprocess.check_output',return_value='4242, GPU-1\n4343, GPU-2\n4444, GPU-1\n'),patch('gpu_runtime.describe_pid',lambda p:'pid:'+p):
            self.assertEqual(nv.foreign({'4444'},True),['pid:4242'])
    def test_amd_vram_counts_only_when_we_hold_nothing_there(self):
        amd=Device('gpu0',{'vendor':'amd','pci':'0000:03:00.0','name':'7900 XTX'})
        with patch.object(Device,'sysfs') as sysfs:
            sysfs.return_value.__truediv__.return_value.read_text.return_value=str(4*1024**3)
            self.assertEqual(amd.foreign(set(),False),['7900 XTX VRAM occupied (>512MiB)'])
            self.assertEqual(amd.foreign(set(),True),[])

class Reply(io.BytesIO):
    def __init__(self,status,data):
        super().__init__(json.dumps(data).encode());self.status=status;self.headers={'Content-Type':'application/json'}

class NativeImageTests(unittest.TestCase):
    def test_native_accepted_job_holds_phase_until_final_result(self):
        runtime=object.__new__(Scheduler);runtime.control=threading.RLock();runtime.tenants={'sd':{'device':'gpu0','routes':SD_ROUTES,'poll':SD_POLL}}
        runtime.device_locks={'gpu0':threading.RLock()};runtime.devices={'gpu0':SimpleNamespace(vendor='amd',power_w=None)};runtime.uses={};runtime.used_at={};runtime.loading={};runtime.freed=threading.Condition(runtime.control)
        runtime._server=lambda name,lane=None:type('Server',(),{'url':'http://127.0.0.1:19020'})()
        locked=[];calls=[]
        @contextlib.contextmanager
        def phase(device):
            self.assertEqual(device,'gpu0','the engine call holds its own device, not the host')
            locked.append(True)
            try:yield
            finally:locked.pop()
        runtime._device_phase=phase
        def urlopen(request,timeout):
            self.assertTrue(locked);calls.append(request)
            return [Reply(202,{'poll_url':'/sdcpp/v1/jobs/native'}),Reply(200,{'status':'running'}),Reply(200,{'status':'done','result':{'image':'fixture'}})][len(calls)-1]
        with tempfile.TemporaryDirectory() as tmp,patch('gpu_runtime.urllib.request.urlopen',side_effect=urlopen),patch('gpu_runtime.time.sleep'):
            out=Path(tmp)/'result';runtime.engine({'engine':'sd','path':'/sdcpp/v1/img_gen','method':'POST','body':'','content_type':'application/json'},out)
            self.assertEqual(len(calls),3);self.assertEqual(json.loads((out/'response.bin').read_text())['status'],'done')
            self.assertEqual(locked,[])


class PluginTests(harness.SchedulerHarness):
    """A plugin tenant (pair_tenant) through the real scheduler and job service: its job's stages run
    on its parts' devices, its parts are evicted, lent around and restored like any tenant."""
    config=staticmethod(lambda:harness.v2(**{'tenants.pair':PAIR,'tenants.pair.front.fallback':'gpu1','policy.idle.restore':['translator','pair']}))

    def test_a_job_runs_its_stages_each_on_its_parts_device(self):
        s=self.scheduler();service=Service(ROOT,self.state,s)
        try:
            job=service.call('pair',{'request':{'n':1},'request_key':'k'})
            self.assertEqual((job['step'],job['lane']),('first','gpu0'))
            done=service.call('job_wait',{'job_id':job['id'],'timeout':10})
            self.assertEqual(done['status'],'completed',done.get('error'))
            self.assertEqual(done['result']['request'],{'n':1},'the last stage\'s result is the job\'s')
            front,back=[w for w in harness.FakeWorker.started if w.command in (['front'],['back'])][:2]
            self.assertEqual((front.command,back.command),(['front'],['back']))
            self.assertIn('DRI_PRIME',front.env);self.assertEqual(back.env['CUDA_VISIBLE_DEVICES'],'GPU-1111')
            status=s.status()
            self.assertEqual(status['parts']['pair.back'],back.process.pid)
            self.assertEqual(status['pair']['back_device'],'gpu1')
        finally:service.close()

    def test_its_parts_are_evicted_lent_around_and_restored_like_any_tenant(self):
        s=self.scheduler();pair=s.plugins['pair']
        front=pair.start('front');s._server('translator')
        self.assertTrue(front.stopped,'an exclusive card: the server evicts the part');self.assertIsNone(pair.part('front'))
        s.lend('gpu0',{'owner':'lab','job_id':'lend-1'})
        self.assertEqual(s.stages('pair',{}),[('first','gpu1'),('second','gpu1')],'its device lent: the first stage goes to the part\'s fallback')
        self.assertEqual(s.stage_devices('pair',{}),('gpu0','gpu1'))
        s.reclaim('gpu0');s.park()
        self.assertEqual(self.occupants(s),{'gpu0':['translator'],'gpu1':['pair.back']},'the idle policy restores its home part only')
        s.stop();self.assertEqual(s.parts,{})

    def test_a_job_kind_is_run_by_one_tenant_only(self):
        path=self.state/'gpu-config.json'
        path.write_text(json.dumps(harness.v2(**{'tenants.pair':PAIR,'tenants.other':PAIR})))
        with self.assertRaisesRegex(ValueError,'job kind pair is already run by pair'):load_config(path)
        path.write_text(json.dumps(harness.v2(**{'tenants.pair':PAIR,'tenants.pair.back.fallback':'gpu0'})))
        with self.assertRaisesRegex(ValueError,'the back part has no fallback'):load_config(path)


if __name__=='__main__':unittest.main()
