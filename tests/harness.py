"""The scheduler without a GPU, for the tests: workers are fakes and device preconditions stubs.

V2 is a two-card host (an AMD card and an NVIDIA one) with three server tenants on the AMD card;
v2(**overrides) copies it with dotted keys replaced, e.g. v2(**{'tenants.audio.device':'gpu1'}).
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import gpu_runtime   # noqa: F401 (base.gpu_runtime in the tests)
from gpu_runtime import Device, Scheduler

AUDIO_ROUTES=['/health','/v1/models','/v1/audio/transcriptions','/v1/audio/transcriptions/details','/v1/audio/speech','/v1/tasks/run','/v1/tasks/unload_models']
SD_ROUTES=['/health','/v1/models','/sdcpp/v1/capabilities','/sdcpp/v1/img_gen','/v1/images/generations','/v1/images/edits']
SD_POLL={'submit':'/sdcpp/v1/img_gen','field':'poll_url','prefix':'/sdcpp/v1/jobs/'}
V2={'schema':2,
    'devices':{'gpu0':{'pci':'0000:03:00.0','name':'7900 XTX','vendor':'amd','power_cap_w':280,'vulkan_icd':'radeon'},'gpu1':{'uuid':'GPU-1111','vendor':'nvidia'}},
    'tenants':{'audio':{'kind':'server','device':'gpu0','api':'vulkan','url':'http://127.0.0.1:19021','cwd':'/x','command':['audio'],'routes':AUDIO_ROUTES},
               'sd':{'kind':'server','device':'gpu0','api':'vulkan','url':'http://127.0.0.1:19020','cwd':'/x','command':['sd'],'health':'/sdcpp/v1/capabilities',
                     'routes':SD_ROUTES,'poll':SD_POLL},
               'translator':{'kind':'server','device':'gpu0','api':'vulkan','url':'http://127.0.0.1:18000','cwd':'/x','command':['llama'],'env':{'LD_LIBRARY_PATH':'/x'},
                             'routes':['/health','/v1/models','/v1/chat/completions']}},
    'policy':{'idle':{'restore':['translator']}}}


def override(cfg, **overrides):
    """`cfg` copied, with each dotted key set to its value."""
    cfg=json.loads(json.dumps(cfg))
    for key,value in overrides.items():
        target=cfg
        *path,last=key.split('.')
        for p in path:target=target[p]
        target[last]=json.loads(json.dumps(value))   # a later key may change inside it: never the caller's object
    return cfg


def v2(**overrides):return override(V2,**overrides)


class FakeProcess:
    _next=[100]
    def __init__(self):FakeProcess._next[0]+=1;self.pid=FakeProcess._next[0]
class FakeServer:
    started=[]
    def __init__(self,command,log,lease_fd,env,cwd,url,health='/health',startup_s=240):
        self.command=command;self.env=env;self.url=url;self.startup_s=startup_s;self.process=FakeProcess();self.stopped=False;FakeServer.started.append(self)
    def stop(self):self.stopped=True
def icds(*present):
    """Only these ICD manifests exist (Device.icd_path looks before it chooses)."""
    return patch('gpu_runtime.os.path.exists',lambda path:path in present)

_icd_path=Device.icd_path
class FakeWorker:
    started=[]
    def __init__(self,command,log,lease_fd,env=None):
        self.command=command;self.env=env;self.process=FakeProcess();self.ready={'event':'ready'};self.stopped=False;FakeWorker.started.append(self)
    def run(self,request,timeout=None):return {'event':'result','request':request}
    def stop(self):self.stopped=True

class SchedulerHarness(unittest.TestCase):
    config=staticmethod(v2)   # the default config of scheduler()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name)
        FakeServer.started.clear();FakeWorker.started.clear()
        self.patches=[patch.object(Scheduler,'server_worker',FakeServer),patch.object(Scheduler,'worker',FakeWorker),
                      patch.object(Device,'check',lambda self:'/cap' if self.vendor=='amd' else ''),
                      # Not this host's ICD files: the first candidate unless a test says which exist.
                      patch.object(Device,'icd_path',lambda self,strict=True:_icd_path(self,False)),
                      patch.object(Device,'foreign',lambda self,owned,occupied:[]),
                      patch.dict('os.environ',{'GPU_RUNTIME_LOCK':str(self.state/'mode.lock'),'GPU_PHASE_LOCK':str(self.state/'phase.lock')}),
                      *self.more_patches()]
        for p in self.patches:p.start()
    def more_patches(self):return []
    def tearDown(self):
        for p in self.patches:p.stop()
        self.temp.cleanup()
    def scheduler(self,cfg=None):return Scheduler(self.state.parent,self.state,cfg or self.config())
    def occupants(self,s):return {name:s._occupants(d) for name,d in s.devices.items()}
