"""The GPU scheduler: devices and tenants from `gpu-config.json`, one lease, one placement rule.

Nothing about a particular card or workload lives in this file. A device is declared with its
vendor and how it is addressed (an AMD card by PCI address, an NVIDIA card by UUID); a tenant is
declared with the device it needs and the API it speaks (vulkan or cuda). From those two facts the
scheduler derives the environment a process needs to land on that device, what must be true
before it may start (an AMD power cap, the card actually present), and who has to leave first:
a device's `placement` is `exclusive` (placing a tenant evicts every other tenant on it) or
`lru` (tenants stay side by side while the card has room for the newcomer's declared `vram_gib`,
measured, and otherwise leave least recently used first; one that declares no `vram_gib` is placed
as on an exclusive card, and nothing is placed beside it).

A `server` tenant is an HTTP server called through the job service (an engine call). A `batch`
tenant is a fresh process per job, placed the same way for exactly the length of that job. Any
other kind is run by a plugin (TenantKind): a tenant made of parts, each placed like a server on
its own declared device, whose jobs are stages on those devices (say, a video model's text encoder
on one card and its diffusion transformer on another).

Each device is worked independently: the job service runs one queue per device, and a job that
needs two devices is two stages, each queued on its own device, so nothing holds one card while it
waits for the other. Work on a device runs under that device's lock; loading a model, and a batch
job heavy on host RAM, also take the host-wide phase lock, because the host's RAM is shared. A
device can be lent out (research) and is then left alone until it is reclaimed; a device with a
process on it that is not ours is blocked, never killed, until that process is gone. Every server
or batch tenant, and every plugin part, may declare a `fallback` device: while its own device is
lent or blocked, its work is queued on the fallback instead, and the scheduler tracks which device
each one actually occupies. The idle policy only ever restores a tenant on its own device.

A `service` in the config is what callers name instead of a device (expand_services): it becomes
one tenant (`prefer`: a device and a fallback, each launched its own way) or one per device
(`spread`), and a tenant with `slots` serves that many calls at once in one engine_session.

Swapping a card therefore means editing the config, not this module.
"""
import base64
import contextlib
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import backend_record
import text_rates
from workers import ServerWorker, Worker, describe_pid

# A server tenant's routes when it declares none (`routes`): an OpenAI-style server.
SERVICE_ROUTES=('/health','/v1/models','/v1/chat/completions','/v1/completions','/v1/embeddings')
LAUNCH_KEYS=('api','command','url','cwd','env','health','startup_s','vram_gib')   # what a service backend may set per device
VRAM_IDLE_BYTES=512*1024*1024
GIB=1024**3
VRAM_MARGIN_GIB=0.5    # kept free on an lru card beyond what the newcomer declares
VRAM_SETTLE_S=10       # how long an evicted process's memory may take to show as free
SERVER_STARTUP_S=240   # how long a server tenant may take to answer its health route; `startup_s` raises it
TENANT_BUSY_WAIT_S=930 # how long moving a tenant waits for a call using it on another device (one call: up to 900 s)
# Where a Vulkan ICD manifest is looked for, in order: Mesa's RADV is under /usr/share; inside the
# container the NVIDIA container toolkit puts NVIDIA's under /etc.
VULKAN_ICD_DIRS=('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d')

# The job kinds the scheduler runs itself: a call to a server tenant, a lend, and a run of a batch
# tenant. BATCH_JOBS: each batch job kind -> the request field naming its tenant (an application may
# add its own name for them, e.g. BATCH_JOBS['enhance']='enhancer').
BATCH_JOBS={'batch':'tenant'}
CORE_JOBS={'engine','lend'}


def tenant_name(kind, request):
    """The tenant an engine call or a batch job names, else None."""
    return request.get('engine') if kind=='engine' else request.get(BATCH_JOBS[kind]) if kind in BATCH_JOBS else None


class TenantKind:
    """A tenant kind beyond `server` and `batch`, run by a plugin: the module `<kind>_tenant` on the
    Python path (or the one the tenant's `plugin` names), whose `Tenant` is a subclass of this.

    Such a tenant is made of parts (`parts`), each declared like a server tenant's slot (device,
    api, optional fallback, vram_gib, on) and placed like one, and it runs the job kinds in `jobs`,
    each a list of stages, one device per stage, queued one after the other. It starts its parts'
    processes itself, with the scheduler's own steps (_lease, _usable, _leave and _started, _place,
    _using, _phase, _device_phase, _log, worker), and hands each to the scheduler (hold), which from
    then on evicts, stops and accounts for it as `<tenant>.<part>` like any other tenant."""
    jobs=()          # the job kinds it runs

    @staticmethod
    def parts(spec):
        """part -> its slot in `spec`."""
        return {}

    @staticmethod
    def validate(name, spec, devices):
        """Refuse what this kind cannot run (a ValueError naming the mistake); the slots of its parts
        are checked by the scheduler already."""

    def __init__(self, scheduler, name, spec):self.scheduler=scheduler;self.name=name;self.spec=spec
    def stages(self, kind, request):
        """[(stage, device)] of one job."""
        raise NotImplementedError
    def first_slot(self, kind, request):
        """The slot whose device (or fallback) the job's first stage waits on."""
        raise NotImplementedError
    def part_of(self, kind, stage):
        """The part a stage runs on, `<tenant>.<part>` (a device's affinity queue looks for it)."""
        raise NotImplementedError
    def run(self, kind, stage, request, out, lane):
        """Run one stage on device `lane`; the last stage's result is the job's."""
        raise NotImplementedError
    def resident(self, device):
        """The stage whose part is loaded on `device`, or None (a queue prefers its jobs)."""
        return None
    def home(self):
        """The device policy.idle.restore puts it back on."""
        raise NotImplementedError
    def restore(self):
        """Put back what it keeps resident while its device is idle."""
    def status(self):return {}
    def ready(self):return {}


def tenant_kind(name, spec):
    """The TenantKind class that runs tenant `name`."""
    module=spec.get('plugin') or spec['kind']+'_tenant'
    try:found=importlib.import_module(module).Tenant
    except (ImportError,AttributeError) as exc:
        raise ValueError('tenant %s: kind must be server, batch or one a plugin runs (%s: %s)'%(name,module,exc))
    if not (isinstance(found,type) and issubclass(found,TenantKind)):raise ValueError('tenant %s: %s.Tenant is no TenantKind'%(name,module))
    return found


def validate_engine(request, routes=SERVICE_ROUTES):
    """An engine call to one of `routes`, the routes of the tenant it calls, with a base64 body."""
    if request['path'] not in routes or request['method'] not in ('GET','POST'):
        raise ValueError('unsupported engine route')
    base64.b64decode(request['body'],validate=True)


def expand_services(config):
    """`services` become server tenants; everything after works on tenants only.

    A service is one thing callers name — by any of its `names`, as /engines/<name> or as the
    model of a /v1 call — never a device. Its `backends` say how it is launched on each device,
    in order of preference, each its own way (llama.cpp over Vulkan on one card, vLLM over CUDA on
    another). `placement` "prefer" (the default) runs it on the first backend's device, and on the
    second's while the first is lent or blocked: one tenant named like the service, the second
    backend its fallback with its own launch (`on`). "spread" runs it on every backend's device at
    once: one tenant per backend, `<service>@<device>`, and each call goes to the least busy.
    `slots`, `routes` and `linger_s` apply to every backend. In policy.idle.restore a spread service
    means its first backend."""
    services=config.get('services') or {}
    if not services:return config
    tenants=dict(config['tenants']);restore=list(config.get('policy',{}).get('idle',{}).get('restore',[]))
    for name,service in services.items():
        key='service '+name+': '
        unknown=set(service)-{'names','placement','backends','slots','routes','linger_s','spill','poll'}
        if unknown:raise ValueError(key+'unknown fields '+', '.join(sorted(unknown)))
        placement=service.get('placement','prefer');backends=service.get('backends')
        if placement not in ('prefer','spread'):raise ValueError(key+'placement must be prefer or spread')
        if not isinstance(backends,list) or not backends or not all(isinstance(b,dict) and b.get('device') for b in backends):
            raise ValueError(key+'backends must be a list of {device, api, command, url, ...}')
        if len({b['device'] for b in backends})!=len(backends):raise ValueError(key+'one backend per device')
        if placement=='prefer' and len(backends)>2:raise ValueError(key+'prefer takes two backends at most: a device and its fallback')
        for backend in backends:
            unknown=set(backend)-{'device',*LAUNCH_KEYS}
            if unknown:raise ValueError(key+'backend '+backend['device']+': unknown fields '+', '.join(sorted(unknown)))
            missing=[k for k in ('api','command','url') if not backend.get(k)]
            if missing:raise ValueError(key+'backend '+backend['device']+' needs '+', '.join(missing))
        common={'kind':'server','service':name,'models':list(service.get('names',[name])),
                'routes':list(service.get('routes') or SERVICE_ROUTES)}
        for key in ('slots','linger_s','spill','poll'):
            if key in service:common[key]=service[key]
        if placement=='prefer':
            first,*rest=backends;spec={**common,**first}
            if rest:spec.update(fallback=rest[0]['device'],on={rest[0]['device']:{k:v for k,v in rest[0].items() if k!='device'}})
            made={name:spec}
        else:
            made={name+'@'+b['device']:{**common,**b} for b in backends}
            restore=[name+'@'+backends[0]['device'] if n==name else n for n in restore]
        for tenant in made:
            if tenant in tenants:raise ValueError(key+'tenant '+tenant+' is already declared')
        tenants.update(made)
    for tenant,spec in tenants.items():
        for alias in spec.get('models',[]):
            if alias in tenants and tenants[alias].get('service',alias)!=spec.get('service',tenant):
                raise ValueError('the name '+alias+' of '+spec.get('service',tenant)+' is another tenant')
    policy=dict(config.get('policy',{}))
    if restore:policy['idle']={**policy.get('idle',{}),'restore':restore}
    return {**config,'tenants':tenants,'policy':policy}


def load_config(path):
    config=json.loads(Path(path).read_text())
    if config.get('schema')!=2:raise ValueError('gpu-config.json must declare "schema": 2')
    config=expand_services(config)
    devices=config['devices'];tenants=config['tenants']
    for name,spec in devices.items():
        if spec.get('vendor') not in ('amd','nvidia'):raise ValueError('device '+name+': vendor must be amd or nvidia')
        if spec['vendor']=='amd' and not spec.get('pci'):raise ValueError('device '+name+': an AMD device is addressed by pci')
        if spec['vendor']=='nvidia' and not spec.get('uuid'):raise ValueError('device '+name+': an NVIDIA device is addressed by uuid')
        if spec.get('sharing','exclusive')!='exclusive':
            raise ValueError('device '+name+': "sharing" is replaced by "placement": exclusive, or lru with vram_gib per tenant')
        if spec.get('placement','exclusive') not in ('exclusive','lru'):raise ValueError('device '+name+': placement must be exclusive or lru')
        if spec.get('queue','priority') not in ('priority','fifo','affinity'):raise ValueError('device '+name+': queue must be priority, fifo or affinity')
    jobs={}   # job kind -> the tenant that runs it
    for name,spec in tenants.items():
        kind=spec.get('kind','server')
        plugin=None if kind in ('server','batch') else tenant_kind(name,spec)
        parts={'':spec} if plugin is None else plugin.parts(spec)   # part -> slot, '' for the tenant itself
        if kind=='batch':
            missing=[k for k in ('python','script','settings') if not spec.get(k)]
            if missing:raise ValueError('tenant '+name+': a batch tenant needs '+', '.join(missing))
        if kind!='server' and {'slots','models','linger_s','startup_s','poll'}&set(spec):
            raise ValueError('tenant '+name+': only a server tenant takes slots, models, linger_s, startup_s or poll')
        poll=spec.get('poll')
        if poll is not None and not (isinstance(poll,dict) and set(poll)=={'submit','field','prefix'} and all(isinstance(v,str) and v for v in poll.values())
                                     and poll['prefix'].startswith('/') and poll['submit'] in (spec.get('routes') or SERVICE_ROUTES)):
            raise ValueError('tenant '+name+': poll must be {"submit": one of its routes, "field": the reply field naming the job, "prefix": "/path/of/its/jobs/"}')
        if not isinstance(spec.get('spill',False),bool):raise ValueError('tenant '+name+': spill must be true or false')
        for key,most in (('linger_s',86400),('startup_s',3600),('vram_gib',1024)):
            for where,value in [(key,spec.get(key))]+[(device+'.'+key,launch.get(key)) for device,launch in spec.get('on',{}).items()]\
                    +[(part+'.'+key,slot.get(key)) for part,slot in parts.items() if part]:
                if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<value<=most):
                    raise ValueError('tenant %s: %s must be a number above 0, at most %d'%(name,where,most))
        calls=spec.get('slots',1)
        if type(calls)!=int or not 1<=calls<=64:raise ValueError('tenant '+name+': slots must be an integer from 1 to 64')
        models=spec.get('models',[])
        if not isinstance(models,list) or not all(isinstance(m,str) and m.strip()==m and m for m in models):
            raise ValueError('tenant '+name+': models must be a list of model names')
        if plugin is not None:
            plugin.validate(name,spec,devices)
            for job in plugin.jobs:
                if job in jobs or job in CORE_JOBS:raise ValueError('tenant %s: job kind %s is already run by %s'%(name,job,jobs.get(job,'the scheduler')))
                jobs[job]=name
        for part,slot in parts.items():
            if slot.get('device') not in devices:raise ValueError('tenant '+name+': unknown device '+str(slot.get('device')))
            if slot.get('api') not in ('vulkan','cuda'):raise ValueError('tenant '+name+': api must be vulkan or cuda')
            fallback=slot.get('fallback')
            if fallback is None:continue
            key='tenant '+name+': '+(part+'.' if part else '')+'fallback'
            if fallback not in devices or fallback==slot['device']:
                raise ValueError(key+' must name another device, not '+repr(fallback))
            api=slot.get('on',{}).get(fallback,{}).get('api',slot['api'])
            if api not in ('vulkan','cuda'):raise ValueError('tenant '+name+': api must be vulkan or cuda')
            if api=='cuda' and devices[fallback]['vendor']!='nvidia':
                raise ValueError(key+' '+fallback+' cannot run cuda (it is '+devices[fallback]['vendor']+')')
    for name in config.get('policy',{}).get('idle',{}).get('restore',[]):
        if name not in tenants:raise ValueError('policy.idle.restore names unknown tenant '+name)
    return config


def kill_group(pgid):
    try:os.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError:pass


def descendants(pid):
    """A process and every process below it, read from /proc (the runtime image has no ps)."""
    children={}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:fields=stat.read_text().rsplit(')',1)[1].split()
        except OSError:continue
        children.setdefault(fields[1],[]).append(stat.parent.name)
    found={str(pid)};frontier=[str(pid)]
    while frontier:
        for child in children.get(frontier.pop(),[]):
            if child not in found:found.add(child);frontier.append(child)
    return found


class Device:
    """One physical card: how to land on it, what must hold first, who else is on it."""
    def __init__(self, name, spec):
        self.name=name;self.vendor=spec['vendor'];self.pci=spec.get('pci');self.uuid=spec.get('uuid')
        self.display=spec.get('name',name);self.placement=spec.get('placement','exclusive');self.queue=spec.get('queue','priority')
        self.vram_margin=spec.get('vram_margin_gib',VRAM_MARGIN_GIB)*GIB
        self.power_cap_w=spec.get('power_cap_w');self.vulkan_icd=spec.get('vulkan_icd','radeon' if self.vendor=='amd' else 'nvidia')
        self.vulkan_index=spec.get('vulkan_index',0);self.vulkan_icd_path=spec.get('vulkan_icd_path')

    def sysfs(self):return Path('/sys/bus/pci/devices')/self.pci

    def check(self):
        """Preconditions; returns the power cap file path for a capped AMD device, else ''."""
        if self.vendor=='amd':
            if self.sysfs().joinpath('vendor').read_text().strip()!='0x1002':raise RuntimeError(self.name+': configured as AMD but the PCI vendor differs')
            if self.power_cap_w is None:return ''
            caps=list((self.sysfs()/'hwmon').glob('hwmon*/power1_cap'))
            if len(caps)!=1 or not 0<int(caps[0].read_text())<=self.power_cap_w*1000000:
                raise RuntimeError(self.name+' power cap must be configured <='+str(self.power_cap_w)+'W')
            return str(caps[0])
        present=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
        if self.uuid not in present:raise RuntimeError(self.name+': '+self.uuid+' is not present (nvidia-smi lists '+', '.join(present)+')')
        return ''

    def icd_path(self, strict=True):
        """The declared manifest, else the first VULKAN_ICD_DIRS entry that has `<icd>_icd.json`.
        None found: a clear refusal, or with strict=False the first candidate (a lend's env)."""
        if self.vulkan_icd_path:return self.vulkan_icd_path
        found=[d+'/'+self.vulkan_icd+'_icd.json' for d in VULKAN_ICD_DIRS]
        path=next((p for p in found if os.path.exists(p)),None)
        if path is None and strict:
            raise RuntimeError('%s: no Vulkan ICD manifest %s_icd.json in %s (declare vulkan_icd_path)'%(self.name,self.vulkan_icd,', '.join(VULKAN_ICD_DIRS)))
        return path or found[0]

    def env(self, api, strict=True):
        env=os.environ.copy()
        if api=='vulkan':
            env.update(VK_DRIVER_FILES=self.icd_path(strict),GGML_VK_VISIBLE_DEVICES=str(self.vulkan_index))
            if self.vendor=='amd':env['DRI_PRIME']='pci-'+self.pci.replace(':','_').replace('.','_')+'!'
            return env
        if api=='cuda' and self.vendor=='nvidia':
            env.update(CUDA_VISIBLE_DEVICES=self.uuid,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');return env
        raise RuntimeError(self.name+': api '+api+' on vendor '+self.vendor+' is not supported by this runtime')

    def vram(self):
        """(total, used) bytes of the card's memory, whoever uses it."""
        if self.vendor=='amd':
            return tuple(int((self.sysfs()/('mem_info_vram_'+k)).read_text()) for k in ('total','used'))
        out=subprocess.check_output(['nvidia-smi','-i',self.uuid,'--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
        return tuple(int(float(v))*1024*1024 for v in out.split(','))

    def foreign(self, owned, occupied):
        """Processes on this device that are not ours. Deliberately without the docker daemon:
        nvidia-smi reports compute processes inside containers too (as host pids), AMD VRAM
        occupancy is indifferent to containers, and /proc names the container."""
        if self.vendor=='nvidia':
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
            found=[]
            for line in output.splitlines():
                parts=[p.strip() for p in line.split(',')]
                if not parts[0]:continue
                if len(parts)>1 and parts[1] and parts[1]!=self.uuid:continue
                if parts[0] not in owned:found.append(describe_pid(parts[0]))
            return found
        # Resident workloads as well as busy compute; desktop graphics remain allowed.
        if not occupied and int((self.sysfs()/'mem_info_vram_used').read_text())>VRAM_IDLE_BYTES:
            return [self.display+' VRAM occupied (>512MiB)']
        return []


class Scheduler:
    server_worker=ServerWorker
    worker=Worker

    def __init__(self, root, state, config=None):
        self.root,self.state=Path(root),Path(state)
        self.control=threading.RLock()
        self.config=config or load_config(self.state/'gpu-config.json')
        self.devices={name:Device(name,spec) for name,spec in self.config['devices'].items()}
        self.tenants=self.config['tenants']
        # The lease: held while anything of ours is on a GPU (other programs on the host may take it
        # shared to know). The phase lock is host-wide: model loads and host-RAM-heavy batch jobs,
        # one at a time; each device's compute has its own beside it (device_phase_path).
        self.mode_path=Path(os.environ.get('GPU_RUNTIME_LOCK') or self.state/'runtime.lock')
        self.phase_path=os.environ.get('GPU_PHASE_LOCK') or str(self.state/'phase.lock')
        self.lease=None;self.servers={};self.batches={}
        self.server_on={};self.batch_on={}   # tenant -> the device it actually occupies
        self.parts={}                        # a plugin tenant's part -> (worker, the device it occupies)
        self.ready=None;self.park_error=None
        self.recorder=backend_record.Recorder(self.state/'backends',descendants)   # what each started server is (X-GPU-Backend)
        # Work on one device never waits for work on another: each has its own lock.
        self.device_locks={name:threading.RLock() for name in self.devices}
        self.lent={};self.blocked={}   # device -> who borrowed it / why it is unusable
        self.drained=set()             # devices taking no new work (maintenance); see lane_for
        self.used_at={}                # tenant -> when it was placed or last used (lru placement evicts the least recent)
        # (tenant, device) -> calls using it there now, counted from before it is placed there: another
        # device's queue does not move it meanwhile (see _leave). `freed` is signalled as each one ends.
        self.uses={};self.freed=threading.Condition(self.control)
        self.loading={}                # tenant -> the device a queue is starting it on (E22: one process per tenant)
        # Tenants of a kind a plugin runs (TenantKind), and the job kinds they run.
        self.plugins={name:tenant_kind(name,spec)(self,name,spec) for name,spec in self.tenants.items() if spec.get('kind','server') not in ('server','batch')}
        self.kinds={job:plugin for plugin in self.plugins.values() for job in plugin.jobs}

    # ---- what is where ------------------------------------------------------------------
    @property
    def lanes(self):return list(self.devices)

    def stages(self, kind, request):
        """The devices a job needs, in order, one (stage, device) per stage.

        A plugin tenant's job may need several devices (an encoder's, then a diffusion model's),
        each a stage queued after the other, so that it never holds one while waiting for another."""
        if kind in self.kinds:return self.kinds[kind].stages(kind,request)
        if kind=='lend':
            if request.get('device') not in self.devices:raise ValueError('unknown device %r; this host has %s'%(request.get('device'),', '.join(self.devices)))
            return [('lend',request['device'])]
        slot=self._tenant_slot(kind,request);lane=self.lane_for(slot);allowed=request.get('devices')
        if allowed and lane not in allowed:
            # A call that may run only on some devices (e.g. for bit-identical output): never its
            # fallback unless allowed; its own device, lent or drained, refuses or makes it wait.
            choices=[d for d in (slot['device'],slot.get('fallback')) if d and d in allowed]
            if not choices:raise ValueError('%s runs on %s, not on %s'%(tenant_name(kind,request),
                                            ' or '.join(d for d in (slot['device'],slot.get('fallback')) if d),' or '.join(allowed)))
            lane=choices[0]
        return [(kind,lane)]

    def _tenant_slot(self, kind, request):
        """The server tenant an engine call goes to, the batch tenant a batch job runs."""
        name=tenant_name(kind,request)
        if name not in self.tenants or name in self.plugins:raise ValueError('no tenant named '+str(name))
        return self.tenants[name]

    def lane_for(self, slot):
        """Where a slot's next job is queued: its own device, or its fallback while the own one is
        lent, blocked or drained and the fallback is not. Otherwise the own device, whose refusal or wait
        then applies as before."""
        primary,fallback=slot['device'],slot.get('fallback')
        down=lambda name:name in self.lent or name in self.blocked or name in self.drained
        return fallback if fallback and down(primary) and not down(fallback) else primary

    def part_of(self, kind, stage):
        """The part a plugin job's stage runs on, as loaded_on names it."""
        return self.kinds[kind].part_of(kind,stage)

    def stage_devices(self, kind, request):
        """(own device, fallback or None) of the stage a job waits in first: the two lanes a
        waiting job moves between. A lend has no fallback; a call's `devices` leave out the others."""
        if kind in self.kinds:slot=self.kinds[kind].first_slot(kind,request)
        elif kind=='engine' or kind in BATCH_JOBS:slot=self._tenant_slot(kind,request)
        else:return request.get('device'),None
        allowed=request.get('devices') or (slot['device'],slot.get('fallback'))
        primary,fallback=slot['device'],slot.get('fallback')
        return (primary if primary in allowed else None),(fallback if fallback in allowed else None)

    def spills(self, kind, request):
        """Whether a waiting job of this tenant may be taken by its idle fallback device while its own
        device is busy with something else (gpu-config `spill`; off by default, never a plugin's stage)."""
        if kind!='engine' and kind not in BATCH_JOBS:return False
        try:return bool(self._tenant_slot(kind,request).get('spill'))
        except ValueError:return False

    def located(self, name):
        """The device tenant `name` lives on now, or None."""
        with self.control:
            if name in self.servers:return self.server_on.get(name)
            if name in self.batches:return self.batch_on.get(name)
            if name in self.parts:return self.parts[name][1]
        return None

    def resident(self, name):
        """Which plugin stage's part is loaded on a device (TenantKind.resident), or None."""
        with self.control:
            return next((stage for stage in (plugin.resident(name) for plugin in self.plugins.values()) if stage),None)

    def validate_engine(self, request):
        spec=self.tenants.get(request.get('engine'))
        if not spec or spec.get('kind','server')!='server':raise ValueError('no server tenant named %r'%request.get('engine'))
        validate_engine(request,tuple(spec.get('routes') or SERVICE_ROUTES))

    def _slot_device(self, slot):return self.devices[slot['device']]

    def loaded_on(self, device):
        """The tenants resident on `device` now (a plugin tenant's parts as <tenant>.<part>)."""
        with self.control:return set(self._occupants(self.devices[device]))

    def _usable(self, device):
        """Refuse to place anything of ours on a lent or blocked device."""
        if device.name in self.lent:
            raise RuntimeError('%s (%s) is lent to %s'%(device.name,device.display,self.lent[device.name].get('owner','research')))
        if device.name in self.blocked:
            raise RuntimeError('%s (%s) is blocked: %s'%(device.name,device.display,self.blocked[device.name]))

    def _occupants(self, device):
        names=[n for n in self.tenants if self.server_on.get(n,self.batch_on.get(n))==device.name
               and (n in self.servers or n in self.batches)]
        names+=[part for part,(_,on) in self.parts.items() if on==device.name]
        return names

    def _stop_tenant(self, name, device=None):
        """Stop one tenant. With `device`, only if it is still on that device: another queue may
        have moved it since the caller listed the occupants."""
        worker=self._detach(name,device)
        if worker:worker.stop()
        elif name in self.batches:
            self._stop_batch(name)

    def _detach(self, name, device=None):
        """Take one tenant off the books (see _stop_tenant); returns its worker, for the caller to stop."""
        with self.control:
            if name in self.parts:
                if device is not None and self.parts[name][1]!=device:worker=None
                else:worker=self.parts.pop(name)[0]
            elif name in self.servers:
                if device is not None and self.server_on.get(name)!=device:worker=None
                else:worker=self.servers.pop(name);self.server_on.pop(name,None)
            else:
                worker=None
            self.used_at.pop(name,None)
        return worker

    def _leave(self, name, device, wait=True):
        """Stop the server or plugin part `name` wherever it is, before it is placed on `device`. While
        a call uses it on another device it is not stopped (invariant 2), and while another device's
        queue is starting it there is no second start (E22): wait until that call or start is done, at
        most TENANT_BUSY_WAIT_S, or with wait=False leave it there and return False. Waited for, taken
        off the books and marked as starting on `device` under one hold of `control`, so no call starts
        using it and no other queue starts it in between; the caller ends the start with _started.
        This waits holding `device`'s lock only, which the call on the other device never needs
        (invariant 1). Do not take that device's lock here as well: the two would wait on each other."""
        deadline=time.monotonic()+TENANT_BUSY_WAIT_S
        with self.control:
            while True:
                where=self.located(name)
                starting=self.loading.get(name)
                if starting not in (None,device):busy='being started on '+starting
                elif where not in (None,device) and self.uses.get((name,where)):busy='in use on '+where
                else:break
                if not wait:return False
                left=deadline-time.monotonic()
                if left<=0:raise RuntimeError('%s is %s; not moved to %s after waiting %d s'%(name,busy,device,TENANT_BUSY_WAIT_S))
                self.freed.wait(left)
            worker=self._detach(name);self.loading[name]=device
        if worker:worker.stop()
        return True

    def _started(self, name):
        """End of a start begun by _leave, whether it succeeded or not: other queues may go on."""
        with self.control:self.loading.pop(name,None);self.freed.notify_all()

    @contextlib.contextmanager
    def _using(self, name, device):
        """`name` is in use on `device` while the block runs (see _leave)."""
        with self.control:self.uses[name,device]=self.uses.get((name,device),0)+1;self.used_at[name]=time.monotonic()
        try:yield
        finally:
            with self.control:
                self.uses[name,device]-=1;self.used_at[name]=time.monotonic()
                if not self.uses[name,device]:del self.uses[name,device]
                self.freed.notify_all()

    def _place(self, name, device):
        """Make room on `device` for `name`. Exclusive: everyone else leaves. LRU: the others stay
        while the card has room for what `name` declares (`vram_gib`), measured; otherwise they leave
        in this order until it has: one declaring nothing (nothing fits beside it), then visitors (not
        this device's idle residency), then residents, least recently used first within each. So a
        one-off text-to-speech between two utterances goes before the ASR a resident translator works with.
        One that declares nothing is placed as on an exclusive card."""
        need=self._vram_need(name,device.name);home=self._residency(device.name)
        others=sorted((o for o in self._occupants(device) if o!=name),
                      key=lambda o:(self._vram_need(o,device.name) is not None,o.split('.')[0] in home,self.used_at.get(o,0)))
        if device.placement=='exclusive' or need is None:
            for other in others:self._stop_tenant(other,device.name)
        else:
            while not self._room(device,need,others):
                if not others:
                    total,used=device.vram()
                    raise RuntimeError('%s: %.1f GiB free, %s needs %.1f GiB (+%.1f margin) and nothing of ours is left to evict'
                                       %(device.name,(total-used)/GIB,name,need/GIB,device.vram_margin/GIB))
                self._stop_tenant(others.pop(0),device.name)
        with self.control:self.used_at[name]=time.monotonic()

    def _residency(self, device):
        """The tenants the idle policy keeps on `device` (a plugin tenant by its home)."""
        return {n for n in self.config.get('policy',{}).get('idle',{}).get('restore',[]) if self._restore_device(n)==device}

    def _vram_need(self, name, device):
        """Bytes `name` declares it needs on `device` (`vram_gib`, the launch's own if it has one), or None."""
        tenant,_,part=name.partition('.')
        spec=self.plugins[tenant].parts(self.tenants[tenant])[part] if part and tenant in self.plugins else self.tenants.get(name,{})
        value=spec.get('on',{}).get(device,{}).get('vram_gib',spec.get('vram_gib'))
        return None if value is None else value*GIB

    def _room(self, device, need, residents):
        """Whether `need` bytes fit beside `residents`: what the card reports used, or what they
        declare if more (a resident may not have grown to its peak yet), plus the margin. Memory of a
        process just stopped takes a moment to show as free, so a shortfall is looked at again."""
        needs=[self._vram_need(r,device.name) for r in residents]
        # One that declares nothing may grow to the whole card (the DiT at rest uses far less than
        # while it samples): nothing is placed beside it.
        if None in needs:return False
        declared=sum(needs)
        deadline=time.monotonic()+VRAM_SETTLE_S
        while True:
            total,used=device.vram()
            if total-max(used,declared)-device.vram_margin>=need:return True
            if time.monotonic()>=deadline:return False
            time.sleep(.25)

    def _pids(self):
        with self.control:
            servers=list(self.servers.values());workers=[*servers,*(worker for worker,_ in self.parts.values())];batches=list(self.batches.values())
        pids={str(w.process.pid) for w in workers if w}
        # A batch job, and a server such as vLLM whose engine core is a child process, hold the
        # device from below the pid we started.
        for process in batches+[s.process for s in servers]:pids|=descendants(process.pid)
        return pids

    def conflicts(self):
        owned=self._pids();found=[]
        for device in self.devices.values():
            found.extend(device.foreign(owned,bool(self._occupants(device))))
        return found

    def check_device(self, name):
        """Look for processes that are not ours on one device. Found ones block the device (its
        queue waits, nothing of ours is placed there) until a later check finds it clear; they
        are never killed."""
        device=self.devices[name]
        found=device.foreign(self._pids(),bool(self._occupants(device)))
        with self.control:
            if found:self.blocked[name]='not ours: '+', '.join(found)
            else:self.blocked.pop(name,None)
            self._refresh()
        return found

    def _lease(self):
        with self.control:
            if self.lease:return
            lease=open(self.mode_path,'a')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:lease.close();raise
            self.lease=lease
        # A busy device blocks only itself; the others go on serving.
        for name in self.devices:self.check_device(name)

    @contextlib.contextmanager
    def _flock(self, path):
        with open(path,'a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            yield

    def _phase(self):
        """Host-wide: model loads and host-RAM-heavy batch jobs, one at a time."""
        return self._flock(self.phase_path)

    def device_phase_path(self, name):
        path=Path(self.phase_path);return str(path.with_name(path.stem+'.'+name+path.suffix))

    def _device_phase(self, name):
        """One device's compute; the workers resident on it take the same lock themselves."""
        return self._flock(self.device_phase_path(name))

    def _log(self, name):
        logdir=self.state/'logs';logdir.mkdir(exist_ok=True)
        return logdir/(str(time.time_ns())+'-'+name+'.log')

    def status(self):
        with self.control:
            return {'devices':{name:{'display':d.display,'placement':d.placement,'queue':d.queue,'occupants':self._occupants(d),
                                     'lent':self.lent.get(name),'blocked':self.blocked.get(name)} for name,d in self.devices.items()},
                    'engines':{k:v.process.pid for k,v in self.servers.items()},
                    'parts':{k:getattr(getattr(w,'process',None),'pid',None) for k,(w,_) in self.parts.items()},
                    'engine_devices':dict(self.server_on),
                    'batches':{k:v.pid for k,v in self.batches.items()},
                    **{name:plugin.status() for name,plugin in self.plugins.items()},
                    'in_use':sorted({name for name,_ in self.uses}),
                    'policy':self.config.get('policy',{}),'park_error':self.park_error}

    def text_servers(self):
        """(tenant, device, url, kind) of each running llama-server or vLLM, for text_rates."""
        with self.control:
            found=[]
            for name,device in self.server_on.items():
                spec=self.tenants[name];kind=text_rates.kind_of({**spec,**spec.get('on',{}).get(device,{})}['command'])
                if kind and name in self.servers:found.append((name,device,self.servers[name].url,kind))
            return found

    def _refresh(self):
        with self.control:
            ready={}
            for plugin in self.plugins.values():ready.update(plugin.ready())
            self.ready={**ready,**self.status()} if self.lease else None

    # ---- tenants ------------------------------------------------------------------------
    def _server(self, name, lane=None, wait=True):
        """The server on device `lane` (default: where its work goes now, see lane_for). Running on
        the other device, it is stopped there first, like a plugin's part, once no call uses it there
        (_leave); with wait=False, None instead of waiting."""
        spec=self.tenants[name];device=self.devices[lane or self.lane_for(spec)]
        with self.device_locks[device.name]:
            self._lease()
            if name in self.servers and self.server_on.get(name)==device.name:return self.servers[name]
            self._usable(device)
            device.check()
            spec={**spec,**spec.get('on',{}).get(device.name,{})}   # this device's own launch, if declared
            env=device.env(spec['api']);env.update(spec.get('env',{}))
            if not self._leave(name,device.name,wait):return None
            try:
                self._place(name,device)
                with self._phase():
                    server=self.server_worker(spec['command'],self._log(name),self.lease.fileno(),env,spec.get('cwd'),spec['url'],spec.get('health','/health'),
                                              startup_s=spec.get('startup_s',SERVER_STARTUP_S))
                recorder=getattr(self,'recorder',None)
                server.backend=recorder.started(name,device.name,device.display,server.process.pid) if recorder else None
                with self.control:self.servers[name]=server;self.server_on[name]=device.name
            finally:self._started(name)
            self._refresh();return server

    def hold(self, part, device, worker):
        """A plugin tenant's part `<tenant>.<part>`, started on `device` (TenantKind): from now on
        evicted, stopped and accounted for like any other tenant."""
        with self.control:self.parts[part]=(worker,device)

    def part(self, name):
        """The worker of a plugin tenant's part, or None."""
        with self.control:return (self.parts.get(name) or (None,))[0]

    def _restore_device(self, name):
        return self.plugins[name].home() if name in self.plugins else self.tenants[name]['device']

    def park(self, device=None):
        """Idle policy: put back what the config says should be resident when nothing is queued,
        on one device (its queue is idle) or on all of them. A lent or blocked device is skipped.

        A plugin tenant puts back what it keeps resident (TenantKind.restore). A tenant in use on
        another device is left there (invariant 2); a later idle brings it back."""
        self.park_error=None;self._lease()
        for name in self.config.get('policy',{}).get('idle',{}).get('restore',[]):
            where=self._restore_device(name)
            if (device is not None and where!=device) or where in self.lent or where in self.blocked:continue
            if name in self.plugins:self.plugins[name].restore()
            else:self._server(name,where,wait=False)   # on its own device, never the fallback
        self._refresh()

    def stop_device(self, name):
        """Stop everything of ours on one device; the other devices keep what they hold."""
        with self.device_locks[name]:
            for tenant in self._occupants(self.devices[name]):self._stop_tenant(tenant,name)
            self._refresh()

    def stop(self):
        locks=[self.device_locks[name] for name in sorted(self.device_locks)]
        for lock in locks:lock.acquire()
        try:
            for name in list(self.servers)+list(self.batches)+list(self.parts):self._stop_tenant(name)
            with self.control:
                self.ready=None
                if self.lease:self.lease.close();self.lease=None
        finally:
            for lock in reversed(locks):lock.release()

    # ---- lending a device ---------------------------------------------------------------
    def lend(self, name, record):
        """Empty one device and hand it out: until reclaim() nothing of ours is placed on it.
        Returns the environment that addresses it, and only it."""
        device=self.devices[name]
        with self.device_locks[name]:
            self._lease()
            for tenant in self._occupants(device):self._stop_tenant(tenant,name)
            with self.control:self.lent[name]=dict(record)
            self._refresh()
        return self._lend_env(device)

    def restore_lend(self, name, record):
        """After a restart: a lend still within its deadline is lent again, without the lease and
        without touching the device (the borrower is on it; nothing of ours is)."""
        with self.control:self.lent[name]=dict(record)
        return self._lend_env(self.devices[name])

    def _lend_env(self, device):
        """Both APIs point at the lent device only: a Vulkan program (llama.cpp's Vulkan build)
        ignores CUDA_VISIBLE_DEVICES and spreads its layers over every card it sees, the other
        card and the iGPU included; a CUDA program on an AMD lend would land on the NVIDIA card.
        The driver is chosen by manifest name, not path: the borrower runs on the host, where the
        manifest is elsewhere than in this container (nvidia's: /usr/share vs /etc)."""
        env={k:v for k,v in device.env('vulkan',strict=False).items() if k in ('GGML_VK_VISIBLE_DEVICES','DRI_PRIME')}
        env['VK_LOADER_DRIVERS_SELECT']='*'+Path(device.icd_path(False)).name.split('.')[0]+'*'
        env['CUDA_VISIBLE_DEVICES']=device.uuid if device.vendor=='nvidia' else ''
        return env

    def reclaim(self, name):
        """Take a lent device back. Whatever of the borrower is still on it blocks it (see
        check_device); returns those processes, empty when the device is clear."""
        with self.device_locks[name]:
            with self.control:self.lent.pop(name,None)
            return self.check_device(name)

    def _stop_batch(self, name):
        """End a batch job's whole process group: the tenant's child starts children of its own."""
        with self.control:process=self.batches.pop(name,None);self.batch_on.pop(name,None)
        if process is None:return
        if process.poll() is None:
            for sig,wait in ((signal.SIGTERM,15),(signal.SIGKILL,15)):
                try:os.killpg(process.pid,sig)
                except ProcessLookupError:break
                try:process.wait(timeout=wait);break
                except subprocess.TimeoutExpired:continue
        kill_group(process.pid)
        if process.stdout:process.stdout.close()
        self._refresh()

    # ---- work ---------------------------------------------------------------------------
    def run_stage(self, kind, stage, request, out, lane=None, cancelled=lambda:False):
        """One stage of a job other than an engine call or a lend, on device `lane`: a batch job,
        or a plugin tenant's stage (TenantKind.run)."""
        if kind in self.kinds:return self.kinds[kind].run(kind,stage,request,out,lane)
        if kind in BATCH_JOBS:return self.batch(kind,request,out,cancelled,lane)
        raise ValueError('unknown job kind '+str(kind))

    def batch(self, kind, request, out, cancelled=lambda:False, lane=None):
        """One job on a batch tenant, run as a fresh process for exactly the job's length.

        Placed like any tenant (on an exclusive device everyone else leaves first, and nothing on
        another card) and run under the lease, its device's lock and the host-wide phase lock (a
        batch job may keep a whole video in host RAM); the idle policy puts back what it evicted.
        `kind` is the batch job kind (BATCH_JOBS), whose request field names the tenant. The job is
        stopped, process group and all, if it is cancelled or outlives the tenant's timeout. `lane`
        is the device whose queue the job waited in (default lane_for: its own, or its fallback).
        """
        name=request[BATCH_JOBS[kind]];spec=self.tenants.get(name)
        if not spec or spec.get('kind')!='batch':raise ValueError('no batch tenant named '+str(name))
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        (out/'request.json').write_text(json.dumps(request,indent=2))
        device=self.devices[lane or self.lane_for(spec)]
        with self.device_locks[device.name],self._using(name,device.name):
            self._lease();self._usable(device)
            self._place(name,device);device.check()
            env=device.env(spec['api']);env.update(spec.get('env',{}))
            log=self._log(name)
            with self._phase(),self._device_phase(device.name),open(log,'ab') as stream:
                process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('parent_guard.py')),spec['python'],spec['script'],
                                          str(out/'request.json'),str(out)],stdout=subprocess.PIPE,stderr=stream,
                                         pass_fds=(self.lease.fileno(),),env=env,cwd=str(self.root),start_new_session=True)
                with self.control:self.batches[name]=process;self.batch_on[name]=device.name
                self._refresh()
                deadline=time.monotonic()+spec.get('timeout_s',7200)
                try:
                    while True:
                        try:process.wait(timeout=1);break
                        except subprocess.TimeoutExpired:pass
                        if cancelled():raise RuntimeError('cancelled while running; the job was stopped')
                        if time.monotonic()>deadline:raise RuntimeError('%s exceeded %ss; the job was stopped'%(name,spec.get('timeout_s',7200)))
                    # Anything the tenant left running in its group is a leak — and may still hold
                    # the result pipe open, which would make reading it wait for that process.
                    kill_group(process.pid)
                    output=process.stdout.read()
                finally:
                    self._stop_batch(name)
        if process.returncode:raise RuntimeError('%s exited %d; see %s'%(name,process.returncode,log))
        lines=output.decode().strip().splitlines()
        if not lines:raise RuntimeError('%s produced no result; see %s'%(name,log))
        return json.loads(lines[-1])

    def engine(self, request, out, lane=None):
        """One call to a server tenant on `lane`, the device whose queue it waited in (default
        lane_for: its own, or its fallback)."""
        self.validate_engine(request)
        with self.engine_session(request['engine'],lane) as (server,device):
            return self.engine_call(request,out,server,device)

    @contextlib.contextmanager
    def engine_session(self, name, lane=None):
        """Server tenant `name` placed on `lane` (default lane_for) and that device held while the
        block runs, so that calls to it (engine_call) can run side by side, up to its `slots`."""
        device=lane or self.lane_for(self.tenants[name])
        with self.device_locks[device],self._using(name,device):
            server=self._server(name,device)
            with self._device_phase(device):yield server,device

    def engine_call(self, request, out, server, device):
        """One HTTP call to a server held by engine_session; its response is streamed to `out`."""
        name=request['engine'];path=request['path'];method=request['method']
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        body=base64.b64decode(request.get('body',''),validate=True)
        req=urllib.request.Request(server.url+path,data=body if method=='POST' else None,
            headers={'Content-Type':request.get('content_type','application/json')},method=method)
        try:response=urllib.request.urlopen(req,timeout=900)
        except urllib.error.HTTPError as e:response=e
        with response:
            meta={'status':response.status,'content_type':response.headers.get('Content-Type','application/octet-stream'),'device':device,
                  **({'backend':server.backend} if getattr(server,'backend',None) else {})}   # the process that answers: backend_record
            # A server that answers a submission with a job to poll (gpu-config `poll`): the call
            # lasts until that job's reply carries `result` or `error`, and that reply is the response.
            declared=self.tenants.get(name,{}).get('poll')
            if declared and path==declared['submit'] and response.status in (200,202):
                submission=json.load(response);poll=submission.get(declared['field'])
                if not isinstance(poll,str) or not poll.startswith(declared['prefix']):raise RuntimeError('invalid job URL %r'%poll)
                deadline=time.monotonic()+900
                while True:
                    with urllib.request.urlopen(server.url+poll,timeout=30) as r:result=json.load(r)
                    if result.get('result') or result.get('error'):break
                    if time.monotonic()>deadline:raise RuntimeError('%s job timeout'%name)
                    time.sleep(.2)
                (out/'response.json').write_text(json.dumps(meta))
                (out/'response.bin').write_text(json.dumps(result))
                return {**meta,'response':str(out/'response.bin'),'device':device}
            (out/'response.json').write_text(json.dumps(meta))
            with (out/'response.bin').open('wb') as stream:
                while chunk:=response.read1(65536):stream.write(chunk);stream.flush()
        return {**meta,'response':str(out/'response.bin'),'device':device}

    def backends(self, name):
        """How server tenant `name` runs on each of its devices, for /v1/models: the card, the API and
        what text_rates.describe tells from the launch there."""
        spec=self.tenants[name];found=[]
        for device in (spec['device'],spec.get('fallback')):
            if not device:continue
            launch={**spec,**spec.get('on',{}).get(device,{})}
            found.append({'device':device,'card':self.devices[device].display,'api':launch.get('api'),**text_rates.describe(launch.get('command',[]))})
        return found

    def model_tenants(self, model):
        """The server tenants that answer to `model` (their `models`), in config order."""
        return [name for name,spec in self.tenants.items() if model in spec.get('models',[])]
