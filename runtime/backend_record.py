"""What each server process the runtime starts is, checked by the runtime itself, so that every
API reply can be tied to the process that produced it (X-GPU-Backend; GET /v1/backends/<id>).

A server may be stopped and started again between two calls (another tenant needed the card), so
a reply names one process: `<boot>-<pid>-<start ticks>`, unique on this machine. Only the runtime
can look inside it: yama ptrace_scope 1 lets an ancestor read /proc/<pid>/maps and environ, nobody
else. Once a started server answers its health check, the runtime records (state/backends/<id>.json):
its executable and command line, the environment that sets numerics, every GPU-side library it has
mapped (and its children, e.g. vLLM's engine core), the weights it was given, all with sha256, and
the Mesa packages of the image. `status` is `checking` until the hashes are done, then `checked`.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import mesa_check
import text_rates

ENV_KEYS=re.compile(r'^(GGML_|VK_|CUDA_|DRI_|LD_LIBRARY_PATH$|HIP_|ROCR_|MESA_|RADV_|AMD_|VLLM_|HF_HUB_OFFLINE$|NVIDIA_|CUBLAS|PYTORCH_|TORCH_)')
LIBRARIES=re.compile(r'(vulkan|radeon|gallium|ggml|llama|mtmd|cuda|cublas|cudnn|nccl|nvidia|libtorch|amdgpu|drm)[^/]*\.so')
WEIGHT_FILES=('.gguf','.safetensors','.json','.model','.txt','.jinja')


def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()[:8]


def start_ticks(pid):
    """The process's start time in clock ticks since boot (stat field 22): with the pid, unique."""
    return int(Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19])


def ident(pid):return '%s-%d-%d'%(boot_id(),pid,start_ticks(pid))


def alive(name):
    """Whether the process a record id names is still running (same pid, same start)."""
    try:boot,pid,ticks=name.rsplit('-',2);return boot==boot_id() and start_ticks(int(pid))==int(ticks)
    except (OSError,ValueError):return False


class Hashes:
    """sha256 of files, kept by (path, size, mtime) in `cache` so a 16 GB GGUF is read once."""
    def __init__(self, cache):
        self.cache=Path(cache);self.lock=threading.Lock()
        try:self.known=json.loads(self.cache.read_text())
        except (OSError,ValueError):self.known={}

    def __call__(self, path):
        stat=os.stat(path);key=[stat.st_size,stat.st_mtime_ns]
        with self.lock:
            if self.known.get(path,{}).get('key')==key:return self.known[path]['sha256']
        digest=hashlib.sha256()
        with open(path,'rb') as f:
            while block:=f.read(1<<24):digest.update(block)
        with self.lock:
            self.known[path]={'key':key,'sha256':digest.hexdigest()}
            partial=self.cache.with_suffix('.tmp');partial.write_text(json.dumps(self.known));partial.replace(self.cache)
        return digest.hexdigest()


def weights(argv):
    """The weight files a text server was given: llama-server's -m / -md / --mmproj, the files of
    the directory vLLM serves."""
    kind=text_rates.kind_of(argv);found=[]
    if kind=='llama.cpp':
        found=[argv[i+1] for i,word in enumerate(argv[:-1]) if word in ('-m','--model','-md','--model-draft','--mmproj')]
    elif kind=='vllm' and 'serve' in argv[:-1]:
        folder=Path(argv[argv.index('serve')+1])
        found=[str(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in WEIGHT_FILES] if folder.is_dir() else []
    return found


def inspect(pid, hashes, family):
    """The facts of a running process and `family`, it and its descendants (see the module docstring)."""
    proc=Path('/proc/%d'%pid)
    argv=[a for a in (proc/'cmdline').read_bytes().decode().split('\0') if a]
    env=dict(item.split('=',1) for item in (proc/'environ').read_bytes().decode().split('\0') if '=' in item)
    exe=os.readlink(proc/'exe');libraries=set()
    for each in sorted(int(p) for p in family):
        try:maps=Path('/proc/%d/maps'%each).read_text()
        except OSError:continue
        for line in maps.splitlines():
            path=line.split(maxsplit=5)[5] if len(line.split(maxsplit=5))==6 else ''
            if path.startswith('/') and LIBRARIES.search(os.path.basename(path)) and not path.endswith(' (deleted)'):libraries.add(path)
    return {'argv':argv,'cwd':os.readlink(proc/'cwd'),'uid':os.stat(proc).st_uid,
            'exe':{'path':exe,'sha256':hashes(exe)},
            'env':{k:v for k,v in sorted(env.items()) if ENV_KEYS.match(k)},
            'libraries':{path:hashes(path) for path in sorted(libraries)},
            'weights':{path:hashes(path) for path in weights(argv)},
            'packages':mesa_check.installed(Path(mesa_check.OWN_STATUS).read_text()) if Path(mesa_check.OWN_STATUS).exists() else {}}


class Recorder:
    def __init__(self, folder, family):
        """`family(pid)`: the pids of a process and of every process below it."""
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.family=family
        self.hashes=Hashes(self.folder/'sha256-cache.json')

    def _write(self, name, record):
        partial=self.folder/(name+'.tmp');partial.write_text(json.dumps(record,indent=1));partial.replace(self.folder/(name+'.json'))

    def started(self, tenant, device, card, pid, wait=False):
        """Record a server the runtime just started; returns its id, or None when `pid` is not a
        process of ours (only our own children are looked inside)."""
        try:
            name=ident(pid)
            if int(Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[1])!=os.getpid():return None
        except (OSError,ValueError):return None
        record={'id':name,'tenant':tenant,'device':device,'card':card,'pid':pid,'started':time.time(),'status':'checking'}
        self._write(name,record)
        def check():
            try:record.update(inspect(pid,self.hashes,self.family(pid)),status='checked',checked=time.time())
            except Exception as exc:record.update(status='failed',error='%s: %s'%(type(exc).__name__,exc))
            if not alive(name) and record['status']=='checked':record.update(status='failed',error='the process ended while it was checked')
            self._write(name,record)
        thread=threading.Thread(target=check,daemon=True);thread.start()
        if wait:thread.join()
        return name

    def get(self, name):
        if not re.fullmatch(r'[0-9a-f]{8}-\d+-\d+',str(name)):raise ValueError('unknown backend %r'%name)
        try:record=json.loads((self.folder/(name+'.json')).read_text())
        except OSError:raise ValueError('unknown backend %r'%name)
        return {**record,'alive':alive(name)}
