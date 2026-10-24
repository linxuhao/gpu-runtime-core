#!/usr/bin/env python3
"""Borrow one GPU from the GPU runtime for research, through its job queue.

  gpu_borrow.py run --device gpu0 [--hours 2] [--label WHAT] -- python train.py ...
      Queued mode: waits its turn among the agent jobs on that device, runs the command with
      only that device visible, and gives the device back when the command ends. At the
      deadline the command gets SIGTERM, and SIGKILL a minute later. Exits with its exit code.
  gpu_borrow.py run --device gpu0 --vram-gib 12 [--hours 2] [--label WHAT] -- python train.py ...
      Shared mode: as queued, but the device is shared with other shared borrowers while what
      they declare fits on it. Declare the most the command uses: nothing enforces it, and a
      command that takes more can run the others out of memory.
  gpu_borrow.py reserve --device gpu0 [--hours 72] [--label WHAT]
      Exclusive mode: waits its turn, then the device is research's until `release` or the
      deadline; meanwhile new jobs that need it are refused. Prints the environment to use.
  gpu_borrow.py release LEND_ID
  gpu_borrow.py status

It talks to the runtime's local socket (state/runtime.sock next to the directory it runs from;
--socket or GPU_RUNTIME_SOCKET to override). Nothing here touches a GPU: the runtime empties the device and says how to address
it. Research processes still share the host's RAM with the runtime.
"""
import argparse
import getpass
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

HEARTBEAT_S=20
KILL_GRACE_S=60
def default_socket(script):
    """GPU_RUNTIME_SOCKET, else state/runtime.sock in the checkout `script` was run from (the path it
    was called by, not where a symlink to it points)."""
    return os.environ.get('GPU_RUNTIME_SOCKET') or str(Path(script).absolute().parents[1]/'state'/'runtime.sock')


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost',timeout=60);self.path=path
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.connect(self.path)


def call(path, action, **arguments):
    connection=UnixConnection(str(path))
    try:
        connection.request('POST','/rpc',json.dumps({'action':action,'arguments':arguments}),{'Content-Type':'application/json'})
        reply=json.loads(connection.getresponse().read())
    finally:connection.close()
    if 'error' in reply:raise SystemExit('gpu_borrow: '+reply['error'])
    return reply['result']


def when(epoch):return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(epoch))


def heartbeat(path, ident, stop):
    while not stop.wait(HEARTBEAT_S):
        try:call(path,'lend_heartbeat',job_id=ident)
        except (OSError,SystemExit):pass     # the runtime restarting; the next beat may get through


def wait_for_grant(path, ident):
    """Until the device is ours; None if the request ended first. Ctrl-C withdraws it."""
    said=None
    try:
        while True:
            state=call(path,'lend_get',job_id=ident)
            if state['granted']:return state['granted']
            job=state['job']
            if job['status'] in ('completed','failed','cancelled','interrupted'):
                print('gpu_borrow: %s ended %s before it was granted: %s'%(ident,job['status'],job.get('error') or ''),file=sys.stderr)
                return None
            if state['ahead'] is not None:
                line='waiting for %s: %s ahead'%(job['lane'],state['ahead'])
                if line!=said:print('gpu_borrow: '+line,file=sys.stderr);said=line
            time.sleep(1)
    except KeyboardInterrupt:
        call(path,'lend_return',job_id=ident);raise SystemExit(130)


def submit(path, args, mode):
    extra={'label':args.label} if args.label else {}
    if getattr(args,'vram_gib',None) is not None:extra['vram_gib']=args.vram_gib
    return call(path,'lend_submit',device=args.device,mode=mode,hours=args.hours,owner=args.owner,
                request_key='borrow-'+uuid.uuid4().hex,**extra)


def run(path, args):
    if not args.command:raise SystemExit('gpu_borrow: give the command after --')
    job=submit(path,args,'queued' if args.vram_gib is None else 'shared');ident=job['id']
    stop=threading.Event();threading.Thread(target=heartbeat,args=(path,ident,stop),daemon=True).start()
    try:
        grant=wait_for_grant(path,ident)
        if grant is None:return 1
        print('gpu_borrow: %s is yours until %s (%s)'%(grant['device'],when(grant['deadline']),ident),file=sys.stderr)
        guard=Path(__file__).with_name('parent_guard.py')   # the command dies with this process
        child=subprocess.Popen([sys.executable,str(guard),*args.command],env={**os.environ,**grant['env']},start_new_session=True)
        def forward(signum,frame):
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,forward)
        try:child.wait(timeout=max(0,grant['deadline']-time.time()))
        except subprocess.TimeoutExpired:
            print('gpu_borrow: deadline reached; stopping the command',file=sys.stderr)
            forward(signal.SIGTERM,None)
            try:child.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:forward(signal.SIGKILL,None);child.wait()
        return child.returncode if child.returncode>=0 else 128-child.returncode
    finally:
        stop.set()
        try:call(path,'lend_return',job_id=ident)
        except (OSError,SystemExit) as exc:print('gpu_borrow: could not return %s: %s'%(ident,exc),file=sys.stderr)


def reserve(path, args):
    job=submit(path,args,'exclusive');ident=job['id']
    stop=threading.Event();threading.Thread(target=heartbeat,args=(path,ident,stop),daemon=True).start()
    try:grant=wait_for_grant(path,ident)
    finally:stop.set()
    if grant is None:return 1
    print('# %s is reserved until %s; give it back with: gpu_borrow.py release %s'%(grant['device'],when(grant['deadline']),ident))
    for key,value in grant['env'].items():print('export %s=%s'%(key,value))
    return 0


def status(path):
    state=call(path,'resource_status')
    for lane,info in state.get('lanes',{}).items():
        lent=info['lent'];line='%-6s queued %-3d active %s'%(lane,info['queued'],info['active'] or '-')
        if lent:line+='  lent (%s) to %s until %s [%s]'%(lent['mode'],lent['owner']+(' (%s)'%lent['label'] if lent.get('label') else ''),when(lent['deadline']),lent['job_id'])
        for lend in info.get('shared',[]):
            line+='\n%-6s shared %s GiB with %s until %s [%s]'%('',lend['vram_gib'],lend['owner']+(' (%s)'%lend['label'] if lend.get('label') else ''),when(lend['deadline']),lend['job_id'])
        if info['blocked']:line+='  BLOCKED: '+info['blocked']
        print(line)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--socket',default=default_socket(__file__))
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('run','reserve'):
        p=sub.add_parser(name);p.add_argument('--device',required=True)
        p.add_argument('--hours',type=float,default=None,help='default 2 for run, 72 for reserve')
        p.add_argument('--owner',default=getpass.getuser()+'@'+socket.gethostname())
        p.add_argument('--label',help='what the device is for, shown beside the owner in the queue and gpu_top')
        if name=='run':
            p.add_argument('--vram-gib',type=float,default=None,help='share the device: the most VRAM the command uses, in GiB')
            p.add_argument('command',nargs=argparse.REMAINDER)
    sub.add_parser('release').add_argument('lend_id')
    sub.add_parser('status')
    args=parser.parse_args(argv)
    if args.action=='run':
        if args.command[:1]==['--']:args.command=args.command[1:]
        if args.hours is None:args.hours=2
        return run(args.socket,args)
    if args.action=='reserve':
        if args.hours is None:args.hours=72
        return reserve(args.socket,args)
    if args.action=='release':
        job=call(args.socket,'lend_return',job_id=args.lend_id);print(job['status']);return 0
    return status(args.socket)


if __name__=='__main__':sys.exit(main())
