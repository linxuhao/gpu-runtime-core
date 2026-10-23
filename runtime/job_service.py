"""The job service: a durable queue per device in front of the scheduler (gpu_runtime), over a
local Unix socket.

It knows jobs, tenants and devices. A job is a list of stages, each on the device it needs, and it
waits in one device's queue at a time, keeping its place by creation time: no job holds one device
while it waits for another. Its status is `running` from its first stage until it ends. The
scheduler itself runs engine calls (one HTTP call to a server tenant, `engine_submit`, by tenant
name or by model), batch jobs (one run of a batch tenant, `batch_submit`) and lends; a plugin tenant
(gpu_runtime.TenantKind) adds job kinds of its own, and an application subclasses Service for the
actions that submit them.

A device can also be lent (research, tools/gpu_borrow.py): a `lend` job queues like any other and,
once granted, keeps the device until it is returned, its deadline passes or its borrower stops
answering. A granted lend is recorded with its job, so a restart within its deadline (a crash, or a
stop: stopping leaves a granted lend as it is) lends the device again instead of dropping the
reservation. While a tenant's device is lent or blocked, its new work is queued on its declared
`fallback` device; an exclusive lend also moves the work already waiting there, and once the device
is usable again the work waiting on the fallback moves back. A queue that serves several stages of
one plugin's jobs prefers the stage whose part is loaded, within BATCH_PASS_LIMIT.

What a job's request carries under `origin` is provenance its submitter wrote for itself; nothing
here reads it.
"""
import argparse
import contextlib
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import shutil
import signal
import socketserver
import sqlite3
import threading
import time
import uuid

from gpu_runtime import BATCH_JOBS, CORE_JOBS, Scheduler, tenant_name
import mesa_check
import text_rates

TERMINAL={'completed','failed','cancelled','interrupted'}


def single_stage():
    """The job kinds the scheduler runs itself in one stage, on one tenant: engine calls, batch jobs."""
    return ('engine',*sorted(BATCH_JOBS))


def tenant_column():
    """SQL naming the tenant a waiting engine call or batch job is for (its request's field)."""
    return 'COALESCE(%s) AS tenant'%','.join("json_extract(request,'$.%s')"%key for key in ('engine',*sorted(set(BATCH_JOBS.values()))))

# Lending a device to research. Queued: one command among the agent jobs, briefly. Exclusive: the
# device is research's for days, and new jobs that need it are refused meanwhile.
LEND_HOURS={'queued':(2,12),'exclusive':(72,14*24)}     # mode -> (default, most)
LEND_HEARTBEAT_S=180       # a borrower silent this long has gone: its request or its device is taken back
BLOCKED_RECHECK_S=10       # how often a device blocked by a process not ours is looked at again
STATUS_WAITING_MAX=50     # resource_status lists at most this many waiting jobs per device (`queued` counts all)
IDLE_COALESCE_S=5          # a device's queue idle this long gets its idle residency back
SPILL_AFTER_S=3           # a device busy this long with another job lets its idle fallback take a call it holds loaded
MODEL_CALL_KEEP_S=24*3600  # a call by model (/v1/*), record and response, is deleted this long after it ended
# On a device that serves two stages of a plugin tenant's jobs (one part's fallback that also holds
# the other part), switching costs a model load each way (seconds to minutes), so the next job is
# preferably one of the stage already loaded; the oldest waiting job is passed over at most this
# many times before it is served regardless.
BATCH_PASS_LIMIT=3

def digest(path):
    """sha256 of a file."""
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def uid(prefix):return prefix+'-'+uuid.uuid4().hex[:20]
class ServiceStopping(Exception):
    """The service stops while a lend is granted: the lend is left as it is (job running, `lent`
    recorded, device not reclaimed), so the next start restores it."""
def packed(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def whose(lend):
    """A lend's borrower and, when they gave one, what it is for."""
    return lend['owner']+(' (%s)'%lend['label'] if lend.get('label') else '')



class Service:
    PREFIXES={'engine':'call','lend':'lend','batch':'batch'}   # job id prefix by kind (an application adds its own)
    DB_NAME='jobs.sqlite3'   # in the state directory

    def __init__(self, root, state, runtime=None, start_worker=True):
        self.root,self.state=Path(root).resolve(),Path(state).resolve();self.state.mkdir(parents=True,exist_ok=True)
        self.instance=open(self.state/'service.lock','a')
        try:fcntl.flock(self.instance,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.instance.close();raise
        self.cv=threading.Condition(threading.RLock());self.closing=False;self.draining=False;self.active=None
        self.running={}            # device -> the job (or 'residency') its queue is working on
        self.lends={};self.heartbeats={};self.resumed={};self.passed={}
        self.calls={}              # device -> {job id: thread} of the engine calls running side by side there
        self.relaned=None          # the lend/block state _relane last moved jobs for
        self.started={}            # device -> when its running job started (see _steal)
        self.lingers={}            # device -> (server tenant, monotonic time) it keeps the device until
        # Devices drained for maintenance: what is queued there finishes, nothing new is taken, and
        # once empty nothing of ours is left on them (no idle restore). Kept across restarts.
        self.drained=set(json.loads((self.state/'drained.json').read_text())) if (self.state/'drained.json').exists() else set()
        self.emptied=set()         # drained devices already emptied
        self.pruned=0              # when calls by model were last pruned
        self.db=sqlite3.connect(self.state/self.DB_NAME,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        # `project` and `shot` are provenance columns filled from the request's `origin`; the
        # service never joins on them. A database from before the split still carries its
        # projects / shots / driver_notes tables — they are left untouched, not read.
        self.db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, project TEXT NOT NULL, shot TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
          request TEXT NOT NULL, result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL, request_key TEXT UNIQUE, request_hash TEXT NOT NULL);
        ''')
        # Where a job waits: `lane` is the device whose queue holds it, `step` the stage it waits for
        # or runs, `waiting` 1 while it is in a queue (queued, or running between two stages).
        columns={row['name'] for row in self.db.execute('PRAGMA table_info(jobs)')}
        with self.db:
            # `lent`: a granted lend's record (owner, mode, granted, deadline), kept so a restart can restore it.
            for column,decl in (('lane','TEXT'),('step','TEXT'),('waiting','INTEGER NOT NULL DEFAULT 0'),('lent','TEXT')):
                if column not in columns:self.db.execute('ALTER TABLE jobs ADD COLUMN %s %s'%(column,decl))
            # `urgent`: an engine call by tenant name (/engines/*) goes before everything else on its
            # device; a call by model (/v1/*) and every other job wait in creation order.
            if 'urgent' not in columns:
                self.db.execute('ALTER TABLE jobs ADD COLUMN urgent INTEGER NOT NULL DEFAULT 0')
                self.db.execute("UPDATE jobs SET urgent=1 WHERE kind='engine'")
            # A granted lend whose deadline is still ahead survives the restart; everything else that
            # was running is interrupted, a lend past its deadline included (the device is ours again).
            keep=[row['id'] for row in self.db.execute("SELECT id,lent FROM jobs WHERE kind='lend' AND status='running' AND lent IS NOT NULL")
                  if json.loads(row['lent'])['deadline']>time.time()]
            self.db.execute("UPDATE jobs SET status='interrupted', error='Service restarted before completion; submit a new take explicitly',waiting=0,updated=? WHERE status IN ('running','cancelling') AND id NOT IN (%s)"
                            %','.join('?'*len(keep)),(time.time(),*keep))
        # The scheduler is entirely configured by the state directory's gpu-config.json.
        self.runtime=runtime if runtime is not None else Scheduler(self.root,self.state)
        self.lanes=list(getattr(self.runtime,'lanes',None) or ['gpu'])
        self.drained&=set(self.lanes)
        if hasattr(self.runtime,'drained'):self.runtime.drained=set(self.drained)
        for ident in keep:
            # Lent again as granted; the lane's queue resumes holding it (deadline, return, heartbeat).
            job=self._job(ident);record=job['lent'];lane=record['device']
            if lane not in self.lanes or lane not in getattr(self.runtime,'devices',{lane:None}):
                # gpu-config.json no longer has that device: nothing to restore it on.
                with self.db:self.db.execute("UPDATE jobs SET status='interrupted',error=?,waiting=0,lent=NULL,updated=? WHERE id=?",
                                             ('Service restarted; device %s is no longer configured'%lane,time.time(),ident))
                continue
            env=self.runtime.restore_lend(lane,record)
            self.lends[lane]=dict(record,env=env);self.heartbeats[ident]=time.time();self.resumed[lane]=ident
        with self.cv:self._relane()
        # Queued before there were queues per device: placed at their first stage now.
        for row in self.db.execute("SELECT id FROM jobs WHERE status='queued' AND (lane IS NULL OR waiting=0)").fetchall():
            job=self._job(row['id'])
            try:stage,lane=self._stages(job['kind'],job['request'])[0]
            except ValueError as exc:
                with self.db:self.db.execute("UPDATE jobs SET status='failed',error=?,updated=? WHERE id=?",(str(exc),time.time(),job['id']))
                continue
            with self.db:self.db.execute('UPDATE jobs SET lane=?,step=?,waiting=1 WHERE id=?',(lane,stage,job['id']))
        self.needs_park=dict.fromkeys(self.lanes,hasattr(self.runtime,'park'))
        self.threads={lane:threading.Thread(target=self._work,args=(lane,),daemon=True) for lane in self.lanes}
        self.rates=text_rates.TextRates()
        self.mesa=mesa_check.MesaCheck();self.mesa.warnings()   # logged at start, then on every change
        if start_worker:
            for thread in self.threads.values():thread.start()
            threading.Thread(target=self._sample_text,daemon=True).start()

    def _sample_text(self):
        """Read the text servers' counters for resource_status (see text_rates)."""
        while not self.closing:
            servers=getattr(self.runtime,'text_servers',list)()
            for tenant,device,url,kind in servers:
                try:self.rates.record(tenant,device,kind,text_rates.read(url,kind),time.monotonic())
                except (OSError,ValueError,KeyError):pass   # starting, stopping or busy: the next reading
            self.rates.keep([server[0] for server in servers])
            time.sleep(text_rates.SAMPLE_S)

    def close(self):
        with self.cv:self.closing=True;self.cv.notify_all()
        self.runtime.stop()
        for thread in self.threads.values():
            if thread.is_alive():thread.join(timeout=20)
        if any(thread.is_alive() for thread in self.threads.values()):raise RuntimeError('worker did not settle')
        self.db.close();self.instance.close()

    def _stages(self, kind, request):
        stages=getattr(self.runtime,'stages',None)
        return stages(kind,request) if stages else [(kind,self.lanes[0])]

    def _job(self, ident):
        row=self.db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
        if row is None:raise ValueError('unknown jobs id')
        result=dict(row)
        for k in ('request','result','lent'):
            if result.get(k) is not None:result[k]=json.loads(result[k])
        return result

    def call(self,action,args):
        with self.cv:
            if action=='models':
                # Per name: how many calls may run at once (the `slots` of every tenant answering to
                # it) and on which devices, so that a caller knows whether to send them in parallel,
                # and how each device runs it (card, API, framework, build, weights).
                tenants=getattr(self.runtime,'tenants',{})
                names=sorted({model for spec in tenants.values() for model in spec.get('models',[])})
                backends=getattr(self.runtime,'backends',lambda name:[])
                return [{'id':model,'max_parallel':sum(tenants[t].get('slots',1) for t in self._serving(model)),
                         'devices':sorted({d for t in self._serving(model) for d in (tenants[t].get('device'),tenants[t].get('fallback')) if d}),
                         'backends':[b for t in self._serving(model) for b in backends(t)]}
                        for model in names]
            if action=='backend':
                recorder=getattr(self.runtime,'recorder',None)
                if recorder is None:raise ValueError('this runtime records no backends')
                return recorder.get(args.get('id'))
            if action=='engine_submit':
                if self.draining:raise ValueError('GPU resources released; acquire before submitting')
                req={k:args[k] for k in ('path','method','body','content_type')}
                name=args['model'] if 'model' in args else args['engine']
                if args.get('devices') is not None:
                    devices=args['devices']
                    if not isinstance(devices,list) or not devices or not all(isinstance(d,str) for d in devices) or len(set(devices))<len(devices):
                        raise ValueError('devices must be a list of distinct device names, e.g. ["gpu1"]')
                    unknown=[d for d in devices if d not in self.lanes]
                    if unknown:raise ValueError('unknown device %s; this host has %s'%(', '.join(unknown),', '.join(self.lanes)))
                    req['devices']=devices
                if 'model' in args or name not in getattr(self.runtime,'tenants',{}) and self._serving(name):
                    # By a name a service answers to (any /v1 model; /engines/<name> when no tenant
                    # is called that): one of the tenants declaring it. A call already taken under
                    # this request_key keeps the tenant it went to.
                    if 'model' in args:self._prune_model_calls();req['model']=name
                    old=self.db.execute('SELECT request FROM jobs WHERE request_key=?',(args['request_key'],)).fetchone()
                    req['engine']=json.loads(old['request']).get('engine') if old else self._route(name,req.get('devices'))
                    req['route']=name
                else:req['engine']=name
                if len(req['body'])>28*1024*1024:raise ValueError('engine request exceeds limit')
                validate=getattr(self.runtime,'validate_engine',None)
                if validate:validate(req)   # the routes the tenant declares
                poll=getattr(self.runtime,'tenants',{}).get(req['engine'],{}).get('poll')
                if poll and req['path']==poll['submit']:
                    # Answered at once with this job to poll (facade); the call itself waits for the result.
                    req['poll']={k:poll[k] for k in ('field','prefix')}
                return self._enqueue('',None,'engine',req,args['request_key'])
            if action=='batch_submit':
                # A run of a batch tenant (gpu-config kind batch), named by the request's `tenant`.
                if self.draining:raise ValueError('GPU resources released; acquire before submitting')
                if not isinstance(args.get('request'),dict) or not isinstance(args['request'].get('tenant'),str):
                    raise ValueError('request must be an object naming its batch tenant')
                if getattr(self.runtime,'tenants',{}).get(args['request']['tenant'],{}).get('kind')!='batch':
                    raise ValueError('no batch tenant named %r'%args['request']['tenant'])
                return self._enqueue('',None,'batch',dict(args['request']),args['request_key'])
            if action=='jobs_get':
                ids=args['job_ids']
                if not isinstance(ids,list) or len(ids)>500:raise ValueError('job_ids must be a list of at most 500 ids')
                return [self._job(ident) for ident in ids]
            if action=='job_get':
                job=self._job(args['job_id'])
                if args.get('verify') and job['status']=='completed' and (job['result'] or {}).get('sha256') and digest(job['result']['output'])!=job['result']['sha256']:
                    raise ValueError('take artifact changed')
                return self._brief(job) if args.get('brief') else job
            if action=='job_wait':
                end=time.monotonic()+min(max(float(args.get('timeout',60)),0),900)
                while self._job(args['job_id'])['status'] not in TERMINAL:
                    left=end-time.monotonic()
                    if left<=0:break
                    self.cv.wait(left)
                job=self._job(args['job_id'])
                return self._brief(job) if args.get('brief') else job
            if action=='job_cancel':
                job=self._job(args['job_id'])
                if job['status']=='queued' or job['status'] not in TERMINAL and job['waiting']:
                    # Not running anywhere (queued, or between two stages): it ends now.
                    with self.db:self.db.execute("UPDATE jobs SET status='cancelled',waiting=0,updated=? WHERE id=?",(time.time(),job['id']))
                    self.cv.notify_all()
                elif job['status'] not in TERMINAL:
                    with self.db:self.db.execute("UPDATE jobs SET status='cancelling',updated=? WHERE id=?",(time.time(),job['id']))
                    self.cv.notify_all()
                return self._job(job['id'])
            if action=='resource_status':
                blocked=getattr(self.runtime,'blocked',{})
                lanes={lane:{'active':self.running.get(lane),'step':self._step_of(self.running.get(lane)),
                             'queued':self.db.execute('SELECT count(*) FROM jobs WHERE waiting=1 AND lane=?',(lane,)).fetchone()[0],
                             'lent':self._lend_view(self.lends[lane]) if lane in self.lends else None,'blocked':blocked.get(lane),
                             'drained':lane in self.drained,
                             'device':self._device_view(lane),'running':(self._queue_view('WHERE id=?',(self.running.get(lane),)) or [None])[0],
                             'also_running':self._also_running(lane),'text':self.rates.lane(lane),
                             'waiting':self._queue_view("WHERE waiting=1 AND lane=? ORDER BY "+self._order(lane)+" LIMIT ?",
                                                        (lane,STATUS_WAITING_MAX))}
                       for lane in self.lanes}
                return {'mode':'loaded' if self.runtime.ready else 'unloaded','draining':self.draining,'active_job':self.active,
                    'workers':self.runtime.ready,'queued':self.db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0],
                    'lanes':lanes,'warnings':self.mesa.warnings(),
                    'cancellation':'queued: immediate; running: finish current computation, discard candidate, then release workers'}
            if action=='resource_acquire':
                # Lazy loading: submit first shot to load; no GPU allocation on the request thread.
                self.draining=False;self.needs_park=dict.fromkeys(self.lanes,hasattr(self.runtime,'park'));self.cv.notify_all()
                return {'mode':'requested','load':'on the next job or idle residency restore'}
            if action=='resource_release':
                if self.lends:raise ValueError('return the lent devices first: '+', '.join('%s to %s'%(lane,whose(lend)) for lane,lend in self.lends.items()))
                if self.db.execute("SELECT count(*) FROM jobs WHERE waiting=1").fetchone()[0]:raise ValueError('cancel or finish queued jobs before releasing')
                self.draining=True
                if not self.running:self.runtime.stop()
                return {'draining':True,'active_job':self.active}
            if action=='lend_submit':
                return self._lend_submit(args)
            if action=='device_drain':
                lane=args.get('device');drain=args.get('drain',True)
                if lane not in self.lanes:raise ValueError('unknown device %r; this host has %s'%(lane,', '.join(self.lanes)))
                if not isinstance(drain,bool):raise ValueError('drain must be a JSON boolean')
                if drain:self.drained.add(lane)
                else:
                    self.drained.discard(lane);self.emptied.discard(lane);self.needs_park[lane]=hasattr(self.runtime,'park')
                partial=self.state/'drained.json.tmp';partial.write_text(json.dumps(sorted(self.drained)));partial.replace(self.state/'drained.json')
                if hasattr(self.runtime,'drained'):self.runtime.drained=set(self.drained)
                self.cv.notify_all()
                return {'device':lane,'drained':drain,'queued':self.db.execute('SELECT count(*) FROM jobs WHERE waiting=1 AND lane=?',(lane,)).fetchone()[0],
                        'running':self.running.get(lane)}
            if action=='lend_get':
                job=self._job(args['job_id'])
                if job['kind']!='lend':raise ValueError('not a lend job')
                lend=self.lends.get(job['lane'])
                granted=lend if lend and lend['job_id']==job['id'] else None
                ahead=None
                if job['waiting']:
                    ahead=self.db.execute("SELECT count(*) FROM jobs WHERE waiting=1 AND lane=? AND (-urgent,created,rowid)<"
                                          "(SELECT -urgent,created,rowid FROM jobs WHERE id=?)",(job['lane'],job['id'])).fetchone()[0]
                    ahead+=1 if self.running.get(job['lane']) else 0
                return {'job':job,'granted':self._lend_view(granted,env=True) if granted else None,'ahead':ahead}
            if action=='lend_heartbeat':
                job=self._job(args['job_id'])
                if job['kind']!='lend' or job['status'] in TERMINAL:raise ValueError('no live lend %s (%s)'%(job['id'],job['status']))
                self.heartbeats[job['id']]=time.time();return {'job_id':job['id'],'status':job['status']}
            if action=='lend_return':
                job=self._job(args['job_id'])
                if job['kind']!='lend':raise ValueError('not a lend job')
                lend=self.lends.get(job['lane'])
                if lend and lend['job_id']==job['id']:
                    # Recorded first: a restart before the device is reclaimed must not restore it.
                    with self.db:self.db.execute('UPDATE jobs SET lent=NULL WHERE id=?',(job['id'],))
                    lend['returned']=True;self.cv.notify_all()
                elif job['status'] not in TERMINAL:
                    with self.db:self.db.execute("UPDATE jobs SET status='cancelled',waiting=0,updated=? WHERE id=?",(time.time(),job['id']))
                    self.cv.notify_all()
                return self._job(job['id'])
            raise ValueError('unknown action')

    def _enqueue(self,project,shot,kind,req,key):
        if not isinstance(key,str) or not 1<=len(key)<=200:raise ValueError('request_key required, <=200 characters')
        fingerprint=self._fingerprint(project,shot,kind,req)
        old=self.db.execute('SELECT id,request_hash FROM jobs WHERE request_key=?',(key,)).fetchone()
        if old:
            if old['request_hash']!=fingerprint:raise ValueError('request_key reused with different inputs')
            return self._job(old['id'])
        stages=self._stages(kind,req)
        for _,lane in stages:
            if lane in self.drained:
                raise ValueError('%s is drained for maintenance: what was queued there finishes, nothing new is taken (tools/gpu_drain.py --resume %s)'%(lane,lane))
            lend=self.lends.get(lane)
            if lend and lend['mode']=='exclusive':
                raise ValueError('%s is reserved for research by %s until %s; what needs it is refused until it is returned'
                                 %(lane,whose(lend),time.strftime('%Y-%m-%d %H:%M %Z',time.localtime(lend['deadline']))))
        ident=uid(self.PREFIXES.get(kind,'job'));now=time.time()
        urgent=int(kind=='engine' and 'model' not in req)
        with self.db:self.db.execute('INSERT INTO jobs(id,project,shot,kind,status,request,result,error,created,updated,request_key,request_hash,lane,step,waiting,urgent)'
            ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)',(ident,project,shot,kind,'queued',packed(req),None,None,now,now,key,fingerprint,stages[0][1],stages[0][0],urgent))
        self.cv.notify_all();return self._job(ident)

    @staticmethod
    def _fingerprint(project, shot, kind, req):return hashlib.sha256(packed([project,shot,kind,req]).encode()).hexdigest()

    def _lend_submit(self, args):
        if self.draining:raise ValueError('GPU resources released; acquire before submitting')
        mode=args.get('mode','queued')
        if mode not in LEND_HOURS:raise ValueError('mode must be queued or exclusive')
        default,most=LEND_HOURS[mode];hours=args.get('hours',default)
        if isinstance(hours,bool) or not isinstance(hours,(int,float)) or not 0<hours<=most:
            raise ValueError('hours must be above 0 and at most %d in %s mode'%(most,mode))
        owner=args.get('owner')
        if not isinstance(owner,str) or not 1<=len(owner.strip())<=100:raise ValueError('owner must say who borrows the device (1-100 characters)')
        req={'device':args.get('device'),'mode':mode,'hours':hours,'owner':owner.strip()}
        label=args.get('label')
        if label is not None:
            if not isinstance(label,str) or not 1<=len(label.strip())<=100:raise ValueError('label must say what the device is borrowed for (1-100 characters)')
            req['label']=label.strip()
        job=self._enqueue('',None,'lend',req,args.get('request_key'))
        self.heartbeats.setdefault(job['id'],time.time())
        return job

    def _serving(self, name):
        """The server tenants that answer to `name` (a service's names, a tenant's `models`)."""
        return getattr(self.runtime,'model_tenants',lambda name:[])(name)

    def _route(self, model, devices=None):
        """The server tenant a call by `model` goes to. Of those declaring it, skipping one whose
        device is reserved for research or not in the call's `devices`: the fewest jobs waiting or
        running on its device, and at equal count the one already loaded there."""
        names=self._serving(model)
        if not names:
            raise ValueError('unknown model %r; this host serves: %s'%(model,', '.join(m['id'] for m in self.call('models',{})) or 'no models'))
        options=[];reserved=[]
        for index,name in enumerate(names):
            try:lane=self._stages('engine',{'engine':name,**({'devices':devices} if devices else {})})[0][1]
            except ValueError:continue   # not on the devices the call allows
            if self.lends.get(lane,{}).get('mode')=='exclusive':reserved.append(name);continue
            busy=self.db.execute('SELECT count(*) FROM jobs WHERE waiting=1 AND lane=?',(lane,)).fetchone()[0]
            busy+=len(self.calls[lane]) if lane in self.calls else int(self.running.get(lane) not in (None,'residency'))
            loaded=getattr(self.runtime,'server_on',{}).get(name)==lane
            options.append((busy,not loaded,index,name))
        if options:return min(options)[3]
        if reserved:return reserved[0]   # all reserved: _enqueue says until when
        served=self.call('models',{});served=next((m['devices'] for m in served if m['id']==model),[])
        raise ValueError('%s runs on %s, not on %s'%(model,' or '.join(served),' or '.join(devices)))

    def _prune_model_calls(self):
        """Calls by model are many and carry whole prompts: once an hour, those ended more than
        MODEL_CALL_KEEP_S ago are deleted, record and response. Calls by tenant name are kept."""
        now=time.time()
        if now-self.pruned<3600:return
        self.pruned=now
        rows=self.db.execute("SELECT id FROM jobs WHERE kind='engine' AND urgent=0 AND status IN (%s) AND updated<?"%','.join('?'*len(TERMINAL)),
                             (*sorted(TERMINAL),now-MODEL_CALL_KEEP_S)).fetchall()
        with self.db:self.db.executemany('DELETE FROM jobs WHERE id=?',[(row['id'],) for row in rows])
        for row in rows:shutil.rmtree(self.state/'jobs'/row['id'],ignore_errors=True)

    def _lend_view(self, lend, env=False):
        view={k:lend[k] for k in ('job_id','device','owner','label','mode','granted','deadline') if k in lend}
        if env:view['env']=lend['env']
        return view

    def _queue_view(self, where, params):
        """Who a queue holds, for resource_status: a job without its request body (an engine
        call's can be 28 MB), plus who borrows (a lend) or which engine it calls."""
        rows=self.db.execute("SELECT id,kind,status,step,project,shot,created,json_extract(request,'$.owner') AS owner,json_extract(request,'$.label') AS label,"
                             "json_extract(request,'$.mode') AS mode,json_extract(request,'$.engine') AS engine,json_extract(request,'$.model') AS model FROM jobs "+where,params)
        return [{k:v for k,v in dict(row).items() if v not in (None,'')} for row in rows]

    def _brief(self, job):
        """A job without its request: an engine call's carries the whole body (up to 28 MB), and
        the facade asks for the status of one it streams many times a second."""
        return {k:v for k,v in job.items() if k!='request'}

    def _also_running(self, lane):
        """The other engine calls running beside `running` on a device (see _engine_session)."""
        ids=[ident for ident in self.calls.get(lane,{}) if ident!=self.running.get(lane)]
        return self._queue_view('WHERE id IN (%s) ORDER BY created,rowid'%','.join('?'*len(ids)),ids) if ids else []

    def _device_view(self, lane):
        device=getattr(self.runtime,'devices',{}).get(lane)
        if device is None:return None
        return {k:getattr(device,k,None) for k in ('display','vendor','uuid','pci','placement','queue')}

    def _step_of(self, ident):
        if ident in (None,'residency'):return ident
        return self._job(ident)['step']

    def _active(self):
        """The one job `active_job` names: any job a queue is on before an idle restore."""
        jobs=[v for v in self.running.values() if v!='residency']
        self.active=jobs[0] if jobs else next(iter(self.running.values()),None)

    def _stop_lane(self, lane):
        """After a failure or cancellation: stop what is ours on that device only."""
        if lane in getattr(self.runtime,'devices',{}):self.runtime.stop_device(lane)
        else:self.runtime.stop()

    def _park(self, lane):
        if lane in getattr(self.runtime,'devices',{}):self.runtime.park(lane)
        else:self.runtime.park()

    def _queue_mode(self, lane):
        """The device's gpu-config `queue`: priority (default), fifo or affinity."""
        return getattr(getattr(self.runtime,'devices',{}).get(lane),'queue','priority')

    def _order(self, lane):
        return 'created,rowid' if self._queue_mode(lane)=='fifo' else 'urgent DESC,created,rowid'

    def _tenant_of(self, row):
        """The tenant a waiting job will run on, as loaded_on names it (a plugin's stage: its part)."""
        return row['tenant'] if row['kind'] in single_stage() else self.runtime.part_of(row['kind'],row['step'])

    def _next(self, lane):
        """The next job on `lane`, by the device's `queue` mode.

        priority (default): engine calls by tenant name first, then by creation time. Where several
        stages of one plugin's jobs wait (a part's fallback device), a job of the stage whose
        part is loaded goes first, unless the oldest has already been passed over BATCH_PASS_LIMIT
        times. Only such stages ahead of any other job are considered: a lend keeps its place by
        creation time.
        fifo: creation time only; nothing goes first and nothing is reordered.
        affinity: as priority, and beyond a plugin's stages: a job whose tenant is loaded on the device
        goes before older ones of the same tier whose tenant is not (so a card serving several
        models swaps less), each passed over at most BATCH_PASS_LIMIT times; a lend is never jumped."""
        mode=self._queue_mode(lane)
        rows=self.db.execute("SELECT id,kind,step,urgent,"+tenant_column()+" FROM jobs WHERE waiting=1 AND lane=? ORDER BY "+self._order(lane),(lane,)).fetchall()
        if not rows:return None
        head=rows[0]
        if mode=='fifo':return head
        resident=getattr(self.runtime,'resident',None)
        loaded=resident(lane) if resident and head['kind'] not in (*single_stage(),*CORE_JOBS) else None
        if loaded and head['step']!=loaded and self.passed.get(head['id'],0)<BATCH_PASS_LIMIT:
            batched=None
            for row in rows[1:]:
                if row['kind']!=head['kind']:break
                if row['step']==loaded:batched=row;break
            if batched:
                self.passed[head['id']]=self.passed.get(head['id'],0)+1
                return batched
        loaded_on=getattr(self.runtime,'loaded_on',None)
        if mode=='affinity' and loaded_on and head['kind']!='lend' and self.passed.get(head['id'],0)<BATCH_PASS_LIMIT:
            loaded=loaded_on(lane)
            if self._tenant_of(head) not in loaded:
                for row in rows[1:]:
                    if row['urgent']!=head['urgent'] or row['kind']=='lend':break
                    if self._tenant_of(row) in loaded:
                        self.passed[head['id']]=self.passed.get(head['id'],0)+1
                        return row
        return head

    def _steal(self, lane):
        """Spill: `lane`'s queue is empty, so it takes the oldest waiting call whose fallback it is and
        whose own device is busy with another job, then returns it as _next would. Only tenants with
        gpu-config `spill`; never onto a lent, blocked or drained device. A tenant already loaded here
        is taken at once; one not loaded anywhere costs the same load either way; one loaded on its
        busy device only once that device has been held SPILL_AFTER_S by another tenant's job (an
        image, an encode: a short call there is cheaper to wait for), never from its own session, never
        when the call's `devices` leave this one out. The move and the start happen under this lock, so nothing moves it back.
        Edge cases E2-E3, E6-E10, E14-E15 in docs/GPU_SCHEDULING_CASES.zh-CN.md. Called holding self.cv."""
        spills=getattr(self.runtime,'spills',None)
        if spills is None or lane in self.drained or lane in getattr(self.runtime,'lent',{}) or lane in getattr(self.runtime,'blocked',{}):return None
        kinds=single_stage()
        rows=self.db.execute("SELECT id,kind,lane,"+tenant_column()+",json_extract(request,'$.devices') AS devices FROM jobs "
                             "WHERE waiting=1 AND lane!=? AND step IN (%s) ORDER BY urgent DESC,created,rowid"%','.join('?'*len(kinds)),(lane,*kinds)).fetchall()
        for row in rows:
            req={'engine' if row['kind']=='engine' else BATCH_JOBS[row['kind']]:row['tenant']}
            if row['devices']:req['devices']=json.loads(row['devices'])
            if not spills(row['kind'],req):continue
            primary,fallback=self.runtime.stage_devices(row['kind'],req)
            if fallback!=lane or row['lane']!=primary:continue
            running=self.running.get(primary)
            if running in (None,'residency'):continue   # its own device takes it next
            name=row['tenant'];where=self.runtime.located(name)
            if where not in (None,lane):
                # Loaded on its busy device: worth a load here only if that device is held by another
                # tenant's job for a while (an image, an encode), not by a short call or by itself.
                busy=self._job(running)
                if where!=primary or name==tenant_name(busy['kind'],busy['request']):continue
                if time.monotonic()-self.started.get(primary,0)<SPILL_AFTER_S:continue
            with self.db:moved=self.db.execute('UPDATE jobs SET lane=?,updated=? WHERE id=? AND waiting=1',(lane,time.time(),row['id'])).rowcount
            if moved:return self._next(lane)
        return None

    def _relane(self):
        """Waiting jobs follow where their stage goes now (the runtime's lane_for), keeping their
        places by creation time: an encode, an engine call or an enhancement whose tenant declares a
        fallback moves onto it while its own device is lent exclusively (a queued lend or a block is
        waited out where it is, as before), and back once the own device is usable again, so that
        the two devices never take turns with one tenant. Looked at whenever a queue chooses its next
        job; only first stages, only waiting ones move. Called holding self.cv."""
        devices=getattr(self.runtime,'stage_devices',None)
        if devices is None:return 0
        # A job is placed by the state at its submission, so only a change of state can move one:
        # with none since the last look, the waiting requests (an engine call's up to 28 MB) are not
        # read again.
        state=(sorted(getattr(self.runtime,'lent',{})),sorted(getattr(self.runtime,'blocked',{})),sorted((k,v['mode']) for k,v in self.lends.items()))
        if state==self.relaned:return 0
        self.relaned=state
        down=lambda name:name in getattr(self.runtime,'lent',{}) or name in getattr(self.runtime,'blocked',{})
        moved=0
        rows=self.db.execute("SELECT id,project,shot,kind,step,request,lane FROM jobs WHERE waiting=1 AND kind!='lend' ORDER BY created,rowid").fetchall()
        for row in rows:
            request=json.loads(row['request'])
            if 'route' in request and self.lends.get(row['lane'],{}).get('mode')=='exclusive':
                # Called by a service name, on a device now reserved for research: to another tenant
                # of that service if one can take it (a spread service has no fallback to wait on).
                try:engine=self._route(request['route'],request.get('devices'));lane=self._stages('engine',{**request,'engine':engine})[0][1]
                except ValueError:continue   # no longer served: it waits, and fails when run
                if lane!=row['lane']:
                    request['engine']=engine
                    with self.db:moved+=self.db.execute('UPDATE jobs SET lane=?,request=?,request_hash=?,updated=? WHERE id=? AND waiting=1',
                        (lane,packed(request),self._fingerprint(row['project'],row['shot'],row['kind'],request),time.time(),row['id'])).rowcount
                    continue
            try:
                stage,lane=self._stages(row['kind'],request)[0]
                if stage!=row['step']:continue   # only a first stage moves
                primary,fallback=devices(row['kind'],request)
            except ValueError:continue
            if fallback is None or lane==row['lane']:continue
            if row['lane']==primary and lane==fallback and self.lends.get(primary,{}).get('mode')=='exclusive':pass
            elif row['lane']==fallback and lane==primary and not down(primary):pass
            else:continue
            with self.db:moved+=self.db.execute('UPDATE jobs SET lane=?,updated=? WHERE id=? AND waiting=1',(lane,time.time(),row['id'])).rowcount
        if moved:self.cv.notify_all()
        return moved

    def _lend(self, job, lane):
        """Hand `lane` to a borrower and wait, holding this queue, until it comes back. A lend restored
        after a restart is already granted: only the waiting resumes."""
        ident,req=job['id'],job['request']
        with self.cv:
            restored=self.lends.get(lane,{}).get('job_id')==ident
            if not restored and time.time()-self.heartbeats.get(ident,0)>LEND_HEARTBEAT_S:
                raise RuntimeError('the borrower stopped answering before %s was free; nothing was lent'%lane)
        if restored:record={k:v for k,v in self.lends[lane].items() if k!='env'}
        else:
            now=time.time()
            record={'job_id':ident,'device':lane,'owner':req['owner'],'mode':req['mode'],'granted':now,'deadline':now+req['hours']*3600}
            if req.get('label'):record['label']=req['label']
            env=self.runtime.lend(lane,record)
        with self.cv:
            if not restored:
                with self.db:self.db.execute('UPDATE jobs SET lent=? WHERE id=?',(packed(record),ident))
                self.lends[lane]=dict(record,env=env);self.heartbeats[ident]=time.time()
                self._relane()
            self.cv.notify_all()
            reason=None
            while reason is None:
                now=time.time();lend=self.lends[lane]
                if lend.get('returned'):reason='returned'
                elif self._job(ident)['status']=='cancelling':reason='cancelled'
                elif now>=lend['deadline']:reason='deadline passed'
                elif lend['mode']=='queued' and now-self.heartbeats.get(ident,0)>LEND_HEARTBEAT_S:reason='borrower stopped answering'
                elif self.closing:raise ServiceStopping()   # still lent: the next start restores it
                else:
                    until=lend['deadline']
                    if lend['mode']=='queued':until=min(until,self.heartbeats.get(ident,0)+LEND_HEARTBEAT_S)
                    self.cv.wait(timeout=min(5,max(.05,until-now)))
            # The end is recorded before the device is reclaimed, so a restart in between cannot restore it.
            with self.db:self.db.execute('UPDATE jobs SET lent=NULL WHERE id=?',(ident,))
            del self.lends[lane]
        left=self.runtime.reclaim(lane)
        with self.cv:self.heartbeats.pop(ident,None)
        return {**record,'returned':time.time(),'reason':reason,'left_behind':left}

    def _work(self, lane):
        """One device's queue: its waiting stages in order, engine calls first; idle restore after."""
        while True:
            row=job=None;recheck=False
            with self.cv:
                while not self.closing:
                    if lane in self.resumed:
                        row=(self.resumed.pop(lane),);break   # a lend restored at startup holds this queue first
                    if lane in getattr(self.runtime,'blocked',{}):
                        self.cv.wait(timeout=BLOCKED_RECHECK_S);recheck=True;break
                    self._relane()
                    row=self._next(lane) or self._steal(lane)
                    if row:break
                    if lane in self.drained:
                        if lane not in self.emptied:break   # emptied below instead of the idle restore
                        self.cv.wait();continue
                    if self.needs_park[lane] and not self.draining:
                        # Coalesce a client's consecutive engine calls before the idle restore;
                        # a server tenant declaring `linger_s` keeps the device that long after its last call.
                        until=max(time.monotonic()+IDLE_COALESCE_S,self._linger_until(lane))
                        while not row and not self.closing and (left:=until-time.monotonic())>0:
                            self.cv.wait(timeout=left);row=self._next(lane) or self._steal(lane)
                        if row:break
                        if not self.draining:break
                    self.cv.wait(timeout=SPILL_AFTER_S)   # also looks again for a call to spill (_steal)
                if self.closing:return
                if recheck:pass
                elif row is None:
                    self.running[lane]='residency';self.needs_park[lane]=False
                else:
                    job=self._job(row[0])
                    self.running[lane]=job['id'];self.passed.pop(job['id'],None);self.started[lane]=time.monotonic()
                    with self.db:self.db.execute("UPDATE jobs SET status='running',waiting=0,updated=? WHERE id=?",(time.time(),job['id']))
                self._active();self.cv.notify_all()
            if recheck:
                try:clear=not self.runtime.check_device(lane)
                except Exception:clear=False
                if clear:
                    with self.cv:self.needs_park[lane]=hasattr(self.runtime,'park');self.cv.notify_all()
                continue
            if row is None:
                if lane in self.drained:
                    self._stop_lane(lane)
                    with self.cv:self.emptied.add(lane)
                else:
                    try:self._park(lane)
                    except Exception as exc:
                        self.runtime.park_error=str(exc)
                with self.cv:
                    self.running.pop(lane,None);self._active()
                    if self.draining and not self.running:self.runtime.stop()
                    self.cv.notify_all()
                continue
            if job['step']=='engine' and self._slots(job)>1:
                self._engine_session(lane,job);continue
            result=error=following=None
            stage=job['step'] or job['kind']
            try:
                out=self.state/'jobs'/job['id']
                if stage=='engine':
                    # On the device whose queue it waited in: the tenant's own, or its fallback.
                    result=self.runtime.engine(job['request'],out,lane)
                elif stage=='lend':
                    result=self._lend(job,lane)
                else:
                    # A batch job, or a plugin tenant's stage, on the device whose queue it waited in.
                    ident=job['id']
                    def cancelled():
                        with self.cv:return self._job(ident)['status']=='cancelling'
                    stages=self._stages(job['kind'],job['request']);names=[name for name,_ in stages]
                    result=self.runtime.run_stage(job['kind'],stage,job['request'],out,lane,cancelled)
                    if stage in names[:-1]:following=stages[names.index(stage)+1];result=None
                    elif isinstance(result,dict) and 'output' in result:
                        # A job that leaves a file: its hash, as reported (checked here) or computed.
                        if 'sha256' not in result:result['sha256']=digest(result['output'])
                        elif digest(result['output'])!=result['sha256']:raise RuntimeError('output does not match its reported hash')
            except ServiceStopping:
                # The lend stays granted and its job running; nothing of ours is on the device.
                with self.cv:self.running.pop(lane,None);self._active();self.cv.notify_all()
                return
            except Exception as exc:
                error=str(exc);self._stop_lane(lane)
            with self.cv:
                cancelled=self._job(job['id'])['status']=='cancelling'
                if following and not error and not cancelled:
                    # On to the next device's queue, keeping its place there by creation time.
                    with self.db:self.db.execute('UPDATE jobs SET step=?,lane=?,waiting=1,updated=? WHERE id=?',(following[0],following[1],time.time(),job['id']))
                else:
                    status='cancelled' if cancelled else ('failed' if error else 'completed')
                    with self.db:self.db.execute('UPDATE jobs SET status=?,result=?,error=?,waiting=0,updated=? WHERE id=?',(status,packed(result) if result else None,error,time.time(),job['id']))
                self.running.pop(lane,None);self._active()
                self.needs_park[lane]=hasattr(self.runtime,'park')
                if stage=='engine' and not error:self._linger(lane,job)
                if cancelled and stage!='lend':self._stop_lane(lane)
                if self.draining and not self.running:self.runtime.stop()
                self.cv.notify_all()


    def _linger(self, lane, job):
        """After an engine call: its tenant keeps `lane` for its `linger_s` before the idle restore."""
        name=job['request']['engine'];linger=getattr(self.runtime,'tenants',{}).get(name,{}).get('linger_s')
        if linger:self.lingers[lane]=(name,time.monotonic()+linger)

    def _linger_until(self, lane):
        """Until when the idle restore of `lane` waits: while the tenant that lingers is still there."""
        name,until=self.lingers.get(lane,(None,0))
        return until if name and getattr(self.runtime,'server_on',{}).get(name)==lane else 0

    def _slots(self, job):
        """How many calls the job's server tenant takes at once (gpu-config `slots`, default 1)."""
        if not hasattr(self.runtime,'engine_session'):return 1
        return getattr(self.runtime,'tenants',{}).get(job['request']['engine'],{}).get('slots',1)

    def _engine_session(self, lane, first):
        """Calls to a server tenant that declares `slots` above 1, up to that many at once on `lane`:
        the server is placed once and the device held while its calls keep coming to the head of this
        queue. Anything else at the head (another tenant's call, a plugin's stage, a lend) ends the session:
        nothing more joins, the calls running finish, then the queue goes on as usual. A call that
        fails stops the device once the session is over, as a failed job does."""
        name=first['request']['engine'];slots=self._slots(first)
        calls=self.calls[lane]={};errors=[];placed=False
        def run(job,server,device):
            result=error=None
            try:result=self.runtime.engine_call(job['request'],self.state/'jobs'/job['id'],server,device)
            except Exception as exc:error=str(exc)
            with self.cv:
                status='cancelled' if self._job(job['id'])['status']=='cancelling' else ('failed' if error else 'completed')
                with self.db:self.db.execute('UPDATE jobs SET status=?,result=?,error=?,waiting=0,updated=? WHERE id=?',
                                             (status,packed(result) if result else None,error,time.time(),job['id']))
                del calls[job['id']]
                if error:errors.append(error)
                if calls:self.running[lane]=next(iter(calls))
                self.cv.notify_all()
        try:
            with self.runtime.engine_session(name,lane) as (server,device):
                placed=True
                with self.cv:
                    job=first
                    while True:
                        if job is not None:
                            calls[job['id']]=threading.Thread(target=run,args=(job,server,device),daemon=True)
                            calls[job['id']].start()
                            if self.running.get(lane) not in calls:self.running[lane]=job['id']
                        job=None
                        if len(calls)<slots and not self.closing:
                            row=self._next(lane)
                            if row and row['kind']=='engine' and self._job(row['id'])['request']['engine']==name:
                                job=self._job(row['id'])
                                with self.db:self.db.execute("UPDATE jobs SET status='running',waiting=0,updated=? WHERE id=?",(time.time(),job['id']))
                                continue
                        if not calls:break
                        self.cv.wait()
        except Exception as exc:
            errors.append(str(exc))
            # The server could not be placed: no call ran, and the first one fails with it.
            if not placed:
                with self.cv,self.db:
                    self.db.execute("UPDATE jobs SET status=?,error=?,waiting=0,updated=? WHERE id=?",
                                    ('cancelled' if self._job(first['id'])['status']=='cancelling' else 'failed',str(exc),time.time(),first['id']))
        if errors:self._stop_lane(lane)
        with self.cv:
            if not errors:self._linger(lane,first)
            self.calls.pop(lane,None);self.running.pop(lane,None);self._active()
            self.needs_park[lane]=hasattr(self.runtime,'park')
            if self.draining and not self.running:self.runtime.stop()
            self.cv.notify_all()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            # base64 of the largest importable media (64 MiB) plus the envelope
            if not 0<length<=90*1024*1024:raise ValueError('request too large or empty')
            req=json.loads(self.rfile.read(length))
            result=self.server.service.call(req['action'],req.get('arguments',{}))
            payload={'result':result};status=200
        except (ValueError,KeyError,TypeError) as e:payload={'error':str(e)};status=400
        except Exception as e:payload={'error':str(e)};status=500
        body=packed(payload).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
    daemon_threads=True

def serve(service_class, root):
    """Run `service_class` for the checkout `root` on <state>/runtime.sock until SIGTERM."""
    parser=argparse.ArgumentParser();parser.add_argument('--state',required=True);args=parser.parse_args()
    service=service_class(root,args.state)
    sock=service.state/'runtime.sock'
    with contextlib.suppress(FileNotFoundError):sock.unlink()
    server=Server(str(sock),Handler);server.service=service;os.chmod(sock,0o660)
    def shutdown(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();service.close();sock.unlink(missing_ok=True)


if __name__=='__main__':serve(Service,Path(__file__).resolve().parents[1])
