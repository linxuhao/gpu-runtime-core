#!/usr/bin/env python3
"""What each GPU is doing and who waits for it: load, VRAM, power and temperature from the cards,
lends and queues from the GPU runtime, and the prefill and decode speeds of its text servers.

  gpu_top.py              one snapshot, like nvidia-smi
  gpu_top.py -w [SEC]     redraw every SEC seconds (default 2) until Ctrl-C, like btop

Cards come from the runtime's devices (from state/gpu-config.json while it is down):
NVIDIA through nvidia-smi, AMD through sysfs. Processes are those holding VRAM on the card; AMD
ones are found through /proc/*/fdinfo, so another user's are counted only in the total.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gpu_borrow import call, default_socket

MIB=1024*1024
PROCESS_MIN_MIB=64      # a process holding less VRAM than this is a context, not a user (Vulkan opens every card)


def number(text):
    try:return float(text)
    except (TypeError,ValueError):return None


def read(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def nvidia_cards():
    """uuid -> metrics, with processes {pid: MiB}."""
    fields='uuid,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu,fan.speed'
    try:
        gpus=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
    except (OSError,subprocess.TimeoutExpired):return {}
    cards={}
    for line in gpus.splitlines():
        uuid,name,*values=[v.strip() for v in line.split(',')]
        util,used,total,power,cap,temp,fan=(number(v) for v in values)
        cards[uuid]={'model':name,'util':util,'used_mib':used,'total_mib':total,'power_w':power,'cap_w':cap,'temp_c':temp,'fan':fan,'processes':{}}
    for line in apps.splitlines():
        pid,uuid,used=[v.strip() for v in line.split(',')]
        if uuid in cards and number(used):cards[uuid]['processes'][int(pid)]=number(used)
    return cards


def amd_processes():
    """PCI address -> {pid: MiB of VRAM}, from the DRM fdinfo of the processes we may read."""
    found={}
    for fdinfo in Path('/proc').glob('[0-9]*/fdinfo'):
        clients={}
        try:entries=list(fdinfo.iterdir())
        except OSError:continue
        for entry in entries:
            text=read(entry)
            if not text or 'drm-pdev' not in text:continue
            info=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
            vram=info.get('drm-memory-vram') or info.get('drm-resident-vram') or '0 KiB'
            amount,unit=(vram.split()+['KiB'])[:2]
            mib=float(amount)*{'KiB':1/1024,'MiB':1,'GiB':1024}.get(unit,1/1024/1024)
            clients[(info['drm-pdev'].strip(),info.get('drm-client-id','').strip())]=mib   # one client may have several fds
        for (pci,_),mib in clients.items():
            pid=int(fdinfo.parent.name);found.setdefault(pci,{})
            found[pci][pid]=found[pci].get(pid,0)+mib
    return found


def amd_card(pci, processes):
    base=Path('/sys/bus/pci/devices')/pci
    hwmon=next(iter(sorted((base/'hwmon').glob('hwmon*'))),None)
    def sensor(name, scale):
        value=number(read(hwmon/name)) if hwmon else None
        return value/scale if value is not None else None
    power=sensor('power1_average',1e6)
    if power is None:power=sensor('power1_input',1e6)
    used,total=number(read(base/'mem_info_vram_used')),number(read(base/'mem_info_vram_total'))
    return {'model':None,'util':number(read(base/'gpu_busy_percent')),
            'used_mib':used/MIB if used is not None else None,'total_mib':total/MIB if total is not None else None,
            'power_w':power,'cap_w':sensor('power1_cap',1e6),'temp_c':sensor('temp1_input',1000),
            'fan':None,'processes':processes.get(pci,{})}


def runtime_state(socket_path):
    try:return call(socket_path,'resource_status'),None
    except (OSError,SystemExit) as exc:return None,str(exc) or type(exc).__name__


def devices_of(state, socket_path):
    """lane -> {display, vendor, uuid, pci}: from the runtime, else its configuration file."""
    if state and all(info.get('device') for info in state['lanes'].values()):
        return {lane:info['device'] for lane,info in state['lanes'].items()}
    try:config=json.loads((Path(socket_path).parent/'gpu-config.json').read_text())
    except (OSError,ValueError):return {}
    return {lane:{'display':spec.get('name',lane),**spec} for lane,spec in config.get('devices',{}).items()}


def tenants_by_pid(state):
    """pid -> the runtime's name for it: a tenant, or a plugin tenant's part (<tenant>.<part>)."""
    workers=(state or {}).get('workers') or {}
    return {pid:name for name,pid in {**workers.get('engines',{}),**workers.get('batches',{}),**workers.get('parts',{})}.items() if pid}


def label(pid, known):
    """A process's runtime name if it or an ancestor has one, else its command, shortened."""
    walk=pid
    for _ in range(8):
        if walk in known:return known[walk]
        stat=read('/proc/%d/stat'%walk)
        if not stat:break
        walk=int(stat.rsplit(')',1)[1].split()[1])
        if walk<=1:break
    argv=(read('/proc/%d/cmdline'%pid) or '').split('\0')
    words=[os.path.basename(argv[0])] if argv[0] else ['?']
    if words[0].startswith('python'):
        script=[a for a in argv[1:] if a and not a.startswith('-')][:1]
        if '-m' in argv:script=[argv[argv.index('-m')+1]] if len(argv)>argv.index('-m')+1 else []
        words+= [os.path.basename(s) for s in script]
    return ' '.join(words)


# ---- drawing -------------------------------------------------------------------------------
class Paint:
    def __init__(self, on):self.on=on
    def __call__(self, text, code):return '\033[%sm%s\033[0m'%(code,text) if self.on else text
    def level(self, text, fraction):
        return self(text,'32' if fraction<.5 else '33' if fraction<.8 else '31')


def bar(fraction, width, paint):
    if fraction is None:return paint('·'*width,'2')
    fraction=min(max(fraction,0),1);eighths=round(fraction*width*8)
    full,part=divmod(eighths,8)
    body='█'*full+(' ▏▎▍▌▋▊▉'[part] if part else '')
    return paint.level(body,fraction)+paint('░'*(width-len(body)),'2')


MIN_QUEUE=3   # queue entries listed per device however small the screen


def ago(seconds):
    seconds=max(0,int(seconds))
    if seconds<60:return '%ds'%seconds
    if seconds<3600:return '%dm'%(seconds//60)
    if seconds<86400:return '%dh%02dm'%(seconds//3600,seconds//60%60)
    return '%dd%02dh'%(seconds//86400,seconds//3600%24)


def what_of(job):
    what=job['kind']+('/'+job['step'] if job.get('step') and job['step']!=job['kind'] else '')
    return what+(' ('+job['mode']+')' if job.get('mode') else '')


def whose(job):
    """A lend's borrower and, when they gave one, what it is for."""
    return job.get('owner','')+(' (%s)'%job['label'] if job.get('label') else '')


def describe(job, now):
    """One queue entry: id, what it is, whose."""
    who=whose(job) or job.get('engine') or '/'.join(x for x in (job.get('project'),job.get('shot')) if x)
    if job.get('model') and job.get('engine'):who=job['model']+' -> '+job['engine']
    return '%-26s %-18s %6s ago  %s'%(job['id'],what_of(job),ago(now-job['created']),who)   # whose last: a narrow screen cuts it


def unlisted(hidden, more):
    """The queue entries left out, in one line: how many, of what and whose."""
    groups=Counter('%s %s'%(what_of(job),whose(job) or job.get('model') or job.get('engine') or job.get('project') or '') for job in hidden)
    parts=['%d %s'%(n,key.strip()) for key,n in groups.most_common(3)]
    if len(groups)>3:parts.append('%d other'%sum(n for _,n in groups.most_common()[3:]))
    if more>len(hidden):parts.append('%d not listed by the runtime'%(more-len(hidden)))
    return '        ... %d more: %s'%(more,', '.join(parts))


def clip(line, width):
    """`line` cut to `width` columns on screen; colour codes take none."""
    out=[];seen=0;i=0
    while i<len(line):
        if line[i]=='\033':
            end=line.find('m',i);end=len(line)-1 if end<0 else end
            out.append(line[i:end+1]);i=end+1;continue
        if seen==width:return ''.join(out)+('\033[0m' if '\033[' in line else '')
        out.append(line[i]);seen+=1;i+=1
    return line


def speeds(text, paint):
    """A text server's prefill and decode speeds (computed prompt tokens; see text_rates): vLLM's in all, then per
    stream; llama-server's from its own timings, then each slot's request."""
    tps=lambda value:'%5s'%('?' if value is None else '%.0f'%value)
    pair=lambda item:'prefill %s tok/s  decode %s tok/s'%(tps(item.get('prefill_tps')),tps(item.get('decode_tps')))
    head='  '+paint('text ','1')+' %-10s %-9s '%(text['tenant'],text['kind'])
    if not text.get('window_s'):return [head+paint('measuring...','2')]
    pad=' '*(len('  text ')+22)
    if text['kind']=='vllm':
        extra=['%d running'%text['running']]+(['%d waiting'%text['waiting']] if text['waiting'] else [])+['cache hits %d%%'%round(100*text['cache_hit'])]
        if text.get('kv') is not None:extra.append('KV %d%%'%round(100*text['kv']))
        if text.get('accept') is not None:extra.append('MTP accepted %d%%'%round(100*text['accept']))
        stream=text['stream'];first=stream.get('first_token_s')
        return [head+'all     '+pair(text)+'  '+', '.join(extra)+paint('  (last %ds)'%text['window_s'],'2'),
                pad+'stream  '+pair(stream)+('  first token %.1f s'%first if first is not None else '')+paint('  (mean per request)','2')]
    busy='  %d/%d slots busy'%(text['running'],text['slots'])
    if text.get('metrics'):
        ages=['%s: last request %ds ago'%(phase,text[phase+'_age_s']) for phase in ('prefill','decode') if phase+'_age_s' in text]
        accept='  MTP accepted %d%%'%round(100*text['accept']) if text.get('accept') is not None else ''
        lines=[head+'engine  '+pair(text)+' per stream'+accept+busy+paint('  (last %ds%s)'%(text['window_s'],''.join('; '+age for age in ages)),'2')]
    else:lines=[head+paint('speeds need llama-server --metrics','2')+busy]
    for slot in text['streams']:
        lines.append(pad+'slot %-2d '%slot['slot']+'prompt %d (%d cached), %d decoded'%(slot['prompt'],slot['cached'],slot['decoded'])
                     +('' if slot['busy'] else paint('  done','2')))
    return lines


def render(state, error, devices, nvidia, amd, width, paint, now=None, height=None):
    """The screen. With `height`, each device's queue is listed only as far as the screen has room
    (at least MIN_QUEUE entries; the rest in one line), and every line is cut to `width`."""
    now=now or time.time()
    lines=frame(state,error,devices,nvidia,amd,width,paint,now,None)
    if height:
        longest=max([len((info or {}).get('waiting',[])) for info in (state or {}).get('lanes',{}).values()]+[0])
        for limit in range(longest-1,MIN_QUEUE-1,-1):
            if len(lines)<=height:break
            lines=frame(state,error,devices,nvidia,amd,width,paint,now,limit)
        lines=[clip(line,width) for line in lines]
    return '\n'.join(lines)


def frame(state, error, devices, nvidia, amd, width, paint, now, limit):
    """The screen's lines, each device's queue listed up to `limit` entries (None: all the runtime sent)."""
    lines=[]
    head='GPU runtime  '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now))
    if state:
        # Waiting in a device's queue, a job between two stages included (its status is `running`).
        waiting=sum(info.get('queued',0) for info in state.get('lanes',{}).values())
        head+='  %s  %d waiting'%(state['mode']+(' (draining)' if state.get('draining') else ''),waiting)
    else:head+='  '+paint('runtime unreachable: '+error,'31')
    lines.append(paint(head,'1'))
    for warning in (state or {}).get('warnings',[]):lines.append(paint('! '+warning,'1;31'))
    lines.append('')
    known=tenants_by_pid(state);span=max(10,min(40,width-40))
    for lane,device in devices.items():
        card=nvidia.get(device.get('uuid')) if device.get('vendor')=='nvidia' else amd_card(device['pci'],amd) if device.get('pci') else None
        info=(state or {}).get('lanes',{}).get(lane,{})
        title=paint(lane,'1;36')+'  '+device.get('display',lane)+(paint('  ('+card['model']+')','2') if card and card.get('model') else '')
        if info.get('lent'):
            lent=info['lent'];title+='  '+paint('lent %s to %s until %s'%(lent['mode'],whose(lent),time.strftime('%m-%d %H:%M',time.localtime(lent['deadline']))),'35')
        for lend in info.get('shared') or []:
            title+='  '+paint('shared %s GiB with %s until %s'%(lend['vram_gib'],whose(lend),time.strftime('%m-%d %H:%M',time.localtime(lend['deadline']))),'35')
        if info.get('blocked'):title+='  '+paint('blocked: '+info['blocked'],'31')
        if info.get('drained'):title+='  '+paint('drained: finishes its queue, takes nothing new','33')
        modes=(info.get('device') or {})
        if modes.get('placement') and (modes['placement'],modes.get('queue'))!=('exclusive','priority'):
            title+=paint('  [%s placement, %s queue]'%(modes['placement'],modes.get('queue')),'2')
        lines.append(title)
        if card is None:lines+=['  '+paint('no readings (card not found)','31'),''];continue
        util=card['util'];used,total=card['used_mib'],card['total_mib']
        extras=[]
        if card['power_w'] is not None:extras.append('%3.0f'%card['power_w']+('/%.0f W'%card['cap_w'] if card['cap_w'] else ' W'))
        if card['temp_c'] is not None:extras.append('%.0f°C'%card['temp_c'])
        if card['fan'] is not None:extras.append('fan %.0f%%'%card['fan'])
        lines.append('  GPU  %s %s   %s'%(bar(util/100 if util is not None else None,span,paint),'%3.0f%%'%util if util is not None else '  ?%','  '.join(extras)))
        fraction=used/total if used is not None and total else None
        lines.append('  VRAM %s %s'%(bar(fraction,span,paint),'%.1f/%.1f GiB'%(used/1024,total/1024) if fraction is not None else '?'))
        if state:
            running,active=info.get('running'),info.get('active')
            if running:lines.append('  '+paint('run ','1')+'  '+describe(running,now))
            elif active=='residency':lines.append('  '+paint('run ','1')+'  restoring idle residency')
            elif active:lines.append('  '+paint('run ','1')+'  %s %s'%(active,info.get('step') or ''))   # a runtime older than `running`
            for job in info.get('also_running',[]):lines.append('        '+describe(job,now))
            waiting=info.get('waiting',[]);shown=waiting[:limit];digits=len(str(len(shown)))
            for index,job in enumerate(shown):lines.append('  '+(paint('queue','1') if index==0 else '     ')+' %*d. %s'%(digits,index+1,describe(job,now)))
            if info.get('queued',0)>len(shown):lines.append(unlisted(waiting[len(shown):],info['queued']-len(shown)))
            if not active and not waiting:lines.append('  '+paint('idle, nothing queued','2'))
            for text in info.get('text',[]):lines+=speeds(text,paint)
        listed=0
        for pid,mib in sorted(card['processes'].items(),key=lambda item:-item[1]):
            if mib<PROCESS_MIN_MIB:continue
            listed+=mib
            lines.append('  '+paint('proc ','2')+' %-8d %-32s %8.1f GiB'%(pid,label(pid,known)[:32],mib/1024))
        if device.get('vendor')=='amd' and used is not None and used-listed>=1024:
            lines.append('  '+paint('proc ','2')+' %-8s %-32s %8.1f GiB'%('-','(other users, unreadable here)',(used-listed)/1024))
        lines.append('')
    return lines


def snapshot(socket_path, width, paint, height=None):
    state,error=runtime_state(socket_path)
    return render(state,error,devices_of(state,socket_path),nvidia_cards(),amd_processes(),width,paint,height=height)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--socket',default=default_socket(__file__))
    parser.add_argument('-w','--watch',type=float,nargs='?',const=2.0,metavar='SEC',help='redraw every SEC seconds (default 2)')
    parser.add_argument('--no-color',action='store_true')
    args=parser.parse_args(argv)
    paint=Paint(sys.stdout.isatty() and not args.no_color and 'NO_COLOR' not in os.environ)
    fit=sys.stdout.isatty()   # to the screen; piped, everything
    if args.watch is None:
        size=shutil.get_terminal_size()
        print(snapshot(args.socket,size.columns,paint,size.lines-1 if fit else None));return 0
    sys.stdout.write('\033[?1049h\033[?25l')      # alternate screen, no cursor
    try:
        while True:
            size=shutil.get_terminal_size()
            screen=snapshot(args.socket,size.columns,paint,size.lines-1 if fit else None)
            sys.stdout.write('\033[H\033[J'+screen+'\n'+paint('every %gs, Ctrl-C to quit'%args.watch,'2'));sys.stdout.flush()
            time.sleep(max(args.watch,.5))
    except KeyboardInterrupt:return 0
    finally:sys.stdout.write('\033[?25h\033[?1049l');sys.stdout.flush()


if __name__=='__main__':sys.exit(main())
