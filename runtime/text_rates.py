"""Prefill and decode speeds of the text servers (llama-server, vLLM), for gpu-top.

The service reads every running one each SAMPLE_S seconds and keeps WINDOW_S seconds of readings;
speeds are what happened in that window. Prefill counts only the prompt tokens computed, not those
served from the prefix cache.

- llama-server: /slots, per slot (exact: each slot is one request). The total is their sum.
- vLLM: /metrics, the whole server. It has no per-request progress, so "per stream" is the mean
  over the requests of the window: decode from the inter-token latencies, prefill from the time to
  first token (which includes any wait in vLLM's own queue).
"""
from collections import deque
import json
import os
from pathlib import Path
import re
import threading
import urllib.request

SAMPLE_S=2
WINDOW_S=30
VLLM={'prompt':'vllm:prompt_tokens_total','hits':'vllm:prefix_cache_hits_total','queries':'vllm:prefix_cache_queries_total',
      'generated':'vllm:generation_tokens_total','itl_s':'vllm:inter_token_latency_seconds_sum',
      'ttft_s':'vllm:time_to_first_token_seconds_sum','ttft_n':'vllm:time_to_first_token_seconds_count',
      'drafts':'vllm:spec_decode_num_draft_tokens_total','accepted':'vllm:spec_decode_num_accepted_tokens_total',
      'running':'vllm:num_requests_running','waiting':'vllm:num_requests_waiting','kv':'vllm:kv_cache_usage_perc'}
GAUGES=('running','waiting','kv')


def kind_of(command):
    """'llama.cpp', 'vllm' or None (not a text server) from a server's command line."""
    words=[os.path.basename(str(word)) for word in command]
    if 'llama-server' in words:return 'llama.cpp'
    if any(word=='vllm' or word.startswith('vllm.') for word in words):return 'vllm'
    return None


def describe(command):
    """What a text server runs, for /v1/models: framework, build and weights, from its command line."""
    kind=kind_of(command)
    if kind is None:return {}
    words=[str(word) for word in command]
    if kind=='llama.cpp':
        server=next(word for word in words if os.path.basename(word)=='llama-server')
        build=os.path.basename(os.path.dirname(server)) or None
        weights=words[words.index('-m')+1] if '-m' in words[:-1] else None
    else:
        # .../venv/bin/python -m vllm... serve <weights>: the version from the venv's installed package
        found=sorted(Path(words[0]).parent.parent.glob('lib/python*/site-packages/vllm-*.dist-info'))
        build='vllm '+found[-1].name[len('vllm-'):-len('.dist-info')] if found else None
        weights=words[words.index('serve')+1] if 'serve' in words[:-1] else None
    return {'framework':kind,'build':build,'weights':os.path.basename(weights.rstrip('/')) if weights else None}


def parse_vllm(text):
    """The counters of VLLM from a /metrics page, each summed over its label sets."""
    found={};names={v:k for k,v in VLLM.items()}
    for line in text.splitlines():
        match=re.match(r'([a-zA-Z_:]+)(\{[^}]*\})?\s+(\S+)$',line)
        if match and match.group(1) in names:
            key=names[match.group(1)];found[key]=found.get(key,0.0)+float(match.group(3))
    return found


def parse_llama(slots):
    """slot id -> its request, from /slots."""
    found={}
    for slot in slots:
        token=slot.get('next_token') or {}
        if isinstance(token,list):token=token[0] if token else {}
        found[slot['id']]={'busy':bool(slot.get('is_processing')),'task':slot.get('id_task'),
                           'computed':slot.get('n_prompt_tokens_processed') or 0,'cached':slot.get('n_prompt_tokens_cache') or 0,
                           'decoded':token.get('n_decoded') or 0}
    return found


def read(url, kind, timeout=2):
    if kind=='vllm':
        with urllib.request.urlopen(url+'/metrics',timeout=timeout) as response:return parse_vllm(response.read().decode())
    with urllib.request.urlopen(url+'/slots',timeout=timeout) as response:return parse_llama(json.loads(response.read()))


def rate(amount, seconds):return round(amount/seconds,1) if seconds>0 else None


class TextRates:
    def __init__(self):
        self.lock=threading.Lock();self.samples={}   # tenant -> (device, kind, deque of (time, reading))

    def record(self, tenant, device, kind, reading, now):
        with self.lock:
            if self.samples.get(tenant,(None,None))[:2]!=(device,kind):self.samples[tenant]=(device,kind,deque())
            readings=self.samples[tenant][2]
            if kind=='vllm' and readings and reading.get('generated',0)<readings[-1][1].get('generated',0):
                readings.clear()   # a new server: its counters started over
            readings.append((now,reading))
            while len(readings)>2 and now-readings[1][0]>=WINDOW_S:readings.popleft()

    def keep(self, tenants):
        """Forget the servers no longer running."""
        with self.lock:
            for tenant in set(self.samples)-set(tenants):del self.samples[tenant]

    def lane(self, device):
        """The speeds of the text servers on a device, for resource_status."""
        with self.lock:
            found=[(tenant,kind,list(readings)) for tenant,(on,kind,readings) in sorted(self.samples.items()) if on==device]
        return [view for view in (self.view(tenant,kind,readings) for tenant,kind,readings in found) if view]

    @staticmethod
    def view(tenant, kind, readings):
        if len(readings)<2:return {'tenant':tenant,'kind':kind,'window_s':0}
        (start,first),(end,last)=readings[0],readings[-1];seconds=end-start
        out={'tenant':tenant,'kind':kind,'window_s':round(seconds)}
        if kind=='vllm':
            delta={k:last.get(k,0)-first.get(k,0) for k in VLLM if k not in GAUGES}
            cached=delta['hits']/delta['queries'] if delta['queries']>0 else 0
            computed=delta['prompt']*(1-cached)
            out.update(running=int(last.get('running',0)),waiting=int(last.get('waiting',0)),kv=round(last['kv'],2) if 'kv' in last else None,
                       prefill_tps=rate(computed,seconds),decode_tps=rate(delta['generated'],seconds),cache_hit=round(cached,2),
                       stream={'prefill_tps':rate(computed,delta['ttft_s']),'decode_tps':rate(delta['generated'],delta['itl_s']),
                               'first_token_s':round(delta['ttft_s']/delta['ttft_n'],2) if delta['ttft_n'] else None},
                       accept=round(delta['accepted']/delta['drafts'],2) if delta['drafts'] else None)
            return out
        # llama-server: what each slot did between two readings, added up over the window.
        done={}
        for (_,before),(_,after) in zip(readings,readings[1:]):
            for slot,now in after.items():
                was=before.get(slot)
                if was is not None and was['task']==now['task']:
                    computed,decoded=max(0,now['computed']-was['computed']),max(0,now['decoded']-was['decoded'])
                else:computed,decoded=now['computed'],now['decoded']   # a request that started in between
                total=done.setdefault(slot,[0,0]);total[0]+=computed;total[1]+=decoded
        streams=[]
        for slot,now in sorted(last.items()):
            computed,decoded=done.get(slot,(0,0))
            if not now['busy'] and not computed and not decoded:continue
            streams.append({'slot':slot,'busy':now['busy'],'prefill_tps':rate(computed,seconds),'decode_tps':rate(decoded,seconds),
                            'prompt':now['computed']+now['cached'],'cached':now['cached'],'decoded':now['decoded']})
        out.update(running=sum(s['busy'] for s in last.values()),slots=len(last),streams=streams,
                   prefill_tps=rate(sum(c for c,_ in done.values()),seconds),decode_tps=rate(sum(d for _,d in done.values()),seconds))
        return out
