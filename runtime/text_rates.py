"""Prefill and decode speeds of the text servers (llama-server, vLLM), for gpu-top.

The service reads every running one each SAMPLE_S seconds and keeps WINDOW_S seconds of readings;
speeds are what happened in that window. Prefill counts only the prompt tokens computed, not those
served from the prefix cache.

- llama-server: /metrics (needs `--metrics`), the engine's own prompt and generation times, the
  same as its log's timings: prefill in all, decode per stream. They count a request's prompt as it
  is computed but its generation only when it ends, so a window without one shows the last finished
  request's speed and how long ago that was. /slots shows each request's progress.
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
import urllib.error
import urllib.request

SAMPLE_S=2
WINDOW_S=30
VLLM={'prompt':'vllm:prompt_tokens_total','hits':'vllm:prefix_cache_hits_total','queries':'vllm:prefix_cache_queries_total',
      'generated':'vllm:generation_tokens_total','itl_s':'vllm:inter_token_latency_seconds_sum',
      'ttft_s':'vllm:time_to_first_token_seconds_sum','ttft_n':'vllm:time_to_first_token_seconds_count',
      'drafts':'vllm:spec_decode_num_draft_tokens_total','accepted':'vllm:spec_decode_num_accepted_tokens_total',
      'running':'vllm:num_requests_running','waiting':'vllm:num_requests_waiting','kv':'vllm:kv_cache_usage_perc'}
GAUGES=('running','waiting','kv')
LLAMA={'prompt':'llamacpp:prompt_tokens_total','prompt_s':'llamacpp:prompt_seconds_total',
       'generated':'llamacpp:tokens_predicted_total','generated_s':'llamacpp:tokens_predicted_seconds_total',
       'drafts':'llamacpp:spec_decode_num_draft_tokens_total','accepted':'llamacpp:spec_decode_num_accepted_tokens_total'}


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


def parse_metrics(text, table):
    """The counters of `table` (VLLM or LLAMA) from a /metrics page, each summed over its label sets."""
    found={};names={v:k for k,v in table.items()}
    for line in text.splitlines():
        match=re.match(r'([a-zA-Z_:]+)(\{[^}]*\})?\s+(\S+)$',line)
        if match and match.group(1) in names:
            key=names[match.group(1)];found[key]=found.get(key,0.0)+float(match.group(3))
    return found


def parse_vllm(text):return parse_metrics(text,VLLM)


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
    with urllib.request.urlopen(url+'/slots',timeout=timeout) as response:slots=parse_llama(json.loads(response.read()))
    try:
        with urllib.request.urlopen(url+'/metrics',timeout=timeout) as response:metrics=parse_metrics(response.read().decode(),LLAMA)
    except urllib.error.HTTPError:metrics=None   # started without --metrics
    return {'slots':slots,'metrics':metrics}


def rate(amount, seconds):return round(amount/seconds,1) if seconds>0 else None


class TextRates:
    def __init__(self):
        self.lock=threading.Lock();self.samples={}   # tenant -> {device, kind, readings: deque of (time, reading), last}

    def record(self, tenant, device, kind, reading, now):
        with self.lock:
            entry=self.samples.get(tenant)
            if not entry or (entry['device'],entry['kind'])!=(device,kind):
                entry=self.samples[tenant]={'device':device,'kind':kind,'readings':deque(),'last':{}}
            readings=entry['readings'];counters=reading if kind=='vllm' else reading['metrics']
            before=readings[-1][1] if readings else None
            if before is not None and kind!='vllm':before=before['metrics']
            if counters and before and counters.get('generated',0)<before.get('generated',0):
                readings.clear();entry['last'].clear();before=None   # a new server: its counters started over
            if kind=='llama.cpp' and counters and before:
                # what finished between the two readings: kept for a window in which nothing does
                for phase,tokens,seconds in (('prefill','prompt','prompt_s'),('decode','generated','generated_s')):
                    took=counters.get(seconds,0)-before.get(seconds,0)
                    if took>0:entry['last'][phase]=((counters.get(tokens,0)-before.get(tokens,0))/took,now)
                drafts=counters.get('drafts',0)-before.get('drafts',0)
                if drafts>0:entry['last']['accept']=((counters.get('accepted',0)-before.get('accepted',0))/drafts,now)
            readings.append((now,reading))
            while len(readings)>2 and now-readings[1][0]>=WINDOW_S:readings.popleft()

    def keep(self, tenants):
        """Forget the servers no longer running."""
        with self.lock:
            for tenant in set(self.samples)-set(tenants):del self.samples[tenant]

    def lane(self, device):
        """The speeds of the text servers on a device, for resource_status."""
        with self.lock:
            found=[(tenant,entry['kind'],list(entry['readings']),dict(entry['last'])) for tenant,entry in sorted(self.samples.items())
                   if entry['device']==device]
        return [view for view in (self.view(*item) for item in found) if view]

    @staticmethod
    def view(tenant, kind, readings, last=None):
        if len(readings)<2:return {'tenant':tenant,'kind':kind,'window_s':0}
        (start,first),(end,final)=readings[0],readings[-1];seconds=end-start
        out={'tenant':tenant,'kind':kind,'window_s':round(seconds)}
        if kind=='vllm':
            delta={k:final.get(k,0)-first.get(k,0) for k in VLLM if k not in GAUGES}
            cached=delta['hits']/delta['queries'] if delta['queries']>0 else 0
            computed=delta['prompt']*(1-cached)
            out.update(running=int(final.get('running',0)),waiting=int(final.get('waiting',0)),kv=round(final['kv'],2) if 'kv' in final else None,
                       prefill_tps=rate(computed,seconds),decode_tps=rate(delta['generated'],seconds),cache_hit=round(cached,2),
                       stream={'prefill_tps':rate(computed,delta['ttft_s']),'decode_tps':rate(delta['generated'],delta['itl_s']),
                               'first_token_s':round(delta['ttft_s']/delta['ttft_n'],2) if delta['ttft_n'] else None},
                       accept=round(delta['accepted']/delta['drafts'],2) if delta['drafts'] else None)
            return out
        # llama-server: each slot's request as it stands, and the engine's timings over the window.
        slots=final['slots'];was=first['slots']
        out.update(running=sum(s['busy'] for s in slots.values()),slots=len(slots),
                   streams=[{'slot':slot,'busy':now['busy'],'prompt':now['computed']+now['cached'],'cached':now['cached'],'decoded':now['decoded']}
                            for slot,now in sorted(slots.items()) if now['busy'] or now!=was.get(slot)])
        if final['metrics'] is None or first['metrics'] is None:
            out['metrics']=False;return out
        delta={k:final['metrics'].get(k,0)-first['metrics'].get(k,0) for k in LLAMA}
        out['metrics']=True;last=last or {}
        for phase,tokens,took in (('prefill','prompt','prompt_s'),('decode','generated','generated_s')):
            if delta[took]>0:out[phase+'_tps']=rate(delta[tokens],delta[took])
            elif phase in last:
                out[phase+'_tps']=round(last[phase][0],1);out[phase+'_age_s']=round(end-last[phase][1])
            else:out[phase+'_tps']=None
        out['accept']=round(delta['accepted']/delta['drafts'],2) if delta['drafts'] else round(last['accept'][0],2) if 'accept' in last else None
        return out
