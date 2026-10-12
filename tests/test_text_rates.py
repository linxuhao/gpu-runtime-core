"""text_rates turns the text servers' counters into prefill and decode speeds: per slot for
llama-server, in all and per request on average for vLLM."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import text_rates
from text_rates import TextRates

METRICS='''# HELP vllm:generation_tokens_total Number of generation tokens processed.
vllm:generation_tokens_total{engine="0",model_name="qwen3-fp8"} %(generated)s
vllm:prompt_tokens_total{engine="0",model_name="qwen3-fp8"} %(prompt)s
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3-fp8"} %(hits)s
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3-fp8"} %(queries)s
vllm:inter_token_latency_seconds_bucket{engine="0",le="0.01",model_name="qwen3-fp8"} 99
vllm:inter_token_latency_seconds_sum{engine="0",model_name="qwen3-fp8"} %(itl)s
vllm:time_to_first_token_seconds_sum{engine="0",model_name="qwen3-fp8"} %(ttft)s
vllm:time_to_first_token_seconds_count{engine="0",model_name="qwen3-fp8"} %(firsts)s
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="qwen3-fp8"} %(drafts)s
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="qwen3-fp8"} %(accepted)s
vllm:num_requests_running{engine="0",model_name="qwen3-fp8"} 16.0
vllm:num_requests_waiting{engine="0",model_name="qwen3-fp8"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-fp8"} 0.64
'''


def vllm(**counters):
    base=dict(generated=0,prompt=0,hits=0,queries=0,itl=0,ttft=0,firsts=0,drafts=0,accepted=0);base.update(counters)
    return text_rates.parse_vllm(METRICS%base)


def slot(ident, task, computed, decoded, busy=True, cached=0):
    return {'id':ident,'id_task':task,'is_processing':busy,'n_prompt_tokens_processed':computed,'n_prompt_tokens_cache':cached,
            'next_token':[{'n_decoded':decoded}]}


class TextRatesTests(unittest.TestCase):
    def test_text_servers_are_known_by_their_command(self):
        self.assertEqual(text_rates.kind_of(['/x/llama-b10679/llama-server','-m','m.gguf']),'llama.cpp')
        self.assertEqual(text_rates.kind_of(['/x/.venv-vllm/bin/python','-m','vllm.entrypoints.cli.main','serve','/m']),'vllm')
        self.assertEqual(text_rates.kind_of(['vllm','serve','/m']),'vllm')
        self.assertIsNone(text_rates.kind_of(['/x/audiocpp_server','--port','19020']))

    def test_what_a_text_server_runs_is_told_from_its_command(self):
        llama=['/x/lcpp_bare/llama-b10679-cuda/llama-server','-m','/hub/snap/Qwen3.8-27B-UD-Q4_K_M.gguf','-c','131072']
        self.assertEqual(text_rates.describe(llama),{'framework':'llama.cpp','build':'llama-b10679-cuda','weights':'Qwen3.8-27B-UD-Q4_K_M.gguf'})
        with tempfile.TemporaryDirectory() as venv:
            (Path(venv)/'lib/python3.12/site-packages/vllm-0.31.0.dist-info').mkdir(parents=True)
            vllm=[venv+'/bin/python','-m','vllm.entrypoints.cli.main','serve','/models/Qwen3.8-27B-FP8/','--port','18200']
            self.assertEqual(text_rates.describe(vllm),{'framework':'vllm','build':'vllm 0.31.0','weights':'Qwen3.8-27B-FP8'})
        self.assertEqual(text_rates.describe(['/x/audiocpp_server']),{})

    def test_vllm_in_all_and_per_request(self):
        rates=TextRates()
        rates.record('qwen3-fp8','gpu0','vllm',vllm(),0)
        rates.record('qwen3-fp8','gpu0','vllm',vllm(generated=8000,prompt=40000,hits=30000,queries=50000,itl=400,ttft=20,firsts=10,drafts=6000,accepted=4800),10)
        [view]=rates.lane('gpu0')
        self.assertEqual((view['decode_tps'],view['prefill_tps'],view['cache_hit']),(800.0,1600.0,0.6),'prefill: only the tokens not from the cache')
        self.assertEqual(view['stream'],{'decode_tps':20.0,'prefill_tps':800.0,'first_token_s':2.0})
        self.assertEqual((view['running'],view['waiting'],view['kv'],view['accept'],view['window_s']),(16,0,0.64,0.8,10))
        self.assertEqual(rates.lane('gpu1'),[])

    def test_a_restarted_vllm_starts_its_window_over(self):
        rates=TextRates()
        rates.record('qwen3-fp8','gpu0','vllm',vllm(generated=9000),0)
        rates.record('qwen3-fp8','gpu0','vllm',vllm(generated=100),5)
        self.assertEqual(rates.lane('gpu0')[0]['window_s'],0,'one reading of the new server: still measuring')

    def test_llama_slots_each_and_added_up(self):
        rates=TextRates()
        rates.record('qwen3-q4','gpu1','llama.cpp',text_rates.parse_llama([slot(0,7,1000,50,cached=4000),slot(1,3,500,20,busy=False)]),0)
        rates.record('qwen3-q4','gpu1','llama.cpp',text_rates.parse_llama([slot(0,7,1000,450,cached=4000),slot(1,3,500,20,busy=False)]),5)
        rates.record('qwen3-q4','gpu1','llama.cpp',text_rates.parse_llama([slot(0,8,2000,100),slot(1,3,500,20,busy=False)]),10)
        [view]=rates.lane('gpu1')
        self.assertEqual([s['slot'] for s in view['streams']],[0],'an idle slot that did nothing is left out')
        self.assertEqual((view['streams'][0]['decode_tps'],view['streams'][0]['prefill_tps']),(50.0,200.0),
                         '400 decoded by task 7, then task 8 started: its 2000 computed and 100 decoded count too')
        self.assertEqual((view['decode_tps'],view['prefill_tps'],view['running'],view['slots']),(50.0,200.0,1,2))
        self.assertEqual((view['streams'][0]['prompt'],view['streams'][0]['decoded']),(2000,100))

    def test_the_window_keeps_the_last_seconds_and_stopped_servers_go(self):
        rates=TextRates()
        for second in range(0,61,2):rates.record('qwen3-fp8','gpu0','vllm',vllm(generated=10*second),second)
        view=rates.lane('gpu0')[0]
        self.assertTrue(text_rates.WINDOW_S<=view['window_s']<=text_rates.WINDOW_S+text_rates.SAMPLE_S,view['window_s'])
        self.assertEqual(view['decode_tps'],10.0)
        rates.keep([]);self.assertEqual(rates.lane('gpu0'),[])

    def test_a_server_moved_to_another_device_starts_over(self):
        rates=TextRates()
        rates.record('translator','gpu1','llama.cpp',text_rates.parse_llama([slot(0,1,10,10)]),0)
        rates.record('translator','gpu0','llama.cpp',text_rates.parse_llama([slot(0,1,10,10)]),2)
        self.assertEqual(rates.lane('gpu1'),[]);self.assertEqual(rates.lane('gpu0')[0]['window_s'],0)


if __name__=='__main__':unittest.main()
