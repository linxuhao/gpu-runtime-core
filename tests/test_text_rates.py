"""text_rates turns the text servers' counters into prefill and decode speeds: llama-server's own
timings with each slot's progress, in all and per request on average for vLLM."""
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


LLAMA_METRICS='''# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total %(prompt)s
llamacpp:prompt_seconds_total %(prompt_s)s
llamacpp:tokens_predicted_total %(generated)s
llamacpp:tokens_predicted_seconds_total %(generated_s)s
llamacpp:spec_decode_num_draft_tokens_total %(drafts)s
llamacpp:spec_decode_num_accepted_tokens_total %(accepted)s
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 99
llamacpp:requests_processing 1
'''


def llama(slots, **counters):
    """A llama-server reading: its slots, and its /metrics page (None: started without --metrics)."""
    if counters.pop('off',False):return {'slots':text_rates.parse_llama(slots),'metrics':None}
    base=dict(prompt=0,prompt_s=0,generated=0,generated_s=0,drafts=0,accepted=0);base.update(counters)
    return {'slots':text_rates.parse_llama(slots),'metrics':text_rates.parse_metrics(LLAMA_METRICS%base,text_rates.LLAMA)}


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

    def test_llama_speeds_are_the_engines_own_timings(self):
        rates=TextRates()
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,7,1000,50,cached=4000),slot(1,3,500,20,busy=False)]),0)
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,7,1000,450,cached=4000),slot(1,3,500,20,busy=False)],
                                                        prompt=2000,prompt_s=4,generated=600,generated_s=10,drafts=200,accepted=180),5)
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,8,2000,100),slot(1,3,500,20,busy=False)],
                                                        prompt=3800,prompt_s=7,generated=600,generated_s=10,drafts=200,accepted=180),10)
        [view]=rates.lane('gpu1')
        self.assertEqual((view['prefill_tps'],view['decode_tps'],view['accept']),(542.9,60.0,0.9),
                         'tokens over the engine\'s seconds for each phase, not over the window')
        self.assertEqual((view['running'],view['slots'],view['metrics'],view['window_s']),(1,2,True,10))
        self.assertNotIn('decode_age_s',view)
        self.assertEqual(view['streams'],[{'slot':0,'busy':True,'prompt':2000,'cached':0,'decoded':100}],'an idle slot that did nothing is left out')

    def test_llama_decode_counted_at_the_end_shows_the_last_request_meanwhile(self):
        rates=TextRates()
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,7,1000,10)],prompt=1000,prompt_s=2),0)
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,8,500,5)],prompt=1500,prompt_s=3,generated=570,generated_s=10),2)
        for second in range(4,41,2):
            rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,8,500,5+second*50)],prompt=1500,prompt_s=3,generated=570,generated_s=10),second)
        [view]=rates.lane('gpu1')
        self.assertEqual((view['decode_tps'],view['decode_age_s']),(57.0,38),'task 7 ended 38 s ago; task 8 is still decoding')
        self.assertEqual((view['prefill_tps'],view['prefill_age_s']),(500.0,38))

    def test_llama_without_metrics_says_so_and_still_shows_its_slots(self):
        rates=TextRates()
        for second in (0,2):rates.record('translator','gpu1','llama.cpp',llama([slot(0,1,10,second)],off=True),second)
        [view]=rates.lane('gpu1')
        self.assertFalse(view['metrics']);self.assertNotIn('decode_tps',view);self.assertEqual(view['streams'][0]['decoded'],2)

    def test_a_restarted_llama_server_starts_its_window_over(self):
        rates=TextRates()
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,1,10,10)],generated=9000,generated_s=100),0)
        rates.record('qwen3-q4','gpu1','llama.cpp',llama([slot(0,1,10,10)],generated=100,generated_s=2),2)
        self.assertEqual(rates.lane('gpu1')[0]['window_s'],0)

    def test_the_window_keeps_the_last_seconds_and_stopped_servers_go(self):
        rates=TextRates()
        for second in range(0,61,2):rates.record('qwen3-fp8','gpu0','vllm',vllm(generated=10*second),second)
        view=rates.lane('gpu0')[0]
        self.assertTrue(text_rates.WINDOW_S<=view['window_s']<=text_rates.WINDOW_S+text_rates.SAMPLE_S,view['window_s'])
        self.assertEqual(view['decode_tps'],10.0)
        rates.keep([]);self.assertEqual(rates.lane('gpu0'),[])

    def test_a_server_moved_to_another_device_starts_over(self):
        rates=TextRates()
        rates.record('translator','gpu1','llama.cpp',llama([slot(0,1,10,10)]),0)
        rates.record('translator','gpu0','llama.cpp',llama([slot(0,1,10,10)]),2)
        self.assertEqual(rates.lane('gpu1'),[]);self.assertEqual(rates.lane('gpu0')[0]['window_s'],0)


if __name__=='__main__':unittest.main()
