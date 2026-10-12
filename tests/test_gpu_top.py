"""gpu_top draws the runtime's queues and lends next to the cards' readings."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import gpu_top

NOW=1_800_000_000
CARD={'model':'NVIDIA RTX 4090','util':91.0,'used_mib':20480.0,'total_mib':65536.0,'power_w':126.0,'cap_w':220.0,'temp_c':66.0,'fan':None,
      'processes':{4242:20000.0,4343:10.0}}
STATE={'mode':'loaded','draining':False,'queued':3,'workers':{'engines':{'translator':4242},'batches':{},'parts':{'pair.back':4343}},
       'lanes':{'gpu0':{'active':'take-a','step':'sample','queued':3,'blocked':None,
                        'lent':{'job_id':'lend-x','device':'gpu0','owner':'lab-7','mode':'queued','granted':NOW-60,'deadline':NOW+3600},
                        'device':{'display':'RTX 4090','vendor':'nvidia','uuid':'GPU-1','pci':None},
                        'running':{'id':'take-a','kind':'render','status':'running','step':'sample','project':'p','shot':'s1','created':NOW-125},
                        'waiting':[{'id':'eng-1','kind':'engine','status':'queued','step':'engine','engine':'audio','created':NOW-5},
                                   {'id':'lend-y','kind':'lend','status':'queued','step':'lend','owner':'lab-8','mode':'exclusive','created':NOW-30}]}}}


class RenderTests(unittest.TestCase):
    def draw(self, state=STATE, error=None):
        devices=gpu_top.devices_of(state,'/nonexistent/runtime.sock')
        return gpu_top.render(state,error,devices,{'GPU-1':CARD},{},100,gpu_top.Paint(False),now=NOW)

    def test_a_card_shows_its_readings_lend_job_and_queue_in_order(self):
        text=self.draw()
        self.assertIn('lent queued to lab-7',text)
        self.assertIn(' 91%',text);self.assertIn('126/220 W',text);self.assertIn('20.0/64.0 GiB',text)
        run=next(l for l in text.splitlines() if l.lstrip().startswith('run'))
        self.assertIn('take-a',run);self.assertIn('render/sample',run);self.assertIn('p/s1',run);self.assertIn('2m ago',run)
        queue=[l for l in text.splitlines() if ' 1. ' in l or ' 2. ' in l]
        self.assertIn('eng-1',queue[0]);self.assertIn('audio',queue[0])
        self.assertIn('lend-y',queue[1]);self.assertIn('lend (exclusive)',queue[1]);self.assertIn('lab-8',queue[1])
        self.assertIn('... 1 more',text,'queued counts more than the listed ones')
        self.assertIn('3 waiting',text.splitlines()[0],'what the queues hold, renders between stages included')

    def test_processes_are_named_by_the_runtime_and_contexts_are_left_out(self):
        text=self.draw()
        self.assertIn('translator',text);self.assertNotIn('4343',text)

    def test_an_unreachable_runtime_still_shows_the_cards(self):
        state=None;devices={'gpu0':{'display':'RTX 4090','vendor':'nvidia','uuid':'GPU-1'}}
        text=gpu_top.render(state,'connection refused',devices,{'GPU-1':CARD},{},100,gpu_top.Paint(False),now=NOW)
        self.assertIn('runtime unreachable: connection refused',text);self.assertIn(' 91%',text)
        self.assertNotIn('queue',text)

    def test_text_servers_show_prefill_and_decode_in_all_and_per_stream(self):
        lane={**STATE['lanes']['gpu0'],'text':[
            {'tenant':'qwen3-fp8','kind':'vllm','window_s':30,'running':16,'waiting':0,'kv':0.64,'prefill_tps':1010.4,'decode_tps':275.8,
             'cache_hit':0.61,'accept':0.84,'stream':{'prefill_tps':1350.0,'decode_tps':18.8,'first_token_s':1.07}},
            {'tenant':'qwen3-q4','kind':'llama.cpp','window_s':30,'running':1,'slots':1,'prefill_tps':0.0,'decode_tps':36.6,
             'streams':[{'slot':0,'busy':True,'prefill_tps':0.0,'decode_tps':36.6,'prompt':41126,'cached':41101,'decoded':9997}]},
            {'tenant':'translator','kind':'llama.cpp','window_s':0}]}
        text=self.draw({**STATE,'lanes':{'gpu0':lane}});lines=text.splitlines()
        everything=next(l for l in lines if 'qwen3-fp8' in l and 'all' in l)
        for part in ('prefill  1010 tok/s','decode   276 tok/s','16 running','cache hits 61%','KV 64%','MTP accepted 84%','last 30s'):self.assertIn(part,everything)
        stream=lines[lines.index(everything)+1]
        for part in ('stream','prefill  1350 tok/s','decode    19 tok/s','first token 1.1 s','mean per request'):self.assertIn(part,stream)
        slot=next(l for l in lines if 'slot 0' in l)
        for part in ('decode    37 tok/s','prompt 41126 (41101 cached), 9997 decoded'):self.assertIn(part,slot)
        self.assertIn('1/1 slots busy',text);self.assertIn('measuring...',next(l for l in lines if 'translator' in l and 'text' in l))

    def test_the_runtimes_warnings_come_first(self):
        text=self.draw({**STATE,'warnings':['Vulkan userspace differs from the host: mesa-vulkan-drivers: host .5, image .4']})
        self.assertEqual(text.splitlines()[1],'! Vulkan userspace differs from the host: mesa-vulkan-drivers: host .5, image .4')

    def test_bars_fill_by_eighths_and_clip(self):
        paint=gpu_top.Paint(False)
        self.assertEqual(gpu_top.bar(.5,4,paint),'██░░');self.assertEqual(gpu_top.bar(1.7,3,paint),'███')
        self.assertEqual(gpu_top.bar(1/16,2,paint),'▏░');self.assertEqual(gpu_top.bar(None,3,paint),'···')

    def test_a_long_queue_fits_the_screen_and_says_what_it_left_out(self):
        waiting=[{'id':'take-%02d'%i,'kind':'render','status':'running','step':'sample','project':'proj-a' if i<40 else 'proj-b','shot':'s%d'%i,'created':NOW-600}
                 for i in range(45)]
        lane={**STATE['lanes']['gpu0'],'queued':47,'waiting':waiting}
        state={**STATE,'lanes':{'gpu0':lane}};devices=gpu_top.devices_of(state,'/nonexistent/runtime.sock')
        full=gpu_top.render(state,None,devices,{'GPU-1':CARD},{},100,gpu_top.Paint(False),now=NOW)
        self.assertEqual(sum(' take-' in l for l in full.splitlines()),46,'piped: everything the runtime sent')
        text=gpu_top.render(state,None,devices,{'GPU-1':CARD},{},60,gpu_top.Paint(True),now=NOW,height=24)
        lines=text.splitlines()
        self.assertLessEqual(len(lines),24);self.assertTrue(lines[0].startswith('\033[1mGPU runtime'),'the header stays on screen')
        self.assertTrue(any(' GPU ' in l for l in lines) and any('VRAM' in l for l in lines))
        self.assertTrue(all(len(re.sub('\033\\[[0-9;]*m','',l))<=60 for l in lines),'no line wraps')
        listed=[l for l in lines if '. take-' in l]
        self.assertGreaterEqual(len(listed),gpu_top.MIN_QUEUE)
        more=next(l for l in lines if '... ' in l)
        self.assertIn('%d more'%(47-len(listed)),more);self.assertIn('render/sample proj-a',more)

    def test_entry_numbers_line_up(self):
        waiting=[{'id':'call-%02d'%i,'kind':'engine','status':'queued','step':'engine','engine':'audio','created':NOW-5} for i in range(12)]
        text=self.draw({**STATE,'lanes':{'gpu0':{**STATE['lanes']['gpu0'],'queued':12,'waiting':waiting}}})
        starts={l.index('call-') for l in text.splitlines() if 'call-' in l}
        self.assertEqual(len(starts),1,'one column whatever the number of digits')

    def test_clip_counts_what_shows_not_colour_codes(self):
        self.assertEqual(gpu_top.clip('\033[1mabcdef\033[0m',3),'\033[1mabc\033[0m')
        self.assertEqual(gpu_top.clip('abc',3),'abc');self.assertEqual(gpu_top.clip('abcdef',4),'abcd')

if __name__=='__main__':unittest.main()
