"""The scheduler and the job service on their own: a host whose config declares no plugin tenant."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from gpu_runtime import GIB, Device, Scheduler, load_config
import job_service
import harness as base

LLM={'names':['qwen'],'backends':[{'device':'gpu0','api':'cuda','command':['llama-server'],'url':'http://127.0.0.1:18000','vram_gib':20}]}
CONFIG={'schema':2,'devices':{'gpu0':{'vendor':'nvidia','uuid':'GPU-1','placement':'lru'}},'tenants':{},
        'services':{'llm':LLM},'policy':{'idle':{'restore':['llm']}}}


class NoPluginTests(base.SchedulerHarness):
    def setUp(self):
        super().setUp();(self.state/'gpu-config.json').write_text(json.dumps(CONFIG))

    def test_the_scheduler_needs_no_plugin(self):
        s=Scheduler(self.state.parent,self.state)
        self.assertEqual(s.plugins,{})
        self.assertEqual(s.stages('engine',{'engine':'llm'}),[('engine','gpu0')])
        with self.assertRaisesRegex(ValueError,'unknown job kind'):s.run_stage('render','encode',{},self.state/'x')
        with patch.object(Device,'vram',lambda device:(64*GIB,0)):s.park()
        self.assertEqual(self.occupants(s),{'gpu0':['llm']})
        self.assertEqual(set(s.status()),{'devices','engines','parts','engine_devices','batches','in_use','policy','park_error'})

    def test_the_job_service_serves_it_by_model(self):
        service=job_service.Service(self.state.parent,self.state,start_worker=False)
        try:
            self.assertEqual([m['id'] for m in service.call('models',{})],['qwen'])
            job=service.call('engine_submit',{'model':'qwen','path':'/v1/chat/completions','method':'POST','body':'','content_type':'application/json','request_key':'k'})
            self.assertEqual((job['kind'],job['lane'],job['request']['engine']),('engine','gpu0','llm'))
            with self.assertRaisesRegex(ValueError,'unknown action'):service.call('render',{})
            with self.assertRaisesRegex(ValueError,'no batch tenant'):service.call('batch_submit',{'request':{'tenant':'llm'},'request_key':'b'})
            self.assertEqual(service.call('resource_status',{})['lanes']['gpu0']['queued'],1)
        finally:service.close()

    def test_an_unknown_kind_names_the_module_it_looked_for(self):
        cfg=json.loads(json.dumps(CONFIG));cfg['tenants']['x']={'kind':'nosuch','device':'gpu0','api':'cuda'}
        path=self.state/'gpu-config.json';path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'nosuch_tenant'):load_config(path)


if __name__=='__main__':unittest.main()
