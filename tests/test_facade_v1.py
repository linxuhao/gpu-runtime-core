"""The facade's OpenAI-style entry: /v1/models lists what services answer to, a /v1 call is
queued by its body's model, and its response streams back as the engine wrote it."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
except ImportError:Starlette=None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'facade'))


BACKENDS=[{'device':'gpu1','card':'7900 XTX','api':'vulkan','framework':'llama.cpp','build':'llama-b10679','weights':'q4.gguf'}]


@unittest.skipIf(Starlette is None,'starlette is not installed here (the facade image has it)')
class FacadeV1Tests(unittest.TestCase):
    def setUp(self):
        import engine_routes
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name)
        (self.state/'token').write_text('secret');self.submitted=[]
        self.env=patch.dict(os.environ,{'GPU_TOKEN_FILE':str(self.state/'token'),'GPU_TRUSTED_CLIENTS':''});self.env.start()
        async def call(action,**args):
            if action=='models':return [{'id':'qwen3','max_parallel':1,'devices':['gpu1','gpu0'],'backends':BACKENDS},{'id':'qwen3-fp8','max_parallel':16,'devices':['gpu0']}]
            if action=='engine_submit':
                if args['model']=='nope':raise ValueError("unknown model 'nope'")
                self.submitted.append(args);directory=self.state/'jobs'/'edit-1';directory.mkdir(parents=True)
                (directory/'response.json').write_text(json.dumps({'status':200,'content_type':'text/event-stream','device':'gpu1','backend':'0123abcd-42-7'}))
                (directory/'response.bin').write_bytes(b'data: one\n\ndata: [DONE]\n\n')
                return {'id':'edit-1','status':'queued'}
            if action=='backend':
                if args['id']!='0123abcd-42-7':raise ValueError('unknown backend %r'%args['id'])
                return {'id':args['id'],'status':'checked','alive':True}
            if action in ('job_get','job_wait'):
                self.assertTrue(args.get('brief'),'the streaming poll never pulls the request body')
                return {'id':args['job_id'],'status':'completed'}
            raise AssertionError(action)
        routes=[]
        class Collect:
            def custom_route(self,path,methods):
                def wrap(fn):routes.append(Route(path,fn,methods=methods));return fn
                return wrap
        engine_routes.register(Collect(),call,self.state)
        self.client=TestClient(engine_routes.TokenGate(Starlette(routes=routes)))
        self.auth={'Authorization':'Bearer secret'}
    def tearDown(self):self.env.stop();self.temp.cleanup()

    def test_models_are_listed_the_openai_way(self):
        response=self.client.get('/v1/models',headers=self.auth)
        self.assertEqual(response.json(),{'object':'list','data':[
            {'id':'qwen3','object':'model','owned_by':'gpu-runtime','max_parallel':1,'devices':['gpu1','gpu0'],'backends':BACKENDS},
            {'id':'qwen3-fp8','object':'model','owned_by':'gpu-runtime','max_parallel':16,'devices':['gpu0'],'backends':[]}]})
        self.assertEqual(self.client.get('/v1/models').status_code,401)

    def test_a_call_is_queued_by_its_model_and_streamed_back(self):
        body={'model':'qwen3','messages':[{'role':'user','content':'hi'}],'stream':True}
        response=self.client.post('/v1/chat/completions',headers={**self.auth,'X-Request-Id':'k1'},json=body)
        self.assertEqual(response.status_code,200);self.assertEqual(response.content,b'data: one\n\ndata: [DONE]\n\n')
        self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
        self.assertEqual(response.headers['x-gpu-device'],'gpu1','which backend of /v1/models answered')
        self.assertEqual(response.headers['x-gpu-backend'],'0123abcd-42-7','which process')
        self.assertEqual(self.client.get('/v1/backends/0123abcd-42-7',headers=self.auth).json(),{'id':'0123abcd-42-7','status':'checked','alive':True})
        self.assertEqual(self.client.get('/v1/backends/0123abcd-1-1',headers=self.auth).status_code,404)
        self.assertEqual(self.client.get('/v1/backends/0123abcd-42-7').status_code,401)
        sent=self.submitted[0]
        self.assertEqual((sent['model'],sent['path'],sent['method'],sent['request_key']),('qwen3','/v1/chat/completions','POST','k1'))
        self.assertEqual(json.loads(base64.b64decode(sent['body'])),body);self.assertNotIn('engine',sent)

    def test_devices_go_to_the_runtime_not_to_the_server(self):
        body={'model':'qwen3','messages':[{'role':'user','content':'hi'}],'devices':['gpu1']}
        self.assertEqual(self.client.post('/v1/chat/completions',headers=self.auth,json=body).status_code,200)
        sent=self.submitted[0];self.assertEqual(sent['devices'],['gpu1'])
        self.assertEqual(json.loads(base64.b64decode(sent['body'])),{'model':'qwen3','messages':[{'role':'user','content':'hi'}]})

    def test_refusals_come_back_as_openai_errors(self):
        response=self.client.post('/v1/chat/completions',headers=self.auth,json={'model':'nope'})
        self.assertEqual((response.status_code,response.json()['error']['message']),(400,"unknown model 'nope'"))
        response=self.client.post('/v1/chat/completions',headers=self.auth,content=b'not json')
        self.assertEqual(response.status_code,400);self.assertIn('model',response.json()['error']['message'])
        self.assertEqual(self.client.post('/v1/chat/completions',json={'model':'qwen3'}).status_code,401)


@unittest.skipIf(Starlette is None,'starlette is not installed here (the facade image has it)')
class PollTests(unittest.TestCase):
    """A tenant declaring `poll` is answered at once with the job to poll, then polled by its id."""
    def setUp(self):
        import engine_routes
        self.temp=tempfile.TemporaryDirectory();self.state=Path(self.temp.name);(self.state/'token').write_text('secret')
        self.env=patch.dict(os.environ,{'GPU_TOKEN_FILE':str(self.state/'token'),'GPU_TRUSTED_CLIENTS':''});self.env.start()
        poll={'field':'poll_url','prefix':'/img/v1/jobs/'};self.status='running';self.submitted=[]
        async def call(action,**args):
            if action=='engine_submit':
                if args['engine']!='img':raise ValueError('unsupported engine route')
                self.submitted.append(args)
                return {'id':'edit-7','kind':'engine','status':'queued','request':{'engine':args['engine'],'path':args['path'],**({'poll':poll} if args['path']=='/img/v1/gen' else {})}}
            if action=='job_get':
                if args['job_id']!='edit-7':raise ValueError('unknown jobs id')
                return {'id':'edit-7','kind':'engine','status':self.status,'error':None,'request':{'engine':'img','poll':poll}}
            raise AssertionError(action)
        routes=[]
        class Collect:
            def custom_route(self,path,methods):
                def wrap(fn):routes.append(Route(path,fn,methods=methods));return fn
                return wrap
        engine_routes.register(Collect(),call,self.state)
        self.client=TestClient(engine_routes.TokenGate(Starlette(routes=routes)));self.auth={'Authorization':'Bearer secret'}
    def tearDown(self):self.env.stop();self.temp.cleanup()

    def test_submission_then_poll(self):
        self.assertEqual(self.client.post('/engines/img/img/v1/gen',headers=self.auth,json={}).json(),{'id':'edit-7','poll_url':'/img/v1/jobs/edit-7'})
        self.assertEqual(self.client.get('/engines/img/img/v1/jobs/edit-7',headers=self.auth).json(),{'id':'edit-7','status':'running','error':None})
        directory=self.state/'jobs'/'edit-7';directory.mkdir(parents=True);(directory/'response.bin').write_text('{"result": "png"}')
        self.status='completed'
        self.assertEqual(self.client.get('/engines/img/img/v1/jobs/edit-7',headers=self.auth).json(),{'result':'png'})
        self.assertEqual(self.client.get('/engines/other/img/v1/jobs/edit-7',headers=self.auth).status_code,400,
                         "another tenant's path is not this job: an engine call, refused as a route it does not have")

if __name__=='__main__':unittest.main()
