"""Authenticated HTTP facade over the host's shared durable GPU queue."""
import asyncio
import base64
import hmac
import json
import os
from pathlib import Path
import uuid

from starlette.responses import JSONResponse, Response, StreamingResponse


class TokenGate:
    def __init__(self, app):self.app=app
    async def __call__(self, scope, receive, send):
        if scope['type']=='http':
            headers=dict(scope.get('headers',[]))
            token=Path(os.getenv('GPU_TOKEN_FILE','/state/gpu-api.token')).read_text().strip()
            trusted=set(os.getenv('GPU_TRUSTED_CLIENTS','').split(','))
            internal=scope.get('client',('',0))[0] in trusted and scope['path'].startswith(('/engines/','/v1/'))
            if not internal and not (token and hmac.compare_digest(headers.get(b'authorization',b''),('Bearer '+token).encode())):
                return await JSONResponse({'error':'unauthorized'},status_code=401)(scope,receive,send)
        await self.app(scope,receive,send)


def register(mcp, call, state):
    token_path=Path(os.environ.get('GPU_TOKEN_FILE','/state/gpu-api.token'))

    def authorized(request):
        token=token_path.read_text().strip()
        return bool(token) and hmac.compare_digest(request.headers.get('authorization',''), 'Bearer '+token)

    @mcp.custom_route('/rpc',methods=['POST'])
    async def rpc(request):
        if not authorized(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        data=await request.json()
        try:return JSONResponse({'result':await call(data['action'],**data.get('arguments',{}))})
        except (ValueError,KeyError,TypeError) as e:return JSONResponse({'error':str(e)},status_code=400)

    def allowed(request):
        return request.client.host in set(os.getenv('GPU_TRUSTED_CLIENTS','').split(',')) or authorized(request)

    def plain_error(message,status):return JSONResponse({'error':message},status_code=status)
    def openai_error(message,status):return JSONResponse({'error':{'message':message,'type':'gpu_runtime','code':status}},status_code=status)

    async def proxy(request,body,error,**target):
        """Queue one engine call (`target`: engine or model, and path) and stream its response back
        as the engine writes it."""
        try:
            job=await call('engine_submit',**target,method=request.method,body=base64.b64encode(body).decode(),
                content_type=request.headers.get('content-type','application/json'),
                request_key=request.headers.get('x-request-id') or 'http-'+uuid.uuid4().hex)
        except ValueError as e:return error(str(e),400)
        poll=job.get('request',{}).get('poll')
        if poll:return JSONResponse({'id':job['id'],poll['field']:poll['prefix']+job['id']})   # see the GET in engine()
        job_id=job['id'];directory=state/'jobs'/job_id
        meta=directory/'response.json'
        while not meta.exists():
            job=await call('job_wait',job_id=job_id,timeout=1,brief=True)
            if job['status'] in ('failed','cancelled','interrupted'):return error(job.get('error') or job['status'],503)
        headers=json.loads(meta.read_text())

        async def content():
            offset=0
            while True:
                path=directory/'response.bin'
                if path.exists():
                    with path.open('rb') as f:
                        f.seek(offset);chunk=f.read(65536)
                    if chunk:offset+=len(chunk);yield chunk;continue
                job=await call('job_get',job_id=job_id,brief=True)
                if job['status'] in ('completed','failed','cancelled','interrupted'):
                    # One final read closes the race with the worker's last write.
                    if path.exists():
                        with path.open('rb') as f:f.seek(offset);chunk=f.read()
                        if chunk:yield chunk
                    break
                await asyncio.sleep(.05)
        # X-GPU-Device: the device that ran the call (its backend in /v1/models), to know which outputs compare.
        # X-GPU-Backend: the very process that answered, as the runtime checked it (GET /v1/backends/<id>).
        extra={'x-gpu-'+key:headers[key] for key in ('device','backend') if headers.get(key)}
        return StreamingResponse(content(),status_code=headers['status'],headers={'content-type':headers['content_type'],**extra})

    @mcp.custom_route('/engines/{engine}/{path:path}',methods=['GET','POST'])
    async def engine(request):
        if not allowed(request):return plain_error('unauthorized',401)
        if request.method=='GET' and '/' in request.path_params['path']:
            # A job a submission was answered with (gpu-config `poll`): its prefix, then its id.
            ident=request.path_params['path'].rsplit('/',1)[1]
            try:job=await call('job_get',job_id=ident)
            except ValueError:job=None
            poll=job and job['kind']=='engine' and job['request'].get('poll')
            if poll and job['request']['engine']==request.path_params['engine'] and '/'+request.path_params['path']==poll['prefix']+ident:
                if job['status']=='completed':return JSONResponse(json.loads((state/'jobs'/ident/'response.bin').read_text()))
                return JSONResponse({'id':ident,'status':job['status'],'error':job.get('error')})
        body=await request.body()
        if len(body)>20*1024*1024:return plain_error('request too large',413)
        return await proxy(request,body,plain_error,engine=request.path_params['engine'],path='/'+request.path_params['path'])

    # OpenAI-style: the caller names a model, never a device or a tenant. Whichever server tenant
    # declares the model in gpu-config (`models`) answers; such calls wait their turn in creation
    # order with other jobs, where calls by tenant name (/engines/*) go first.
    @mcp.custom_route('/v1/models',methods=['GET'])
    async def models(request):
        if not allowed(request):return openai_error('unauthorized',401)
        return JSONResponse({'object':'list','data':[{'id':m['id'],'object':'model','owned_by':'gpu-runtime','max_parallel':m['max_parallel'],
                                                       'devices':m['devices'],'backends':m.get('backends',[])} for m in await call('models')]})

    @mcp.custom_route('/v1/backends/{ident}',methods=['GET'])
    async def backend(request):
        """The process an X-GPU-Backend header names: executable, command line, environment, GPU
        libraries and weights with their sha256, as the runtime that started it checked them."""
        if not allowed(request):return openai_error('unauthorized',401)
        try:return JSONResponse(await call('backend',id=request.path_params['ident']))
        except ValueError as e:return openai_error(str(e),404)

    @mcp.custom_route('/v1/{path:path}',methods=['POST'])
    async def by_model(request):
        if not allowed(request):return openai_error('unauthorized',401)
        body=await request.body()
        if len(body)>20*1024*1024:return openai_error('request too large',413)
        try:parsed=json.loads(body);model=parsed.get('model')
        except (ValueError,AttributeError):model=None
        if not isinstance(model,str) or not model:return openai_error('a JSON body naming a "model" is required',400)
        target={}
        if 'devices' in parsed:
            # Ours, not the server's: the devices this call may run on (e.g. only gpu1, for output
            # identical to earlier calls); taken out of the body the server gets.
            target['devices']=parsed.pop('devices');body=json.dumps(parsed).encode()
        return await proxy(request,body,openai_error,model=model,path='/v1/'+request.path_params['path'],**target)
