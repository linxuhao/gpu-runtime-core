"""The GPU layer's HTTP facade. No MCP here.

A client (an MCP server, an agent's tools, a script) calls this
over HTTP with a bearer token. This process only adds what must sit beside the runtime:
the unix-socket hop to it (/rpc), the engine proxy the other GPU-machine containers use
(/engines/*), and serving generated files from the state directory (/files/*).
"""
import argparse
import os
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from engine_routes import TokenGate, register

STATE=Path(os.getenv('GPU_STATE','/state'))

async def call(action,**arguments):
    transport=httpx.AsyncHTTPTransport(uds=str(STATE/'runtime.sock'))
    async with httpx.AsyncClient(transport=transport,timeout=930) as client:
        response=await client.post('http://runtime/rpc',json={'action':action,'arguments':arguments})
    try:value=response.json()
    except ValueError:
        # Most often the runtime being down: a plain-text 500 with no error envelope.
        raise ValueError('backend returned HTTP %d with a non-JSON body: %s'
                         %(response.status_code,response.text[:200].strip() or '(empty)'))
    if response.status_code!=200:raise ValueError(value.get('error','backend request failed'))
    return value['result']

class Routes:
    """The decorator shape engine_routes.register() expects, collecting Starlette routes."""
    def __init__(self):self.routes=[]
    def custom_route(self,path,methods):
        def wrap(fn):self.routes.append(Route(path,fn,methods=methods));return fn
        return wrap

routes=Routes()

@routes.custom_route('/files/{path:path}',methods=['GET'])
async def files(request:Request):
    path=(STATE/request.path_params['path']).resolve()
    # Only generated media and reports are downloadable, never DB/socket/credentials.
    if not path.is_relative_to((STATE/'jobs').resolve()) or path.suffix not in ('.mp4','.png','.wav','.json') or not path.is_file():
        return JSONResponse({'error':'not found'},status_code=404)
    return FileResponse(path)

@routes.custom_route('/health',methods=['GET'])
async def health(request:Request):
    try:return JSONResponse(await call('resource_status'))
    except Exception as e:return JSONResponse({'backend':'unavailable','error':str(e)},status_code=503)

register(routes,call,STATE)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=9041);args=p.parse_args()
    uvicorn.run(TokenGate(Starlette(routes=routes.routes)),host='0.0.0.0',port=args.port)
