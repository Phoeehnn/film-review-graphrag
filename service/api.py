import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from redis.exceptions import RedisError

from .agent import AgentService
from .engine import Engine
from .schemas import AgentRequest, QueryRequest, QueryResult
from .settings import Settings
from .redis_layer import RedisLayer, replay_cached
from .topics import topic_config
from .version import SERVICE_VERSION, runtime_source_hashes
from .session_models import ConditionPatch, SessionRequest
from .session_store import SessionStore, SessionConflict
from .conversation import ConversationService


def create_app(engine=None, manage_lifecycle=True):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or Engine(Settings())
        app.state.agent = AgentService(app.state.engine)
        app.state.redis_layer = RedisLayer(app.state.engine.settings)
        app.state.session_store = SessionStore(app.state.engine.settings.session_dir)
        app.state.conversation = ConversationService(app.state.engine,app.state.session_store)
        task = asyncio.create_task(app.state.engine.start()) if manage_lifecycle else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            if manage_lifecycle:
                await app.state.engine.close()
            await app.state.redis_layer.close()

    app = FastAPI(title='影视评论混合检索与 Agent 服务',version=SERVICE_VERSION,lifespan=lifespan)

    @app.middleware('http')
    async def clock_request(request,call_next):
        request.state.started=time.monotonic()
        layer = getattr(request.app.state, 'redis_layer', None)
        if layer and layer.settings.rate_limit_enabled and request.method == 'POST' and request.url.path in (
                '/api/v1/query', '/api/v1/query/stream', '/api/v1/agent') or (
                layer and layer.settings.rate_limit_enabled and request.method=='POST' and '/sessions/' in request.url.path and '/query' in request.url.path):
            try:
                allowed = await layer.allow(request.client.host if request.client else 'unknown')
            except RedisError:
                return JSONResponse({'detail': {'code': 'RATE_LIMIT_UNAVAILABLE', 'message': 'Redis 限流暂不可用'}}, status_code=503)
            if not allowed:
                return JSONResponse({'detail': {'code': 'RATE_LIMITED', 'message': '请求过于频繁，请稍后再试'}}, status_code=429)
        return await call_next(request)

    @app.get('/health')
    async def health():
        settings=app.state.engine.settings
        return {'status':'alive','service':'filmreview-service','version':SERVICE_VERSION,'runtime_source_sha256':runtime_source_hashes(),
            'conversation_config':{'context_budget_bytes':settings.session_context_bytes,'recent_turns':settings.session_recent_turns,
                'summary_chars':settings.session_summary_chars,'max_turns':settings.session_max_turns}}

    @app.get('/ready')
    async def ready():
        state = dict(await app.state.engine.ready())
        state['redis'] = await app.state.redis_layer.ready()
        state['ready'] = state['ready'] and state['redis']['ready']
        return JSONResponse(state,status_code=200 if state['ready'] else 503)

    async def require_ready():
        state = dict(await app.state.engine.ready())
        if app.state.redis_layer.enabled:
            state['redis'] = await app.state.redis_layer.ready()
            state['ready'] = state['ready'] and state['redis']['ready']
        if not state['ready']:
            raise HTTPException(status_code=503,detail={'code':'NOT_READY','state':state})
        return state

    @app.get('/api/v1/stats')
    async def stats():
        await require_ready()
        return await app.state.engine.graph.stats()

    @app.get('/api/v1/catalog')
    async def catalog():
        await require_ready()
        return {'movies':app.state.engine.movies,'topics':app.state.engine.topics,'topic_scope_note':topic_config()['scope_note'],'service_version':SERVICE_VERSION}

    @app.post('/api/v1/query',response_model=QueryResult)
    async def query(request: QueryRequest,http_request: Request):
        state = await require_ready()
        layer = app.state.redis_layer
        key = None
        if layer.settings.cache_enabled:
            key = layer.cache_key(request,state['database']['manifest']['fingerprint'])
            try:
                saved = await layer.get(key)
                if saved:
                    return replay_cached(saved,http_request.state.started)
            except RedisError:
                key = None
        result = await app.state.engine.query(request,started=http_request.state.started)
        if layer.settings.cache_enabled:
            result = result.model_copy(update={'cache': {'hit': False, 'stored': bool(key and result.status == 'ok')}})
            if key and result.status == 'ok':
                try:
                    await layer.set(key,result)
                except RedisError:
                    result = result.model_copy(update={'cache': {'hit': False, 'stored': False}})
        return result

    @app.post('/api/v1/query/stream')
    async def stream_query(request: QueryRequest,http_request: Request):
        await require_ready()
        async def body():
            async for item in app.state.engine.events(request,started=http_request.state.started):
                yield 'event: '+item['event']+'\ndata: '+json.dumps(item['data'],ensure_ascii=False)+'\n\n'
        return StreamingResponse(body(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.post('/api/v1/agent',response_model=QueryResult)
    async def agent(request: AgentRequest,http_request: Request):
        await require_ready()
        return await app.state.agent.ask(request,started=http_request.state.started)

    @app.exception_handler(SessionConflict)
    async def conflict_handler(request,exc):
        return JSONResponse({'detail':{'code':'SESSION_CONFLICT','message':str(exc)}},status_code=409)

    def session(identifier):
        try:return app.state.session_store.load(identifier)
        except KeyError as exc:raise HTTPException(404,detail={'code':'SESSION_EXPIRED','message':str(exc)}) from exc

    @app.post('/api/v1/sessions',status_code=201)
    async def create_session():
        return app.state.session_store.create().model_dump(exclude={'replay'})

    @app.get('/api/v1/sessions/{identifier}')
    async def read_session(identifier: str):
        return session(identifier).model_dump(exclude={'replay'})

    @app.delete('/api/v1/sessions/{identifier}',status_code=204)
    async def delete_session(identifier: str):
        app.state.session_store.delete(identifier)
        return Response(status_code=204)

    @app.patch('/api/v1/sessions/{identifier}/conditions')
    async def patch_conditions(identifier: str,patch: ConditionPatch):
        session(identifier)
        try:state=app.state.conversation.patch(identifier,patch)
        except ValueError as exc:raise HTTPException(422,detail={'code':'INVALID_CONDITIONS','message':str(exc)}) from exc
        return state.model_dump(exclude={'replay'})

    @app.post('/api/v1/sessions/{identifier}/query',response_model=QueryResult)
    async def session_query(identifier: str,request: SessionRequest,http_request: Request):
        await require_ready();session(identifier)
        return await app.state.conversation.ask(identifier,request,started=http_request.state.started)

    @app.post('/api/v1/sessions/{identifier}/query/stream')
    async def session_stream(identifier: str,request: SessionRequest,http_request: Request):
        await require_ready();state=session(identifier)
        if request.expected_version is not None and state.version!=request.expected_version and not any(r['id']==request.client_request_id for r in state.replay):
            raise SessionConflict('会话版本已改变，请刷新后再提交')
        async def body():
            try:
                async for item in app.state.conversation.events(identifier,request,started=http_request.state.started):
                    yield 'event: '+item['event']+'\ndata: '+json.dumps(item['data'],ensure_ascii=False)+'\n\n'
            except SessionConflict as exc:
                from uuid import uuid4
                result=QueryResult(request_id=str(uuid4()),strategy='session',status='error',answer=str(exc),error={'code':'SESSION_CONFLICT','message':str(exc)})
                yield 'event: done\ndata: '+result.model_dump_json()+'\n\n'
        return StreamingResponse(body(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    return app


app = create_app()
