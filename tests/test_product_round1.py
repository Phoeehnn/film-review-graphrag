import httpx
import pytest
from fastapi.testclient import TestClient
from tests.test_conversation import service, MiniEngine
from service.api import create_app
from service.session_models import SessionRequest
from ui.request_errors import failure_message


@pytest.mark.asyncio
async def test_comparison_preserves_pinned_and_replays(tmp_path):
    svc,store=service(tmp_path)
    state=store.create()
    state.pinned={'movie':'电影甲'}
    state.active_conditions={'movie':'电影甲'}
    state=store.commit(state,0)
    request=SessionRequest(question='对比电影甲和电影乙的情绪',client_request_id='compare',expected_version=state.version)
    result=await svc.ask(state.id,request)
    assert result.status=='ok' and result.conversation['comparison']
    saved=store.load(state.id)
    assert saved.pinned==state.pinned and saved.active_conditions==state.active_conditions
    assert saved.turns==[] and saved.last_results==[]
    again=await svc.ask(state.id,request)
    assert again.request_id==result.request_id and svc.engine.calls==1


def test_catalog_and_stats_survive_model_outage(tmp_path):
    class Graph:
        async def ready(self):return {'ready':True}
        async def catalog(self):return ['电影甲'],[]
        async def stats(self):return {'node_total':1}
    engine=MiniEngine(tmp_path)
    engine.graph=Graph()
    async def unavailable():return {'ready':False,'reason':'model unavailable'}
    engine.ready=unavailable
    with TestClient(create_app(engine,manage_lifecycle=False)) as client:
        assert client.get('/api/v1/catalog').json()['movies']==['电影甲']
        assert client.get('/api/v1/stats').status_code==200
        assert client.post('/api/v1/query',json={'question':'多少条'}).status_code==503


@pytest.mark.parametrize('status,word',[(402,'余额'),(429,'请求较多'),(409,'会话'),(422,'条件'),(503,'未就绪')])
def test_errors_are_actionable_and_do_not_echo_provider(status,word):
    response=httpx.Response(status,request=httpx.Request('POST','http://example.test'),text='private provider body')
    exc=httpx.HTTPStatusError('private token',request=response.request,response=response)
    message=failure_message(exc)
    assert word in message and 'private' not in message


def test_frontend_retains_failed_question_and_retry_payload(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ui import explorer
    sent=[]
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def stream(self,method,url,json):
            sent.append(json.copy())
            raise httpx.ConnectError('offline')
    monkeypatch.setattr(explorer,'catalog_stats',lambda:(None,None))
    monkeypatch.setattr(explorer,'sidebar_controls',lambda *a:('检索问答','auto',5,{}))
    monkeypatch.setattr(explorer,'session_controls',lambda *a:(False,None))
    monkeypatch.setattr(explorer,'example_cards',lambda *a:None)
    monkeypatch.setattr(explorer.httpx,'Client',Client)
    app=AppTest.from_string('from ui.explorer import render_explorer\nrender_explorer()',default_timeout=30).run()
    app.text_input(key='first_question_text').set_value('电影甲有多少条评论')
    app.button(key='send_first_question').click().run()
    assert not app.exception
    assert app.session_state['failed_request']['question']=='电影甲有多少条评论'
    app.button(key='retry_after_error').click().run()
    assert not app.exception and len(sent)==2 and sent[0]==sent[1]
