import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from service.api import create_app
from service.conversation import ConversationService,extract_patch
from service.cypher_repair import run_with_repair
from service.llm import Usage
from service.safety import sanitize_cypher, UnsafeCypher
from service.schemas import Evidence, QueryResult
from service.session_models import ConditionPatch, SessionRequest
from service.session_store import SessionConflict, SessionStore
from service.settings import Settings
from service.references import reference_result
from service.graph import scope_generated
from service.routing import resolve_filters
from service.schemas import Filters


class Model:
    def __init__(self,delay=0):self.delay=delay
    async def complete(self,messages,usage,**kwargs):
        usage.calls+=1
        await asyncio.sleep(self.delay)
        if 'standalone_question' in messages[0]['content']:
            c=json.loads(messages[-1]['content'])['context']['conditions']
            text=json.dumps({'standalone_question':f"找{c.get('top_k',2)}条{c.get('movie','')}的{c.get('sentiment','')}评论。"},ensure_ascii=False)
        else:text='此前讨论了电影评论；本摘要不提供新条件。'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class MiniEngine:
    def __init__(self,path,delay=0):
        self.settings=Settings(_env_file=None,session_dir=path)
        self.movies=['电影甲','电影乙'];self.topics=[];self.llm=Model(delay)
        self.calls=0
    async def ready(self):return {'ready':True}
    async def events(self,request,started=None,retry_budget=None):
        self.calls+=1
        yield {'event':'progress','data':{'stage':'retrieval'}}
        result=QueryResult(request_id='test-'+str(self.calls),strategy='vector',status='ok',answer='两条相关评论。',
            evidence=[Evidence(id='c_'+str(i),kind='comment',source='vector',data={'text':'测试评论'}) for i in range(1,3)])
        yield {'event':'done','data':result.model_dump()}


def service(tmp_path,delay=0):
    engine=MiniEngine(tmp_path,delay);store=SessionStore(tmp_path)
    return ConversationService(engine,store),store


@pytest.mark.asyncio
async def test_confirm_revoke_and_compress_cannot_restore_pinned(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    for i,question in enumerate(['找两条电影甲的负面评论。','那正面的呢？','找两条电影甲的评论，只看2018-01-01之后。','不用限定时间了。']):
        r=await svc.ask(state.id,SessionRequest(question=question,client_request_id=str(i),confirm_conditions=True))
        assert r.status=='ok',r.answer
    final=store.load(state.id)
    assert final.pinned['movie']=='电影甲'
    assert final.pinned['sentiment']=='正面'
    assert 'date_from' not in final.pinned and 'date_from' not in final.active_conditions
    assert 'date_from' in final.revoked and final.summary


@pytest.mark.asyncio
async def test_unconfirmed_candidates_not_promoted_and_idempotency(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    req=SessionRequest(question='找两条电影甲的负面评论。',client_request_id='once',expected_version=0)
    first=await svc.ask(state.id,req);second=await svc.ask(state.id,req)
    assert first.request_id==second.request_id and svc.engine.calls==1
    saved=store.load(state.id)
    assert not saved.pinned and saved.candidates['sentiment']=='负面'
    with pytest.raises(SessionConflict):
        await svc.ask(state.id,req.model_copy(update={'question':'其他问题'}))


def test_sqlite_concurrent_version_and_expiry(tmp_path):
    now=[1000.0];store=SessionStore(tmp_path,clock=lambda:now[0]);state=store.create()
    store.commit(state,0)
    with pytest.raises(SessionConflict):store.commit(state,0)
    now[0]=state.expires_at
    with pytest.raises(KeyError):store.load(state.id)


@pytest.mark.asyncio
async def test_session_isolation_no_history_and_cancel_before_commit(tmp_path):
    svc,store=service(tmp_path);a,b=store.create(),store.create()
    await svc.ask(a.id,SessionRequest(question='找两条电影甲的评论。',client_request_id='1',confirm_conditions=True))
    assert not store.load(b.id).pinned
    r=await svc.ask(a.id,SessionRequest(question='那正面的呢？',client_request_id='2',context_mode='none'))
    assert 'movie' not in r.conversation.get('effective_conditions',{})
    stream=svc.events(b.id,SessionRequest(question='找两条电影乙的评论。',client_request_id='3'))
    await stream.__anext__();await stream.aclose()
    assert store.load(b.id).version==0


@pytest.mark.asyncio
async def test_summary_never_changes_fixed_conditions(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找两条电影甲的负面评论。',client_request_id='0',confirm_conditions=True))
    for i in range(1,5):
        await svc.ask(state.id,SessionRequest(question='找关于生活的评论。',client_request_id=str(i)))
    saved=store.load(state.id)
    assert saved.summary and saved.pinned['movie']=='电影甲' and saved.pinned['sentiment']=='负面'


def test_condition_patch_is_explicit_and_invalid_scope_not_committed(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    saved=svc.patch(state.id,ConditionPatch(expected_version=0,updates={'movie':'电影甲','top_k':2}))
    with pytest.raises(ValueError):svc.patch(state.id,ConditionPatch(expected_version=1,updates={'date_from':'2018-01-01','date_to':'2010-01-01'}))
    assert store.load(state.id).version==1
    cleared=svc.patch(state.id,ConditionPatch(expected_version=1,unset=['movie']))
    assert 'movie' not in cleared.pinned


def test_session_api_lifecycle_conflict_and_sse(tmp_path):
    engine=MiniEngine(tmp_path)
    with TestClient(create_app(engine,manage_lifecycle=False)) as client:
        s=client.post('/api/v1/sessions').json()
        assert client.patch('/api/v1/sessions/'+s['id']+'/conditions',json={'expected_version':0,'updates':{'movie':'电影甲'}}).status_code==200
        assert client.patch('/api/v1/sessions/'+s['id']+'/conditions',json={'expected_version':0,'updates':{}}).status_code==409
        response=client.post('/api/v1/sessions/'+s['id']+'/query/stream',json={'question':'找两条电影甲的评论。','client_request_id':'x'})
        assert response.status_code==200 and 'event: done' in response.text
        assert client.delete('/api/v1/sessions/'+s['id']).status_code==204
        assert client.get('/api/v1/sessions/'+s['id']).status_code==404


class SequenceModel:
    def __init__(self,outputs):self.outputs=iter(outputs)
    async def complete(self,messages,usage,**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.outputs)))])


class CheckedGraph:
    def __init__(self,rows=None):self.executed=[];self.rows=rows if rows is not None else [{'n':1}]
    async def generated(self,q,filters=None):
        safe=sanitize_cypher(q)
        if 'bad_syntax' in q:raise ValueError('syntax error')
        self.executed.append(safe);return safe,self.rows


async def repair(outputs,graph,limit=2):
    trace=[]
    result=await run_with_repair(llm=SequenceModel(outputs),graph=graph,messages=[],usage=Usage(),filters=None,max_rewrites=limit,trace=trace,redact=str)
    return result,trace


@pytest.mark.asyncio
async def test_repair_third_attempt_and_all_candidates_safe():
    graph=CheckedGraph()
    result,trace=await repair(['MATCH (n) bad_syntax RETURN n','MATCH (n) bad_syntax RETURN n','MATCH (n) RETURN count(n) AS n'],graph)
    assert len(trace)==3 and len(graph.executed)==1 and result[1]==[{'n':1}]
    graph=CheckedGraph()
    with pytest.raises(UnsafeCypher):await repair(['MATCH (n) DELETE n RETURN n','MATCH (n) SET n.x=1 RETURN n','MATCH (n) RETURN n'],graph)
    assert not graph.executed


@pytest.mark.asyncio
async def test_empty_result_stops_without_rewrite_and_no_rewrite_control():
    graph=CheckedGraph(rows=[])
    result,trace=await repair(['MATCH (n) RETURN n','MATCH (n) RETURN n'],graph)
    assert len(trace)==1 and len(graph.executed)==1 and not result[1] and trace[-1]['adopted'] and trace[-1]['stop_reason']=='empty_result_preserved'
    graph=CheckedGraph()
    with pytest.raises(ValueError):await repair(['MATCH (n) bad_syntax RETURN n'],graph,0)
    assert not graph.executed


@pytest.mark.asyncio
async def test_changed_condition_is_inherited_without_automatic_confirmation(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找两条电影甲的负面评论。',client_request_id='a',confirm_conditions=True))
    await svc.ask(state.id,SessionRequest(question='那正面的呢？',client_request_id='b'))
    r=await svc.ask(state.id,SessionRequest(question='找关于生活的评论。',client_request_id='c'))
    assert r.conversation['effective_conditions']['sentiment']=='正面'
    assert store.load(state.id).pinned['sentiment']=='负面'
    r=await svc.ask(state.id,SessionRequest(question='找电影乙的评论。',filters={'movie':'电影甲'},client_request_id='d'))
    assert r.conversation['clarification_required'] and store.load(state.id).version==3


@pytest.mark.asyncio
async def test_comment_reference_uses_actual_id_and_handles_missing_author(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    state.last_results=[{'ordinal':1,'kind':'comment','id':'c_real'}]
    class Graph:
        def __init__(self):self.calls=[];self.authors=[{'user_id':'u_real'}]
        async def run(self,q,params):
            self.calls.append((q,params))
            return self.authors if len(self.calls)%2 else [{'comment_id':'c_other'}]
        async def comments(self,ids):return [{'comment_id':'c_other','movie':'电影乙','text':'另一条原文'}],{'nodes':[],'relationships':[]}
    svc.engine.graph=Graph()
    req=SessionRequest(question='第一条的作者还写过什么？',client_request_id='r',top_k=2)
    result=await reference_result(svc.engine,state,1,req,'ref')
    assert result.status=='ok' and result.evidence[0].id=='c_other'
    assert svc.engine.graph.calls[0][1]=={'id':'c_real'}
    assert svc.engine.graph.calls[1][1]=={'user_id':'u_real','id':'c_real','limit':2}
    svc.engine.graph.authors=[];svc.engine.graph.calls=[]
    result=await reference_result(svc.engine,state,1,req,'ref2')
    assert result.status=='cannot_answer' and len(svc.engine.graph.calls)==1
    result=await reference_result(svc.engine,state,2,req,'ref3')
    assert result.status=='cannot_answer' and not result.evidence


def test_scope_prefix_predicates_and_reviewed_movie_alias():
    for predicate in ("c.date STARTS WITH '2018'","c.text ENDS WITH '自由'"):
        query,_=scope_generated('MATCH (c:Comment) WHERE '+predicate+' RETURN count(c) AS n',Filters(movie='三块广告牌'))
        assert '('+predicate+')' in query
        assert 'RETURN count(c)' in query
    movie='西游记之大圣归来 Monkey King: Hero Is Back'
    scope=resolve_filters('找两条大圣归来的负面评论',Filters(),[movie,'三块广告牌'],[])
    assert scope.movies==[movie] and scope.sentiment=='负面'


def test_gender_topic_does_not_create_neutral_sentiment():
    engine=SimpleNamespace(movies=['三块广告牌','西游记之大圣归来'],topics=[])
    for question in ('找两条三块广告牌中性别议题的评论。','找两条大圣归来中性别议题的评论。'):
        updates,_=extract_patch(question,engine,Filters())
        assert updates['topic']=='t_6' and 'sentiment' not in updates
    updates,_=extract_patch('找三块广告牌的中性评论。',engine,Filters())
    assert updates['sentiment']=='中性'


@pytest.mark.asyncio
async def test_explicit_count_can_change_inherited_count(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找两条电影甲的评论',client_request_id='1'))
    r=await svc.ask(state.id,SessionRequest(question='找谈生活的评论',client_request_id='2',top_k=7))
    assert r.conversation['effective_conditions']['top_k']==7
    r=await svc.ask(state.id,SessionRequest(question='找三条电影甲的评论',client_request_id='3',top_k=7))
    assert r.conversation['effective_conditions']['top_k']==3


# --- 2026-10-03 review fixes -------------------------------------------------

@pytest.mark.asyncio
async def test_valid_empty_result_never_requests_a_refusing_rewrite():
    # Real case Q10 (no movie has >1000 comments): the rewrite answered
    # CANNOT_ANSWER and the correct zero-row result used to be discarded.
    graph=CheckedGraph(rows=[])
    result,trace=await repair(['MATCH (m:Movie) WITH m WHERE m.total_reviews>1000 RETURN m.name AS movie','CANNOT_ANSWER'],graph)
    safe,rows,unsupported=result
    assert safe and rows==[] and unsupported is False
    assert len(trace)==1 and trace[-1]['result_kind']=='empty'


@pytest.mark.asyncio
async def test_repeating_an_empty_query_keeps_it_without_rerunning():
    graph=CheckedGraph(rows=[])
    result,trace=await repair(['MATCH (n) RETURN n','MATCH (n)\n  RETURN n'],graph)
    assert result[1]==[] and len(graph.executed)==1 and len(trace)==1 and trace[-1]['status']=='empty'


@pytest.mark.asyncio
async def test_repeated_failed_candidate_is_not_executed_again():
    graph=CheckedGraph()
    result,trace=await repair(['MATCH (n) bad_syntax RETURN n','MATCH (n) bad_syntax RETURN n','MATCH (n) RETURN count(n) AS n'],graph)
    assert [a['status'] for a in trace]==['failed','repeated','ok'] and len(graph.executed)==1


def test_scope_shape_errors_are_not_counted_as_write_attempts():
    from service.cypher_repair import classify_failure
    from service.safety import ScopeUnsupported
    query='MATCH (c:Comment)<-[:POSTED]-(u:User) WITH DISTINCT u MATCH (u)-[:POSTED]->(c2:Comment) RETURN count(c2) AS n'
    with pytest.raises(ScopeUnsupported) as caught:scope_generated(query,Filters(movie='肖申克'))
    assert classify_failure(caught.value)=='scope' and '范围' in str(caught.value)
    with pytest.raises(UnsafeCypher) as caught:sanitize_cypher('MATCH (n) DETACH DELETE n RETURN 1')
    assert classify_failure(caught.value)=='unsafe'


def test_star_ranges_counts_and_years_are_parsed_as_ranges():
    engine=SimpleNamespace(movies=['肖申克的救赎 The Shawshank Redemption'],topics=[])
    cases={'找肖申克4星以上的评论':{'min_star':4},'找肖申克3星以下的评论':{'max_star':3},
           '只看1到2星的评论':{'min_star':1,'max_star':2},'至少4星的评论':{'min_star':4},'只看五星的':{'min_star':5,'max_star':5},
           '列出前十二条评论':{'top_k':12},'找二十条评论':{'top_k':20},'肖申克2023年之前的评论':{'date_to':'2022-12-31'}}
    for question,expected in cases.items():
        updates,_=extract_patch(question,engine,Filters())
        assert {k:updates[k] for k in expected}==expected and not ({'min_star','max_star'}-set(expected))&set(updates),question


def test_two_sentiments_ask_for_clarification_instead_of_guessing():
    engine=SimpleNamespace(movies=['肖申克的救赎 The Shawshank Redemption'],topics=[])
    for question in ('正面和负面评论各找两条','找肖申克中的负面评论和正面评论对比'):
        with pytest.raises(ValueError):extract_patch(question,engine,Filters())
    updates,_=extract_patch('找两条肖申克中性别议题的评论',engine,Filters())
    assert 'sentiment' not in updates


@pytest.mark.asyncio
async def test_global_question_does_not_inherit_unconfirmed_scope_and_carry_is_visible(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找两条电影甲的负面评论。',client_request_id='1'))
    r=await svc.ask(state.id,SessionRequest(question='找关于生活的评论。',client_request_id='2'))
    assert r.conversation['effective_conditions']['sentiment']=='负面' and any('未固定的条件' in w for w in r.warnings)
    r=await svc.ask(state.id,SessionRequest(question='哪部电影的评论数量最多？',client_request_id='3'))
    assert 'movie' not in r.conversation['effective_conditions'] and 'sentiment' not in r.conversation['effective_conditions']
    await svc.ask(state.id,SessionRequest(question='找两条电影甲的评论。',client_request_id='4',confirm_conditions=True))
    r=await svc.ask(state.id,SessionRequest(question='哪部电影的评论数量最多？',client_request_id='5'))
    assert r.conversation['effective_conditions']['movie']=='电影甲'


@pytest.mark.asyncio
async def test_rewrite_context_drops_duplicate_fields_and_reports_rewrite_tokens(tmp_path):
    from service.conversation import build_context
    svc,store=service(tmp_path);state=store.create()
    for i,q in enumerate(['找两条电影甲的负面评论。','那正面的呢？']):
        r=await svc.ask(state.id,SessionRequest(question=q,client_request_id=str(i)))
    body,_=build_context(store.load(state.id),{'movie':'电影甲'},'full',svc.settings)
    assert all(set(t)=={'question','standalone_question','answer'} for t in body['recent_turns'])
    assert 'conversation_prompt_tokens' in r.model_usage


def test_long_bilingual_title_does_not_change_route():
    from service.routing import route
    assert route('找两条三块广告牌 Three Billboards Outside Ebbing, Missouri 的评论。','auto')[0]=='vector'
    assert route('找两条关于电影《三块广告牌 Three Billboards Outside Ebbing, Missouri》的正面评论。','auto')[0]=='vector'
    assert route('《三块广告牌》有多少条评论？','auto')[0]=='graph'


@pytest.mark.asyncio
async def test_executed_empty_aggregate_becomes_evidence_of_absence():
    from service.engine import Engine
    from service.schemas import QueryRequest
    from service.settings import Settings
    class Graph:
        async def generated(self,q,params=None,filters=None):return sanitize_cypher(q),[]
        async def comments(self,ids):return [],{'nodes':[],'relationships':[]}
    class LLM:
        async def complete(self,messages,usage,**k):
            usage.calls+=1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='MATCH (m:Movie) WHERE m.total_reviews>1000 RETURN m.name AS movie'))])
        async def stream(self,messages,usage):
            assert 'empty_result' in messages[-1]['content']
            yield '按本次查询条件，样本中没有评论数超过1000条的电影 [g_1]。'
    class Retriever:
        async def vector(self,question,scope,limit):return []
    engine=Engine(Settings(_env_file=None),graph=Graph(),encoder=SimpleNamespace(),llm=LLM(),retriever=Retriever())
    engine.movies,engine.topics=['三块广告牌'],[]
    graph=await engine.query(QueryRequest(question='哪些电影的评论数量超过1000条？',strategy='graph',max_cypher_rewrites=0))
    assert graph.status=='no_results' and '没有满足条件的记录' in graph.answer
    hybrid=await engine.query(QueryRequest(question='哪些电影的评论数量超过1000条？',strategy='hybrid',max_cypher_rewrites=0))
    assert hybrid.status=='ok' and hybrid.evidence[0].data['empty_result'] and '没有' in hybrid.answer
