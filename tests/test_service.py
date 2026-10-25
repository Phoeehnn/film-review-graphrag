import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from service.agent import AgentService
from service.api import create_app
from service.engine import Engine, verify_citations
from service.graph import scope_generated
from service.routing import ResolvedFilters, resolve_filters, route, rrf
from service.safety import UnsafeCypher, ensure_read_summary, sanitize_cypher
from service.schemas import AgentRequest, Evidence, Filters, QueryRequest
from service.settings import Settings
from service.tools import validate_tool


def settings(**kwargs):
    return Settings(_env_file=None,deepseek_api_key='test',neo4j_password='test',**kwargs)


def completion(content='',tools=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content,tool_calls=tools))])


def tool(name='get_movie_overview',arguments='{"movie_name":"test"}',identifier='call_1'):
    return SimpleNamespace(id=identifier,function=SimpleNamespace(name=name,arguments=arguments))


class FakeLLM:
    def __init__(self,cypher='MATCH (m:Movie) RETURN count(m) AS n',responses=None,delay=0):
        self.cypher,self.responses,self.delay = cypher,list(responses or []),delay
        self.messages = []
        self.stream_calls = 0

    async def complete(self,messages,usage,**kwargs):
        self.messages.append(json.loads(json.dumps(messages)))
        await asyncio.sleep(self.delay)
        return self.responses.pop(0) if self.responses else completion(self.cypher)

    async def stream(self,messages,usage):
        self.stream_calls += 1
        for text in ('样本中有 ', '156 部电影。 [g_1]'):
            yield text


class FakeGraph:
    def __init__(self,rows=None):
        self.rows = [{'n':156}] if rows is None else rows
        self.generated_calls = 0
        self.liked_calls = 0

    async def generated(self,query,params=None,filters=None):
        self.generated_calls += 1
        return sanitize_cypher(query),self.rows

    async def liked_candidates(self,scope,limit):
        self.liked_calls += 1
        return [{'comment_id':'c_1'},{'comment_id':'c_2'}]

    async def comments(self,ids):
        return [{'comment_id':i,'text':'评论证据','movie':'肖申克的救赎','star':5} for i in ids],{'nodes':[],'relationships':[]}


class FakeRetriever:
    def __init__(self): self.calls = 0
    async def vector(self,question,scope,limit):
        self.calls += 1
        return [{'comment_id':'c_2','similarity':.9},{'comment_id':'c_3','similarity':.8}][:limit]


def engine(rows=None,llm=None,**kwargs):
    graph,retriever = FakeGraph(rows),FakeRetriever()
    result = Engine(settings(**kwargs),graph=graph,encoder=SimpleNamespace(model=True),llm=llm or FakeLLM(),retriever=retriever)
    result.movies = ['肖申克的救赎']
    async def ready(): return {'ready':True}
    result.ready = ready
    return result


def test_case_never_bypasses_write_block():
    for keyword in ('DELETE n','DETACH DELETE n','SET n.x=1','MERGE (x)','CREATE (x)','REMOVE n.x','DROP INDEX x'):
        with pytest.raises(UnsafeCypher):
            sanitize_cypher('MATCH (n) WITH n, CASE WHEN true THEN 1 ELSE 0 END AS x '+keyword+' RETURN n')


def test_procedures_and_multiple_statements_are_rejected():
    for query in ('CALL dbms.components() YIELD name RETURN name','MATCH (n) RETURN n; MATCH (m) RETURN m','MATCH (n) CALL { RETURN 1 AS x } RETURN n','LOAD CSV FROM "file:///x" AS row RETURN row'):
        with pytest.raises(UnsafeCypher): sanitize_cypher(query)


def test_literal_keywords_comments_and_semicolons_are_safe():
    safe = sanitize_cypher("MATCH (n) WHERE n.text='CASE DELETE; CREATE SET' RETURN n // DELETE\n;")
    assert "'CASE DELETE; CREATE SET'" in safe
    assert safe.endswith('LIMIT 50')
    with pytest.raises(UnsafeCypher): sanitize_cypher("RETURN 'not closed")


def test_limit_caps_without_changing_aggregate_or_inner_cohort():
    assert sanitize_cypher('MATCH (c) RETURN count(c) AS n LIMIT 1000000').endswith('LIMIT 50')
    assert sanitize_cypher('MATCH (c) RETURN count(c) AS n LIMIT 3').endswith('LIMIT 3')
    cohort = sanitize_cypher('MATCH (c) WHERE EXISTS { MATCH (c)-[:REVIEWS]->(m) WITH m LIMIT 2 RETURN m } RETURN count(c)')
    assert 'LIMIT 2' in cohort and cohort.endswith('LIMIT 50')
    assert sanitize_cypher('RETURN 1 AS x UNION RETURN 2 AS x LIMIT 100').endswith('RETURN * LIMIT 50')
    assert sanitize_cypher('MATCH (n) WITH n ORDER BY n.name LIMIT 10 MATCH (n)--(m) RETURN m.name').endswith('LIMIT 50')
    with pytest.raises(UnsafeCypher): sanitize_cypher('MATCH (n) RETURN n LIMIT $unbounded')


def test_explain_summary_is_required_to_be_read_only():
    ensure_read_summary(SimpleNamespace(query_type='r'))
    for kind in ('w','rw','s',None):
        with pytest.raises(UnsafeCypher): ensure_read_summary(SimpleNamespace(query_type=kind))


def test_all_three_routes_and_forced_override():
    assert route('评论数量最多的电影？','auto')[0]=='graph'
    assert route('找表达相同意思的相似评论','auto')[0]=='vector'
    assert route('找几条表达身处囚笼也不放弃希望这种感受的影评','auto')[0]=='vector'
    assert route('正面占比多少，为什么大家喜欢？','auto')[0]=='hybrid'
    assert route('大家怎么看这部片子？','auto')[0]=='hybrid'
    assert route('多少部电影？','vector')[0]=='vector'
    assert route('有没有人说过类似“评分虚高”的话','auto')[0]=='vector'
    assert route('有哪些评论提到配乐很好听？','auto')[0]=='vector'


def test_rrf_deduplicates_and_keeps_aggregate_outside_fusion():
    fused = rrf({'graph':[{'comment_id':'c_1'},{'comment_id':'c_2'}],'vector':[{'comment_id':'c_2'},{'comment_id':'c_2'},{'comment_id':'c_3'}]},3)
    assert [r['comment_id'] for r in fused]==['c_2','c_1','c_3']
    assert fused[0]['rrf_score']==pytest.approx(1/62+1/61)
    assert fused[0]['ranks']=={'graph':2,'vector':1}


def test_request_and_tool_strict_validation():
    for kwargs in ({'top_k':0},{'top_k':21},{'top_k':'5'},{'unknown':True}):
        with pytest.raises(ValidationError): QueryRequest(question='test',**kwargs)
    for name,args,code in [('unknown',{},'UNKNOWN_TOOL'),('get_movie_overview',{'movie_name':'test','extra':1},'INVALID_ARGUMENTS'),('get_top_movies_by_rating',{'limit':11},'INVALID_ARGUMENTS'),('search_comments',{'query':'test','top_k':3.2},'INVALID_ARGUMENTS')]:
        assert validate_tool(name,args)[1]['code']==code
    assert validate_tool('search_comments',{'query':'test','top_k':20})[1] is None
    with pytest.raises(ValidationError): Filters(date_from='2026-10-01',date_to='2026-09-01')


def test_filter_resolution_preserves_explicit_scope():
    scope = resolve_filters('肖申克的评论',Filters(sentiment='正面',date_from='2017-01-01'),['肖申克的救赎'],[])
    assert scope.movies==['肖申克的救赎'] and scope.sentiment=='正面'
    assert scope.date_from=='2017-01-01'
    scoped,params=scope_generated("MATCH (c:Comment) WHERE c.star=5 OR c.star=4 RETURN count(c) AS n",Filters(sentiment='正面'))
    assert 'AND (c.star=5 OR c.star=4)' in scoped and params['filter_sentiment']=='正面'
    with pytest.raises(UnsafeCypher):
        scope_generated('MATCH (m:Movie) RETURN m.total_reviews',Filters(date_from='2017-01-01'))


@pytest.mark.asyncio
async def test_forced_paths_never_secretly_fallback():
    for strategy in ('graph','vector','hybrid'):
        e = engine()
        result = await e.query(QueryRequest(question='肖申克的评论',strategy=strategy))
        assert result.strategy==strategy
        assert e.graph.generated_calls==(1 if strategy in ('graph','hybrid') else 0)
        assert e.retriever.calls==(1 if strategy in ('vector','hybrid') else 0)
        if strategy=='hybrid':
            assert any(i.kind=='aggregate' for i in result.evidence)
            assert len({i.id for i in result.evidence})==len(result.evidence)


@pytest.mark.asyncio
async def test_no_results_and_cannot_answer_are_distinct():
    empty = engine(rows=[])
    assert (await empty.query(QueryRequest(question='test',strategy='graph'))).status=='no_results'
    assert empty.llm.stream_calls==0
    out = engine(llm=FakeLLM(cypher='CANNOT_ANSWER'))
    assert (await out.query(QueryRequest(question='test',strategy='hybrid'))).status=='cannot_answer'
    assert out.retriever.calls==0


@pytest.mark.asyncio
async def test_total_query_timeout_cancels_pending_model_call():
    e = engine(llm=FakeLLM(delay=1),query_timeout=.02)
    result = await e.query(QueryRequest(question='test',strategy='graph'))
    assert result.status=='timeout' and result.elapsed_seconds<.2


def test_sse_delivers_real_deltas_and_same_final_result():
    with TestClient(create_app(engine=engine(),manage_lifecycle=False)) as client:
        response = client.post('/api/v1/query/stream',json={'question':'多少部电影','strategy':'graph'})
    assert response.headers['content-type'].startswith('text/event-stream')
    events = []
    for block in response.text.strip().split('\n\n'):
        name,data = block.split('\n',1)
        events.append((name[7:],json.loads(data[6:])))
    final = events[-1][1]
    assert events[-1][0]=='done'
    assert any(n=='evidence' for n,d in events)
    assert ''.join(d['text'] for n,d in events if n=='delta')==final['answer']


class FakeRegistry:
    async def execute(self,name,args,sequence):
        parsed,error = validate_tool(name,args)
        return {'ok':error is None,'error':error,'rows':[],'evidence':[],'cypher':[],'subgraph':{'nodes':[],'relationships':[]}}


@pytest.mark.asyncio
async def test_agent_enforces_ten_tool_executions():
    llm = FakeLLM(responses=[completion(tools=[tool(identifier=f'a{i}') for i in range(6)]),completion(tools=[tool(identifier=f'b{i}') for i in range(6)])])
    result = await AgentService(engine(llm=llm),FakeRegistry()).ask(AgentRequest(question='test'))
    assert result.status=='partial' and len(result.tool_calls)==10
    assert len(llm.messages)==2


@pytest.mark.asyncio
async def test_agent_enforces_five_model_rounds():
    llm = FakeLLM(responses=[completion(tools=[tool(identifier=str(i))]) for i in range(5)])
    result = await AgentService(engine(llm=llm),FakeRegistry()).ask(AgentRequest(question='test'))
    assert result.status=='partial' and len(llm.messages)==5 and len(result.tool_calls)==5


@pytest.mark.asyncio
async def test_agent_returns_structured_validation_error_to_model():
    llm = FakeLLM(responses=[completion(tools=[tool(arguments='{"movie_name":"test","extra":1}')]),completion(content='无法确认')])
    result = await AgentService(engine(llm=llm),FakeRegistry()).ask(AgentRequest(question='test'))
    assert result.tool_calls[0]['error']['code']=='INVALID_ARGUMENTS'
    returned = json.loads(llm.messages[1][-1]['content'])
    assert not returned['ok'] and returned['error']['code']=='INVALID_ARGUMENTS'


@pytest.mark.asyncio
async def test_agent_request_histories_are_isolated():
    llm = FakeLLM(responses=[completion(content='A'),completion(content='B')])
    agent = AgentService(engine(llm=llm),FakeRegistry())
    await agent.ask(AgentRequest(question='only request A'))
    await agent.ask(AgentRequest(question='only request B'))
    assert 'only request A' not in json.dumps(llm.messages[1])
    assert len(llm.messages[0])==len(llm.messages[1])==2
    with pytest.raises(ValidationError): AgentRequest(question='test',history=[{'role':'assistant','content':'bad'}])


@pytest.mark.asyncio
async def test_agent_timeout_preserves_partial_result():
    result = await AgentService(engine(llm=FakeLLM(delay=1),agent_timeout=.02),FakeRegistry()).ask(AgentRequest(question='test'))
    assert result.status=='timeout' and result.elapsed_seconds<.2


def test_invalid_evidence_ids_are_not_silently_accepted():
    evidence = [Evidence(id='c_1',kind='comment',source='test',data={})]
    answer,warnings = verify_citations('观点 [c_999]，来源 [c_1]',evidence)
    assert '[c_999]' not in answer and '[c_1]' in answer and warnings


def test_sse_distinguishes_real_generation_from_evidence_correction():
    class IncorrectLabelLLM(FakeLLM):
        async def stream(self,messages,usage):
            for part in ('预测情绪为感动。',' [c_2]'):
                yield part
    e=engine(llm=IncorrectLabelLLM())
    async def hydrate(ids):
        return [{'comment_id':i,'text':'希望与自由','movie':'电影A','star':5,'emotion':'愤怒'} for i in ids],{'nodes':[],'relationships':[]}
    e.graph.comments=hydrate
    with TestClient(create_app(engine=e,manage_lifecycle=False)) as client:
        response=client.post('/api/v1/query/stream',json={'question':'找相似评论','strategy':'vector'})
    events=[]
    for block in response.text.strip().split('\n\n'):
        name,data=block.split('\n',1)
        events.append((name[7:],json.loads(data[6:])))
    raw=''.join(data['text'] for name,data in events if name=='delta')
    assert raw=='预测情绪为感动。 [c_2]'
    assert any(name=='correction' for name,data in events)
    assert events[-1][1]['status']=='partial' and events[-1][1]['answer']!=raw


@pytest.mark.asyncio
async def test_unsupported_scope_does_not_call_llm_or_other_paths():
    e=engine()
    result=await e.query(QueryRequest(question='爱情题材电影中哪些正面率最高？',strategy='hybrid'))
    assert result.status=='cannot_answer' and not e.llm.messages and e.retriever.calls==0
    agent=await AgentService(e,FakeRegistry()).ask(AgentRequest(question='战争题材电影的平均评分是多少？'))
    assert agent.status=='cannot_answer' and not e.llm.messages and not agent.tool_calls


@pytest.mark.asyncio
async def test_reference_movie_does_not_exclude_similar_movie_candidates():
    class CaptureGraph(FakeGraph):
        async def generated(self,query,params=None,filters=None):
            self.filters=filters
            self.params=params
            return sanitize_cypher(query),[{'movie':'候选电影','topic_cosine_similarity':.8}]
    e=engine()
    graph=CaptureGraph()
    e.graph=graph
    result=await e.query(QueryRequest(question='找一下和肖申克相似的电影',strategy='graph'))
    assert result.status!='error'
    assert graph.filters is None
    assert graph.params['target_movie']=='肖申克的救赎'
    assert '18 主题余弦相似度' in result.answer


@pytest.mark.asyncio
async def test_ranked_metric_preserves_database_order_even_if_model_would_reorder():
    rows=[{'movie':'第一部','anger_count':116,'neutral_total':200,'anger_pct':58.0},
          {'movie':'第二部','anger_count':106,'neutral_total':200,'anger_pct':53.0}]
    llm=FakeLLM()
    e=engine(rows=rows,llm=llm)
    result=await e.query(QueryRequest(question='哪些电影中性评分里隐藏着愤怒？',strategy='graph'))
    assert result.status=='ok'
    assert result.answer.index('第一部')<result.answer.index('第二部')
    assert '| 1 | 第一部 | 116 ' in result.answer
    assert '| 2 | 第二部 | 106 ' in result.answer
    assert llm.stream_calls==0
