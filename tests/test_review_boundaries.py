"""Review regressions: real scope/condition contracts, no effect grades."""
import json
from types import SimpleNamespace

import pytest

from service.conversation import ConversationService,build_context
from service.cypher_repair import _normalized,classify_failure
from service.graph import scope_generated
from service.routing import ResolvedFilters,route
from service.safety import ScopeUnsupported
from service.schemas import Filters,QueryRequest
from service.session_models import ConditionPatch,SessionRequest
from service.settings import Settings
from service.statistics import metric_query
from scripts.evaluate_v3 import adopted_success
from tests.test_conversation import service,repair,CheckedGraph


@pytest.mark.asyncio
async def test_pinned_value_restored_after_temporary_override_and_global_question(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找电影甲的评论',client_request_id='a',confirm_conditions=True))
    await svc.ask(state.id,SessionRequest(question='找电影乙的评论',client_request_id='b'))
    result=await svc.ask(state.id,SessionRequest(question='哪部电影评论最多？',client_request_id='c'))
    assert result.conversation['effective_conditions']['movie']=='电影甲'
    assert store.load(state.id).pinned['movie']=='电影甲'


@pytest.mark.asyncio
async def test_explicit_global_clears_temporary_even_followup_but_clarifies_pinned(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找电影甲的负面评论',client_request_id='a'))
    result=await svc.ask(state.id,SessionRequest(question='那所有电影评论最多的是哪部？',client_request_id='b'))
    assert 'movie' not in result.conversation['effective_conditions'] and 'sentiment' not in result.conversation['effective_conditions']
    await svc.ask(state.id,SessionRequest(question='找电影甲的评论',client_request_id='c',confirm_conditions=True))
    before=store.load(state.id).version
    result=await svc.ask(state.id,SessionRequest(question='那全库有多少评论？',client_request_id='d'))
    assert result.conversation['clarification_required'] and store.load(state.id).version==before


@pytest.mark.asyncio
async def test_star_range_replacement_preserves_pinned_and_revocation(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    await svc.ask(state.id,SessionRequest(question='找电影甲2星的评论',client_request_id='a',confirm_conditions=True))
    r=await svc.ask(state.id,SessionRequest(question='改成4星以上的评论',client_request_id='b'))
    assert r.status=='ok' and r.conversation['effective_conditions']['min_star']==4 and 'max_star' not in r.conversation['effective_conditions']
    assert store.load(state.id).pinned['max_star']==2
    r=await svc.ask(state.id,SessionRequest(question='改成3星以下的评论',client_request_id='c'))
    assert r.conversation['effective_conditions']['max_star']==3 and 'min_star' not in r.conversation['effective_conditions']
    r=await svc.ask(state.id,SessionRequest(question='星级不限，找评论',client_request_id='d'))
    assert not {'min_star','max_star'}&set(r.conversation['pinned'])


def test_condition_panel_replaces_rating_range_in_active_and_pinned(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    svc.patch(state.id,ConditionPatch(expected_version=0,updates={'min_star':2,'max_star':2}))
    r=svc.patch(state.id,ConditionPatch(expected_version=1,updates={'min_star':4}))
    assert r.pinned=={'min_star':4} and r.active_conditions=={'min_star':4} and 'max_star' not in r.pinned_sources


@pytest.mark.asyncio
async def test_identifier_and_literal_changes_are_not_deduplicated():
    class Graph(CheckedGraph):
        async def generated(self,q,filters=None):
            if ':comment)' in q:raise ValueError('label does not exist')
            return await super().generated(q,filters)
    result,trace=await repair(['MATCH (c:comment) RETURN c','MATCH (c:Comment) RETURN c'],Graph())
    assert len(trace)==2 and trace[1]['adopted'] and trace[1]['status']=='ok'
    assert _normalized("RETURN 'IT'")!=_normalized("RETURN 'It'")
    assert _normalized("RETURN 'a  b'")!=_normalized("RETURN 'a b'")


@pytest.mark.parametrize('q',['MATCH (m:Movie), (:Comment) RETURN count(*) AS n',
    'MATCH (c:Comment)<-[:POSTED]-(u:User) MATCH (u)-[:POSTED]->(:Comment) RETURN count(*) AS n'])
def test_anonymous_comments_cannot_bypass_scope(q):
    with pytest.raises(ScopeUnsupported) as error:scope_generated(q,Filters(movie='电影甲'))
    assert classify_failure(error.value)=='scope'


def test_author_other_movies_template_declares_two_different_scopes():
    request=QueryRequest(question='评论过电影甲的作者中，有多少位还评论过其他电影？')
    plan=metric_query(request,ResolvedFilters(movies=['电影甲'],sentiment='负面'))
    assert plan.name=='authors_other_movies' and plan.parameters['source_movie']=='电影甲'
    assert plan.parameters['filter_sentiment']=='负面' and 'other.name <> $source_movie' in plan.query
    assert 'count(DISTINCT u)' in plan.query and 'other_comment:Comment' in plan.query


def test_adopted_success_never_uses_stop_reason_strings():
    assert not adopted_success({'cypher_attempts':[{'status':'ok','executed':True,'stop_reason':'empty_result_preserved'}]})
    assert adopted_success({'adopted_attempt':2,'cypher_attempts':[{'attempt':2,'adopted':True,'executed':True,'result_kind':'empty'}]})


def test_context_byte_budget_compatibility_and_metadata(tmp_path):
    settings=Settings(_env_file=None,session_context_tokens=1000)
    assert settings.session_context_bytes==1000
    svc,store=service(tmp_path);state=store.create()
    state.turns=[{'question':'问题','standalone_question':'问题','answer':'很长的中文回答'*200} for _ in range(3)]
    body,meta=build_context(state,{'movie':'电影甲'},'full',settings)
    assert meta['context_bytes']<=1000 and meta['discarded_turns']>0 and body['conditions']=={'movie':'电影甲'}


def test_long_title_mask_is_catalog_based_without_erasing_unknown_targets():
    title='三块广告牌 Three Billboards Outside Ebbing, Missouri'
    assert route('找两条'+title+'的评论','auto',[title])[0]=='vector'
    assert route('找两条'+title+'的评论','auto',[])[0]=='hybrid'
    assert route('找类似 The Last Quiet Moment 的评论','auto',[])[0]=='vector'


@pytest.mark.asyncio
async def test_runtime_failure_never_commits_half_turn(tmp_path):
    svc,store=service(tmp_path);state=store.create()
    async def events(request,**kwargs):
        from service.schemas import QueryResult
        yield {'event':'done','data':QueryResult(request_id='failed',strategy='graph',status='error',answer='数据库不可用').model_dump()}
    svc.engine.events=events
    result=await svc.ask(state.id,SessionRequest(question='找电影甲评论',client_request_id='failure'))
    assert result.status=='error' and store.load(state.id).version==0 and not store.load(state.id).turns
    assert result.conversation['state_committed'] is False


@pytest.mark.asyncio
async def test_empty_result_does_not_trigger_failing_model_call():
    from service.cypher_repair import run_with_repair
    from service.llm import Usage
    class Model:
        calls=0
        async def complete(self,*args,**kwargs):
            self.calls+=1
            if self.calls>1:raise RuntimeError('model unavailable')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='MATCH (n) RETURN n'))])
    model=Model();trace=[]
    result=await run_with_repair(llm=model,graph=CheckedGraph([]),messages=[],usage=Usage(),filters=None,max_rewrites=2,trace=trace,redact=str)
    assert model.calls==1 and result[1]==[] and trace[0]['adopted']
