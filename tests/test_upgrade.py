"""Regression cases from real audits; no automatic performance grades."""
import json
from types import SimpleNamespace

import pytest

from service.evidence_checks import validate_answer
from service.graph import filter_where, scope_generated
from service.routing import ResolvedFilters, comment_scope_violations, resolve_filters, route
from service.schemas import Evidence, Filters, QueryRequest
from service.statistics import metric_metadata, metric_query
from service.topics import canonical_topic, decorate_topic, topic_config, topic_display, unsupported_scope_reason
from service.tools import ToolRegistry


def test_six_route_examples_include_original_audit_failures():
    examples=[('战争话题下情感最负面的前5条','graph'),('数据库里一共有多少部电影？','graph'),
        ('有没有人说过类似希望让人重获自由的话？','vector'),('找表达相同意思的相似评论','vector'),
        ('正面占比多少，为什么大家喜欢？','hybrid'),('大家怎么看这部片子？','hybrid')]
    for question,expected in examples:
        assert route(question,'auto')[0]==expected


def test_all_aliases_are_complete_and_preserve_original_assignments():
    entries=topic_config()['topics']
    assert len(entries)==18 and entries[6]['display_name']=='性别议题'
    for entry in entries:
        for spelling in [entry['id'],entry['display_name'],entry['raw_name'],*entry['aliases']]:
            assert canonical_topic(spelling)==entry['raw_name']
            assert topic_display(spelling)==entry['display_name']
        row=decorate_topic({'id':entry['id'],'name':entry['raw_name'],'keywords':'原始词'})
        assert row['name']==row['raw_name']==entry['raw_name'] and row['keywords']=='原始词'
    with pytest.raises(ValueError):
        decorate_topic({'id':'t_6','name':'另一个聚类'})


def test_alias_resolution_does_not_turn_positive_named_theme_into_sentiment():
    scope=resolve_filters('综合好评主题有多少评论？',Filters(),[],[])
    assert scope.topics==[canonical_topic('t_5')] and scope.sentiment is None
    ids=resolve_filters('性别议题讨论的相似评论',Filters(),[],[])
    assert ids.topics==[canonical_topic('t_6')]
    where,params=filter_where(ids)
    assert params['filter_topics']==[canonical_topic('t_6')]
    query,forced=scope_generated('MATCH (c:Comment) RETURN count(c) AS n',Filters(topic=canonical_topic('t_6')))
    assert 'filter_topics' in query and forced['filter_topics']==ids.topics


def test_missing_genre_scope_is_distinct_from_comment_content():
    for question in ('爱情题材电影中哪些的正面率最高？','战争题材电影的感动率和愤怒率对比','喜剧电影中哪些评分最高？'):
        assert unsupported_scope_reason(question)
    for question in ('找提到爱情的评论','找类似科幻电影里也会想家这种意思的影评','战争与反战主题的正面评论占比是多少？'):
        assert unsupported_scope_reason(question) is None
    assert unsupported_scope_reason('test',Filters(topic='爱情'))
    assert unsupported_scope_reason('test',Filters(topic='性别议题')) is None


def test_neutral_anger_uses_exact_label_and_no_unasked_threshold():
    plan=metric_query(QueryRequest(question='哪些电影中性评分里隐藏着愤怒？'),ResolvedFilters())
    assert plan.name=='neutral_anger'
    assert "c.emotion='愤怒'" in plan.query and '厌恶' not in plan.query
    assert 'neutral_total > 30' not in plan.query and 'anger_pct > 20' not in plan.query
    assert 'ORDER BY anger_count DESC' in plan.query
    topics=[{'name':entry['raw_name']} for entry in topic_config()['topics']]
    scope=resolve_filters('哪些电影中性评分里隐藏着愤怒？',Filters(),[],topics)
    assert not scope.topics
    scope=resolve_filters('哪些电影评分最高？',Filters(),[],topics)
    assert not scope.topics
    assert resolve_filters('性别话题讨论最多的电影',Filters(),[],topics).topics==[canonical_topic('t_6')]
    selection=metric_query(QueryRequest(question='战争话题下情感最负面的前5条'),resolve_filters('战争话题下情感最负面的前5条',Filters(),[],topics))
    assert selection.comment_selection and 'c.star ASC' in selection.query and selection.query.endswith('LIMIT 5')


def test_inferred_conflict_filters_explanation_but_never_the_denominator():
    request=QueryRequest(question='三块广告牌的矛盾评论占比是多少？结合至少两条评论解释评分与预测情绪可能不一致的原因。')
    scope=resolve_filters(request.question,request.filters,['三块广告牌 Three Billboards'],[])
    assert scope.consistency=='矛盾'
    plan=metric_query(request,scope)
    assert 'filter_layer_consistency' not in plan.parameters
    assert 'count(c) AS total' in plan.query
    _,explicit=filter_where(scope)
    assert explicit['filter_layer_consistency']=='矛盾'
    wrong={'movie':'三块广告牌 Three Billboards','layer_consistency':'一致'}
    assert 'layer_consistency' in comment_scope_violations(wrong,scope)


def test_semantic_five_star_and_predicted_negative_facets_are_enforced():
    scope=resolve_filters('找几条说法相似的评论：明知有缺点仍愿给它五星',Filters(),[],[])
    assert scope.min_star==scope.max_star==5
    assert 'min_star' in comment_scope_violations({'star':3},scope)
    bert=resolve_filters('评分是5星但BERT判断为负面情绪的评论有哪些？',Filters(),[],[])
    assert bert.min_star==5 and set(bert.emotions)=={'失望','愤怒','厌恶'}
    _,params=filter_where(bert)
    assert params['filter_emotions']==bert.emotions
    rejects=resolve_filters('找类似这部片不值五星的评论',Filters(),[],[])
    assert rejects.min_star is None and rejects.max_star is None


def test_wrong_fraction_is_corrected_to_confirmed_evidence():
    plan=metric_query(QueryRequest(question='哪些电影的隐性负面率最高？'),ResolvedFilters())
    rows=[{'movie':'电影A','hidden_negative':86,'total':219,'hidden_negative_pct':39.27}]
    evidence=[Evidence(id='g_1',kind='aggregate',source='graph',data={'rows':rows,**metric_metadata(plan,rows)})]
    answer,warnings,checks=validate_answer('隐性负面率最高？','graph','39.3%（219/86）。[g_1]',evidence)
    assert checks['corrected_answer'] and '86/219' in answer
    assert any(i['code']=='INVALID_FRACTION' for i in checks['issues'])
    assert warnings


def test_prediction_label_is_checked_without_reclassifying_comment():
    evidence=[Evidence(id='c_2',kind='comment',source='vector',data={'text':'我很感动','emotion':'愤怒','movie':'电影A','star':5})]
    for claim in ('预测情绪为感动。 [c_2]','预测情绪为感动 [c_2]'):
        bad,_,checks=validate_answer('观众情绪？','vector',claim,evidence)
        assert checks['corrected_answer'] and '预测情绪“愤怒”' in bad
    good,_,checks=validate_answer('观众情绪？','vector','从文本解读可看出感动，但预测情绪为愤怒 [c_2]',evidence)
    assert not checks['corrected_answer'] and '从文本解读' in good


def test_tied_leaders_are_not_reported_as_unique_first():
    evidence=[Evidence(id='g_1',kind='aggregate',source='graph',data={'rows':[{'movie':'电影A','positive_count':300,'tied_movie_count':86,'tied_movies':['电影A','电影B']}]})]
    answer,_,checks=validate_answer('正面评论最多的电影？','hybrid','电影A第一，300条。[g_1]',evidence)
    assert checks['corrected_answer'] and '86' in answer and '并列' in answer
    _,_,checks=validate_answer('正面评论最多的电影？','hybrid','86部电影并列最高300条；选电影A作代表。[g_1]',evidence)
    assert not checks['corrected_answer']


def test_semantic_few_comments_are_not_mistaken_for_corpus_count():
    evidence=[Evidence(id='c_1',kind='comment',source='vector',data={})]
    _,_,checks=validate_answer('找几条表达希望的影评','vector','希望 [c_1]',evidence)
    assert not checks['statistics_missing']
    _,_,checks=validate_answer('评论数量最多的电影','vector','只有评论 [c_1]',evidence)
    assert checks['statistics_missing']


@pytest.mark.asyncio
async def test_topic_tool_resolves_alias_and_rejects_invented_theme():
    class Graph:
        async def run(self,query,params):
            self.parameters=params
            return [{'topic':canonical_topic('t_6'),'movie':'电影A','comment_count':1}]
    graph=Graph()
    engine=SimpleNamespace(graph=graph,topics=[{'name':canonical_topic('t_6'),'keywords':'女人,男人,女性'}])
    registry=ToolRegistry(engine)
    outcome=await registry.execute('search_movies_by_topic',{'topic_keyword':'性别议题'},1)
    assert outcome['ok'] and graph.parameters['topic_keyword']==canonical_topic('t_6')
    assert outcome['rows'][0]['topic']=='性别议题'
    assert outcome['evidence'][0].data['rows'][0]['topic']==canonical_topic('t_6')
    rejected=await registry.execute('search_movies_by_topic',{'topic_keyword':'爱情片'},2)
    assert not rejected['ok'] and rejected['error']['code']=='UNKNOWN_TOPIC'
