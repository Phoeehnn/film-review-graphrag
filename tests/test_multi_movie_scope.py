"""Multi-movie questions must keep every named movie in the resolved scope.

Regression for 1.4.1 Q44/C19: the alias 大圣归来 resolved first and the
short-title pass for 肖申克 was skipped, so a one-movie filter was then forced
into the graph query and the comparison silently lost one side.
"""
from service.routing import movie_catalog, resolve_filters
from service.schemas import Filters, QueryRequest
from service.statistics import metric_query

SHAWSHANK = '肖申克的救赎 The Shawshank Redemption'
MONKEY_KING = '西游记之大圣归来'


def movies_of(question):
    return resolve_filters(question, Filters(), movie_catalog(), []).movies


def test_alias_and_short_title_are_both_kept():
    assert movies_of('帮我对比肖申克和大圣归来的情绪指纹') == sorted([SHAWSHANK, MONKEY_KING])
    assert movies_of('分别给出肖申克和大圣归来的样本评论数量与平均星级，不合并两部电影。') == sorted([SHAWSHANK, MONKEY_KING])
    assert movies_of('肖申克的救赎和大圣归来哪部评分高') == sorted([SHAWSHANK, MONKEY_KING])


def test_single_movie_questions_are_unchanged():
    assert movies_of('大圣归来的正面率是多少') == [MONKEY_KING]
    assert movies_of('肖申克的矛盾评论占比是多少？') == [SHAWSHANK]
    assert movies_of('哪些电影的负面评论最多？') == []


def test_explicit_movie_filter_still_wins():
    scope = resolve_filters('对比肖申克和大圣归来', Filters(movie=MONKEY_KING), movie_catalog(), [])
    assert scope.movies == [MONKEY_KING]


def test_comparison_template_receives_both_movies():
    question = '帮我对比肖申克和大圣归来的情绪指纹'
    scope = resolve_filters(question, Filters(), movie_catalog(), [])
    plan = metric_query(QueryRequest(question=question, strategy='graph'), scope)
    assert plan is not None and plan.name == 'movie_rates_and_emotions'
    assert sorted(plan.parameters['filter_movies']) == sorted([SHAWSHANK, MONKEY_KING])
