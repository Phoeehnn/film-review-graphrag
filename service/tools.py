"""The original five fixed Cypher tools plus semantic comment search."""
from pydantic import Field, ValidationError

from .engine import merge_subgraphs
from .routing import comment_scope_violations, resolve_filters
from .schemas import Evidence, Filters, ShortText, StrictModel, Text
from .topics import canonical_topic, decorate_topic, friendly_value, topic_definition, unsupported_scope_reason


class MovieArgs(StrictModel):
    movie_name: ShortText


class EmptyArgs(StrictModel):
    pass


class TopicArgs(StrictModel):
    topic_keyword: ShortText
    limit: int = Field(default=5,ge=1,le=10)


class RatingArgs(StrictModel):
    min_reviews: int = Field(default=50,ge=0,le=1000000)
    limit: int = Field(default=5,ge=1,le=10)


class SearchArgs(StrictModel):
    query: Text
    top_k: int = Field(default=5,ge=1,le=20)
    filters: Filters = Field(default_factory=Filters)


TOOL_MODELS = {'get_movie_overview':MovieArgs,'get_movie_emotions':MovieArgs,'list_topics':EmptyArgs,'search_movies_by_topic':TopicArgs,'get_top_movies_by_rating':RatingArgs,'search_comments':SearchArgs}
DESCRIPTIONS = {
    'get_movie_overview':'查询电影五分制平均星级、平均点赞和样本评论数。',
    'get_movie_emotions':'查询电影七类预测情绪的样本分布。',
    'list_topics':'列出18个主题的显示名称、编号、原始名称、范围说明和关键词；涉及主题时先调用本工具定位。评论主题不是电影题材。',
    'search_movies_by_topic':'按主题编号、显示别名或原始关键词查询该评论主题下的电影、样本评论数和平均星级。不能替代电影题材分类。',
    'get_top_movies_by_rating':'查询达到最少样本评论数的电影平均星级排名。',
    'search_comments':'语义检索评论证据；支持电影、主题编号/别名、情感、预测情绪、日期、星级和双层一致性筛选，不能计算全库统计。',
}
TOOL_DEFINITIONS = [{'type':'function','function':{'name':name,'description':DESCRIPTIONS[name],'parameters':model.model_json_schema()}} for name,model in TOOL_MODELS.items()]

QUERIES = {
    'get_movie_overview': '''MATCH (m:Movie) WHERE toLower(m.name) CONTAINS toLower($movie_name)
        OPTIONAL MATCH (c:Comment)-[:REVIEWS]->(m)
        WITH m,count(c) AS counted_reviews,avg(c.like_count) AS mean_like
        RETURN m.name AS movie, round(m.avg_star,2) AS avg_star,round(mean_like,2) AS avg_like,
        coalesce(m.total_reviews,counted_reviews) AS total_reviews ORDER BY total_reviews DESC LIMIT 5''',
    'get_movie_emotions': '''MATCH (m:Movie)<-[:REVIEWS]-(c:Comment)-[:HAS_EMOTION]->(e:Emotion)
        WHERE toLower(m.name) CONTAINS toLower($movie_name)
        RETURN m.name AS movie,e.name AS emotion,count(c) AS comment_count,e.source AS prediction_source
        ORDER BY movie,comment_count DESC LIMIT 50''',
    'list_topics': '''MATCH (t:Topic) RETURN t.topic_id AS id,t.name AS name,t.keywords AS keywords,t.count AS comment_count
        ORDER BY toInteger(substring(t.topic_id,2)) LIMIT 18''',
    'search_movies_by_topic': '''MATCH (m:Movie)<-[:REVIEWS]-(c:Comment)-[:BELONGS_TO]->(t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($topic_keyword) OR toLower(t.keywords) CONTAINS toLower($topic_keyword)
        RETURN m.name AS movie,t.name AS topic,count(c) AS comment_count,round(avg(c.star),2) AS avg_star
        ORDER BY comment_count DESC LIMIT $limit''',
    'get_top_movies_by_rating': '''MATCH (m:Movie) WHERE m.total_reviews >= $min_reviews
        RETURN m.name AS movie,round(m.avg_star,2) AS avg_star,m.total_reviews AS total_reviews
        ORDER BY avg_star DESC,total_reviews DESC LIMIT $limit''',
}


def validate_tool(name, arguments):
    model = TOOL_MODELS.get(name)
    if model is None:
        return None, {'code':'UNKNOWN_TOOL','message':f'未知工具 {name}','allowed_tools':list(TOOL_MODELS)}
    if not isinstance(arguments,dict):
        return None, {'code':'INVALID_ARGUMENTS','message':'工具参数必须为 JSON 对象'}
    try:
        return model.model_validate(arguments),None
    except ValidationError as exc:
        return None, {'code':'INVALID_ARGUMENTS','message':'参数校验失败，请按 schema 修正','details':exc.errors(include_input=False,include_context=False)}


class ToolRegistry:
    def __init__(self, engine):
        self.engine = engine

    async def execute(self, name, arguments, sequence):
        parsed,error = validate_tool(name,arguments)
        if error:
            return {'ok':False,'error':error,'evidence':[],'cypher':[],'subgraph':{'nodes':[],'relationships':[]}}
        if name == 'search_comments':
            unsupported=unsupported_scope_reason(parsed.query,parsed.filters)
            if unsupported:
                return {'ok':False,'error':{'code':'UNSUPPORTED_SCOPE','message':unsupported},'evidence':[],'cypher':[],'subgraph':{'nodes':[],'relationships':[]}}
            scope = resolve_filters(parsed.query,parsed.filters,self.engine.movies,self.engine.topics)
            candidates = await self.engine.retriever.vector(parsed.query,scope,parsed.top_k)
            comments,subgraph = await self.engine.graph.comments([c['comment_id'] for c in candidates])
            scores = {c['comment_id']:c.get('similarity') for c in candidates}
            by_id = {c['comment_id']:c for c in comments}
            for comment in comments:
                if comment_scope_violations(comment,scope):
                    raise ValueError('评论证据未满足已解析的工具筛选范围')
            evidence = [Evidence(id=identifier,kind='comment',source='agent_vector',data=by_id[identifier],score=scores[identifier]) for identifier in scores if identifier in by_id]
            return {'ok':True,'rows':[{'evidence_id':e.id,**friendly_value({k:v for k,v in e.data.items() if k not in ('nodes','relationships','text')}),'text':e.data.get('text','')} for e in evidence],'evidence':evidence,'cypher':[],'subgraph':subgraph,'filters':scope.params()}
        query = QUERIES[name]
        parameters = parsed.model_dump()
        if name=='search_movies_by_topic':
            entry=topic_definition(parsed.topic_keyword)
            if entry:
                parameters['topic_keyword']=entry['raw_name']
            elif not any(parsed.topic_keyword in str(t.get('keywords','')) or parsed.topic_keyword in t['name'] for t in self.engine.topics):
                return {'ok':False,'error':{'code':'UNKNOWN_TOPIC','message':'实际主题名称与关键词中没有这个表达，请先list_topics选择现有主题；电影题材尚无标签。'},'evidence':[],'cypher':[],'subgraph':{'nodes':[],'relationships':[]}}
        # Fixed, reviewed, parameterized queries. READ mode chooses routing only;
        # the generated-Cypher validator/EXPLAIN boundary lives in GraphStore.
        rows = await self.engine.graph.run(query,parameters)
        if name=='list_topics':
            rows=[decorate_topic(row) for row in rows]
        evidence = [Evidence(id=f'g_{sequence}',kind='aggregate',source=name,data={'rows':rows,'parameters':parameters,'scope':'历史样本内结果'},cypher=query)] if rows else []
        return {'ok':True,'rows':friendly_value(rows),'evidence':evidence,'cypher':[query],'subgraph':{'nodes':[],'relationships':[]},'evidence_ids':[e.id for e in evidence],
            'scope_note':'按指定评论主题/电影的历史样本统计；评论主题不等于电影题材。'}
