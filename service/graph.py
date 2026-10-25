import json
import re
from typing import Any

from neo4j import AsyncGraphDatabase, Query, READ_ACCESS
from .cypher_repair import QueryFailure


class SchemaMismatch(ValueError):
    pass
from neo4j.graph import Node, Path, Relationship

from .routing import ResolvedFilters
from .safety import ScopeUnsupported, UnsafeCypher, ensure_read_summary, mask_literals, sanitize_cypher
from .schemas import Filters
from .settings import Settings
from .topics import decorate_topic, topic_display


def json_value(value: Any):
    if isinstance(value, Node):
        return {'_element_id': value.element_id, '_labels': sorted(value.labels), **{k: json_value(v) for k, v in value.items() if k != 'embedding'}}
    if isinstance(value, Relationship):
        return {'_element_id': value.element_id, '_type': value.type, 'source': value.start_node.element_id, 'target': value.end_node.element_id, **{k: json_value(v) for k, v in value.items()}}
    if isinstance(value, Path):
        return {'nodes': [json_value(n) for n in value.nodes], 'relationships': [json_value(r) for r in value.relationships]}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'iso_format'):
        return value.iso_format()
    return value


def collect_subgraph(rows: list[dict]) -> dict:
    nodes, edges = {}, {}
    def visit(value):
        if isinstance(value, dict):
            if '_labels' in value and '_element_id' in value:
                nodes[value['_element_id']] = {'id': value['_element_id'], 'labels': value['_labels'], 'properties': {k: v for k, v in value.items() if not k.startswith('_')}}
            elif '_type' in value and '_element_id' in value:
                edges[value['_element_id']] = {'id': value['_element_id'], 'type': value['_type'], 'source': value['source'], 'target': value['target'], 'properties': {k: v for k, v in value.items() if not k.startswith('_') and k not in ('source', 'target')}}
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(rows)
    # Relationships are shown only with their real endpoints.
    return {'nodes': list(nodes.values()), 'relationships': [e for e in edges.values() if e['source'] in nodes and e['target'] in nodes]}


def filter_where(scope: ResolvedFilters, alias: str = 'c') -> tuple[str, dict]:
    conditions = []
    parameters = {}
    if scope.movies:
        conditions.append(f'any(movie IN $filter_movies WHERE toLower({alias}.movie_name) CONTAINS toLower(movie))')
        parameters['filter_movies'] = scope.movies
    if scope.topics:
        conditions.append(f'any(topic IN $filter_topics WHERE toLower({alias}.topic_name) CONTAINS toLower(topic))')
        parameters['filter_topics'] = scope.topics
    if scope.sentiment:
        conditions.append(f'{alias}.sentiment = $filter_sentiment')
        parameters['filter_sentiment'] = scope.sentiment
    if scope.date_from:
        conditions.append(f'{alias}.date >= $filter_date_from')
        parameters['filter_date_from'] = scope.date_from
    if scope.date_to:
        conditions.append(f'{alias}.date <= $filter_date_to')
        parameters['filter_date_to'] = scope.date_to
    for property_name,value in [('emotion',scope.emotion),('layer_consistency',scope.consistency)]:
        if value:
            parameter='filter_'+property_name
            conditions.append(f'{alias}.{property_name} = ${parameter}')
            parameters[parameter]=value
    if scope.emotions:
        conditions.append(f'{alias}.emotion IN $filter_emotions')
        parameters['filter_emotions']=scope.emotions
    for property_name,operator,value in [('min_star','>=',scope.min_star),('max_star','<=',scope.max_star)]:
        if value is not None:
            parameter='filter_'+property_name
            conditions.append(f'{alias}.star {operator} ${parameter}')
            parameters[parameter]=value
    return ' AND '.join(conditions) or 'true', parameters


def scope_generated(query: str, filters: Filters) -> tuple[str, dict]:
    """Force explicit API filters into the first scoped MATCH before aggregation.

    Unrepresentable scope is rejected for one repair, never silently ignored.
    """
    if not filters.model_dump(exclude_none=True):
        return query,{}
    masked = mask_literals(query)
    if re.search(r'\(\s*:\s*Comment\b',masked):
        raise ScopeUnsupported('带筛选的生成查询不能含匿名 Comment：无法证明所有被统计评论都受筛选约束。请使用单一显式 Comment 范围；跨电影作者查询应使用受限模板。')
    if re.search(r'\bUNION\b',masked,re.I):
        raise ScopeUnsupported('筛选查询请改成单一 MATCH 范围，避免 UNION 绕过筛选')
    comments = list(re.finditer(r'\(\s*([A-Za-z_]\w*)\s*:\s*Comment\b',masked))
    movies = list(re.finditer(r'\(\s*([A-Za-z_]\w*)\s*:\s*Movie\b',masked))
    scope = ResolvedFilters([filters.movie] if filters.movie else [],[filters.topic] if filters.topic else [],filters.sentiment,filters.date_from,filters.date_to,True,filters.emotion,filters.consistency,filters.min_star,filters.max_star)
    non_movie = bool(filters.topic or filters.sentiment or filters.date_from or filters.date_to or filters.emotion or filters.consistency or filters.min_star is not None or filters.max_star is not None)
    if comments:
        aliases = {match.group(1) for match in comments}
        if len(aliases)!=1:
            raise ScopeUnsupported('筛选查询只允许一个显式 :Comment 变量；跨电影作者等多范围任务需要受限模板，不能改匿名节点或删除筛选绕过范围约束。')
        match = comments[0]
        predicate,params = filter_where(scope,match.group(1))
        if non_movie:
            for movie in movies:
                if re.search(r'\b'+re.escape(movie.group(1))+r'\s*\.\s*(?:avg_star|avg_like|total_reviews)\b',masked):
                    raise ScopeUnsupported('情感/主题/日期筛选后的统计必须从筛选 Comment 计算，不能使用 Movie 全库预计算值')
    elif filters.movie and not non_movie and movies:
        if len({m.group(1) for m in movies})!=1:
            raise ScopeUnsupported('请将电影筛选查询修复为单一 Movie 变量')
        match=movies[0]
        predicate=f'toLower({match.group(1)}.name) CONTAINS toLower($filter_movie)'
        params={'filter_movie':filters.movie}
    else:
        raise ScopeUnsupported('筛选查询必须使用显式 :Comment 标签；仅电影筛选也可使用 :Movie')
    # Only keywords at the outer query level can delimit this MATCH's predicate.
    depth=0
    top=list(' '*len(masked))
    depths=[]
    for i,char in enumerate(masked):
        depths.append(depth)
        if char in '([{': depth+=1
        elif char in ')]}': depth-=1
        elif depth==0: top[i]=char
    if depths[match.start()]!=0:
        raise ScopeUnsupported('请将带筛选节点移到最外层 MATCH，避免子查询范围遗漏')
    outer=''.join(top)
    clauses=[m for m in re.finditer(r'\b(?:WHERE|WITH|RETURN|OPTIONAL\s+MATCH|MATCH|UNWIND|ORDER\s+BY|LIMIT|SKIP)\b',outer,re.I)
             if not (m.group().upper()=='WITH' and re.search(r'\b(?:STARTS|ENDS)\s*$',outer[:m.start()],re.I))]
    following=[m for m in clauses if m.start()>match.end()]
    if not following:
        raise ScopeUnsupported('无法确定筛选作用域')
    first=following[0]
    if first.group().upper()=='WHERE':
        end=following[1].start() if len(following)>1 else len(query)
        query=query[:first.end()]+f' ({predicate}) AND ('+query[first.end():end].strip()+') '+query[end:]
    else:
        query=query[:first.start()]+f' WHERE {predicate} '+query[first.start():]
    return query,params


class GraphStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=10, max_transaction_retry_time=0)

    async def close(self):
        await self.driver.close()

    async def run(self, cypher: str, params: dict | None = None) -> list[dict]:
        async with self.driver.session(database=self.settings.neo4j_database, default_access_mode=READ_ACCESS) as session:
            result = await session.run(Query(cypher, timeout=self.settings.database_timeout), params or {})
            records = [{key:json_value(record[key]) for key in record.keys()} async for record in result]
            await result.consume()
            return records

    async def generated(self, cypher: str, params: dict | None = None, filters: Filters | None = None) -> tuple[str, list[dict]]:
        cypher=sanitize_cypher(cypher)
        if filters:
            cypher,forced=scope_generated(cypher,filters)
            params={**(params or {}),**forced}
        safe = sanitize_cypher(cypher)
        stage='preflight'
        try:
            async with self.driver.session(database=self.settings.neo4j_database, default_access_mode=READ_ACCESS) as session:
                plan = await session.run(Query('EXPLAIN ' + safe, timeout=self.settings.database_timeout), params or {})
                summary=await plan.consume()
                ensure_read_summary(summary)
                for notification in summary.notifications or []:
                    if any(name in str(notification.get('code','')) for name in ('UnknownPropertyKey','UnknownLabel','UnknownRelationshipType')):
                        raise SchemaMismatch(str(notification.get('description','未知schema字段')))
                stage='execution'
                result = await session.run(Query(safe, timeout=self.settings.database_timeout), params or {})
                rows = [{key:json_value(record[key]) for key in record.keys()} async for record in result]
                await result.consume()
        except Exception as exc:
            raise QueryFailure(exc,stage) from exc
        return safe, rows[:50]

    async def catalog(self):
        movies = await self.run('MATCH (m:Movie) RETURN m.name AS name ORDER BY name')
        topics = await self.run('MATCH (t:Topic) RETURN t.topic_id AS id,t.name AS name,t.keywords AS keywords ORDER BY toInteger(substring(t.topic_id,2))')
        return [m['name'] for m in movies], [decorate_topic(t) for t in topics]

    async def stats(self):
        rows = await self.run('MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label')
        rels = await self.run('MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type')
        return {'nodes': {r['label']: r['count'] for r in rows}, 'relationships': {r['type']: r['count'] for r in rels}, 'node_total': sum(r['count'] for r in rows), 'relationship_total': sum(r['count'] for r in rels)}

    async def ready(self):
        await self.driver.verify_connectivity()
        state = await self.run('MATCH (m:Movie) WHERE m._dataset_ready = true RETURN m._dataset_fingerprint AS fingerprint, m._import_manifest AS manifest LIMIT 1')
        if not state:
            return {'ready': False, 'reason': '数据初始化尚未完成'}
        indexes = await self.run('SHOW VECTOR INDEXES YIELD name, state, populationPercent WHERE name=$name RETURN name,state,populationPercent', {'name': self.settings.vector_index})
        online = bool(indexes and indexes[0]['state'] == 'ONLINE')
        return {'ready': online, 'dataset_fingerprint': state[0]['fingerprint'], 'index': indexes[0] if indexes else None, 'manifest': json.loads(state[0]['manifest'])}

    async def comments(self, ids: list[str]) -> tuple[list[dict], dict]:
        if not ids:
            return [], {'nodes': [], 'relationships': []}
        rows = await self.run('''
            MATCH (c:Comment) WHERE c.comment_id IN $ids
            OPTIONAL MATCH (c)-[rm:REVIEWS]->(m:Movie)
            OPTIONAL MATCH (c)-[rt:BELONGS_TO]->(t:Topic)
            OPTIONAL MATCH (c)-[rs:HAS_SENTIMENT]->(s:Sentiment)
            OPTIONAL MATCH (c)-[re:HAS_EMOTION]->(e:Emotion)
            OPTIONAL MATCH (u:User)-[ru:POSTED]->(c)
            RETURN c,m,t,s,e,u,rm,rt,rs,re,ru
        ''', {'ids': ids})
        candidates = []
        for row in rows:
            c = row['c']
            candidates.append({'comment_id': c['comment_id'], 'text': c['text'], 'movie': (row.get('m') or {}).get('name'), 'topic': (row.get('t') or {}).get('name'), 'topic_id':(row.get('t') or {}).get('topic_id'),'topic_display_name':topic_display((row.get('t') or {}).get('name')), 'star': c.get('star'), 'like_count': c.get('like_count'), 'date': c.get('date'), 'sentiment': (row.get('s') or {}).get('name'), 'emotion': (row.get('e') or {}).get('name'), 'emotion_source': (row.get('e') or {}).get('source'), 'emotion_confidence': (row.get('re') or {}).get('confidence'), 'layer_consistency': c.get('layer_consistency'), 'nodes': [row[k] for k in ('c','m','t','s','e','u') if row.get(k)], 'relationships': [row[k] for k in ('rm','rt','rs','re','ru') if row.get(k)]})
        return candidates, collect_subgraph(rows)

    async def liked_candidates(self, scope: ResolvedFilters, limit=20) -> list[dict]:
        where, params = filter_where(scope)
        rows = await self.run(f'MATCH (c:Comment) WHERE {where} RETURN c.comment_id AS comment_id ORDER BY c.like_count DESC, c.comment_id LIMIT $limit', {**params, 'limit': limit})
        return rows
