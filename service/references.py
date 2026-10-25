"""Follow a displayed comment ordinal using real graph relationships."""
from .schemas import Evidence, QueryResult
from .routing import route


async def reference_result(engine,state,ordinal,request,request_id):
    strategy,_=route(request.question,request.strategy)
    result=QueryResult(request_id=request_id,strategy=strategy,status='cannot_answer',requested_strategy=request.strategy,routed_strategy=strategy)
    if request.strategy=='vector':
        result.answer='这项任务需要沿评论—作者关系查询，强制语义评论模式不支持；可选择自动或图谱。'
        return result
    if ordinal<1 or ordinal>len(state.last_results):
        result.answer='上一轮没有可对应的这条评论，请先查看结果序号或重新找评论。'
        return result
    reference=state.last_results[ordinal-1]
    if reference.get('kind')!='comment':
        result.answer='这个序号不是一条评论，无法定位其作者。'
        return result
    identifier=reference['id']
    if request.strategy=='auto':result.strategy='graph'
    result.warnings.append('本轮按所指评论的作者跨电影查询其他评论，不套用上一轮的电影、主题或星级条件；原会话条件仍保留供后续追问使用。')
    authors=await engine.graph.run('MATCH (u:User)-[:POSTED]->(c:Comment {comment_id:$id}) RETURN u.user_id AS user_id LIMIT 2',{'id':identifier})
    result.execution_paths=['graph_reference']
    if len(authors)!=1:
        result.answer='这条评论没有唯一可定位的作者关系，无法查询作者的其他评论。'
        return result
    query='MATCH (u:User {user_id:$user_id})-[:POSTED]->(c:Comment) WHERE c.comment_id<>$id RETURN c.comment_id AS comment_id ORDER BY c.like_count DESC,c.comment_id LIMIT $limit'
    rows=await engine.graph.run(query,{'user_id':authors[0]['user_id'],'id':identifier,'limit':request.top_k})
    result.cypher=[query]
    if not rows:
        result.status='no_results';result.answer='该作者在当前历史样本中没有其他已记录的评论；这不代表其在平台上没有其他发言。'
        result.evidence=[Evidence(id='g_1',kind='aggregate',source='reference',data={'rows':[],'reference_comment_id':identifier,'scope':'该评论作者在本样本中的其他评论'},cypher=query)]
        return result
    comments,graph=await engine.graph.comments([r['comment_id'] for r in rows])
    by_id={c['comment_id']:c for c in comments}
    result.evidence=[Evidence(id=r['comment_id'],kind='comment',source='reference',data=by_id[r['comment_id']]) for r in rows if r['comment_id'] in by_id]
    result.subgraph=graph
    result.status='ok'
    result.answer='以下是上一轮第'+str(ordinal)+'条评论作者在本样本中的其他评论（跨电影查找）：\n\n'+'\n\n'.join(
        f"{i}. {e.data.get('movie','')}：{e.data.get('text','')} [{e.id}]" for i,e in enumerate(result.evidence,1))
    return result
