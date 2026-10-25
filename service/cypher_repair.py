"""Bounded generated-query repair; every candidate uses the normal safety gate."""
import re
import time

from .safety import ScopeUnsupported, UnsafeCypher


class QueryFailure(RuntimeError):
    def __init__(self, error, stage):
        self.original=error
        self.stage=stage
        super().__init__(str(error))


def classify_failure(error):
    original=getattr(error,'original',error)
    code=str(getattr(original,'code',''))
    name=type(original).__name__
    message=str(original).lower()
    if isinstance(original,ScopeUnsupported):return 'scope'
    if isinstance(original,UnsafeCypher):return 'unsafe'
    if name=='SchemaMismatch':return 'schema'
    if 'SyntaxError' in code or 'syntax' in message:return 'syntax'
    if any(s in code+message for s in ('UnknownIdentifier','TypeError','property','relationshiptype','label does not exist')):return 'schema'
    if isinstance(original,TimeoutError) or 'TimedOut' in code or 'timeout' in message or 'timed out' in message:return 'timeout'
    if any(s in name+code for s in ('ServiceUnavailable','SessionExpired','Connection','AuthError','DatabaseUnavailable')):return 'infrastructure'
    return 'execution'


def schema_fragment(query):
    schema={
        'Movie':'Movie: name, movie_id, avg_star, total_reviews, avg_like.',
        'Comment':'Comment: comment_id,text,star,like_count,date,comment_len,sentiment,layer_consistency,topic_name. No bert_emotion property.',
        'Emotion':'Emotion: name,polarity,source. (Comment)-[:HAS_EMOTION]->(Emotion).',
        'Sentiment':'Sentiment: name,source. (Comment)-[:HAS_SENTIMENT]->(Sentiment).',
        'Topic':'Topic: topic_id,name,keywords,count. (Comment)-[:BELONGS_TO]->(Topic).',
        'User':'User: user_id,username. (User)-[:POSTED]->(Comment).',
    }
    labels={label for label in schema if re.search(r'\b'+label+r'\b',query)}|{'Comment'}
    return '\n'.join(schema[label] for label in schema if label in labels)+'\n(Comment)-[:REVIEWS]->(Movie). Only one read query; no CALL or writes; outer LIMIT <=50. Preserve all scopes and denominators.'


def _normalized(query):
    # Cypher identifiers and literal contents are case/whitespace sensitive.
    text=query.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:cypher)?\s*\n?', '', text, flags=re.I)
        text=re.sub(r'\s*```$', '', text)
    return text.strip()


async def run_with_repair(*,llm,graph,messages,usage,filters,max_rewrites,trace,redact):
    """Initial generation plus <=2 rewrites; an executed empty result stops.

    Safety/schema/scope checks still run before execution. Execution success is
    not semantic correctness: human grading remains a separate measurement.
    """
    messages=list(messages)
    unsafe_count=scope_count=0
    last_error=None
    seen=set()
    for attempt in range(max_rewrites+1):
        started=time.monotonic()
        try:
            completion=await llm.complete(messages,usage,max_tokens=1400)
        except Exception as exc:
            trace.append({'attempt':attempt+1,'status':'generation_failed','executed':False,
                'adopted':False,'failure_type':'infrastructure','error':redact(exc),
                'stop_reason':'generation_failed','elapsed_seconds':round(time.monotonic()-started,4)})
            raise
        generated=(completion.choices[0].message.content or '').strip()
        record={'attempt':attempt+1,'cypher':generated,'status':'pending','executed':False,'adopted':False,'elapsed_seconds':0.0}
        trace.append(record)
        if generated.strip('` \n')=='CANNOT_ANSWER':
            record.update(status='cannot_answer',elapsed_seconds=round(time.monotonic()-started,4))
            record['stop_reason']='unsupported_request'
            return None,[],True
        key=_normalized(generated)
        if key in seen:
            # Re-running an identical candidate cannot change the outcome, so it
            # is not executed; it still consumes one attempt of the budget.
            record.update(status='repeated',elapsed_seconds=round(time.monotonic()-started,4))
            kind=classify_failure(last_error) if last_error else 'execution'
            unsafe_count+=int(kind=='unsafe')
            scope_count+=int(kind=='scope')
            if attempt==max_rewrites or unsafe_count>=2 or scope_count>=2:
                record.update(failure_type=kind,stop_reason='repeated_candidate')
                raise last_error or RuntimeError('重复的查询候选')
            error='这条查询与之前失败的候选完全相同，没有执行。上一次的错误：'+redact(last_error)
        else:
            seen.add(key)
            try:
                safe,rows=await graph.generated(generated,filters=filters)
                record.update(safe_cypher=safe,executed=True,row_count=len(rows),status='ok' if rows else 'empty',elapsed_seconds=round(time.monotonic()-started,4))
                record.update(adopted=True,result_kind='rows' if rows else 'empty',
                    stop_reason='success' if rows else 'empty_result_preserved')
                return safe,rows,False
            except Exception as exc:
                kind=classify_failure(exc)
                last_error=exc
                record.update(status='failed',failure_type=kind,error=redact(exc),executed=getattr(exc,'stage',None)=='execution',elapsed_seconds=round(time.monotonic()-started,4))
                unsafe_count+=int(kind=='unsafe')
                scope_count+=int(kind=='scope')
                if attempt==max_rewrites or kind=='infrastructure' or unsafe_count>=2 or scope_count>=2:
                    record['stop_reason']=('infrastructure' if kind=='infrastructure' else 'unsafe_limit' if unsafe_count>=2
                        else 'scope_limit' if scope_count>=2 else 'repair_budget_exhausted')
                    raise
                error=redact(exc)
        record['failure_type']=kind
        record['stop_reason']='rewrite'
        messages.extend([{'role':'assistant','content':generated},{'role':'user','content':
            f'错误类型：{kind}。第{attempt+1}次查询需要核对。错误：{error}\n'
            f'相关schema：\n{schema_fragment(generated)}\n'
            '只输出修正后的只读Cypher，保留原始任务、固定条件与统计分母。时间问题应优化查询结构，不能截断聚合输入。不要重复输出与之前完全相同的查询。仍不支持就返回CANNOT_ANSWER。'}])
    raise RuntimeError('Unreachable repair state')
