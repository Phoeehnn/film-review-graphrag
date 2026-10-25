"""Reviewed metric definitions; free-form questions still use Text-to-Cypher.

These rules describe statistics, never memorize benchmark answers or movie lists.
Every template runs against the same current graph, scope, timeout and EXPLAIN.
"""
import re
from dataclasses import dataclass

from .graph import filter_where


NEGATIVE_EMOTIONS=['失望','愤怒','厌恶']
RANK_COLUMNS={
    'neutral_anger':(('movie','电影'),('anger_count','中性评论中预测愤怒数'),('neutral_total','中性评论数'),('anger_pct','比例 %')),
    'hidden_negative_rate':(('movie','电影'),('hidden_negative','隐性负面数'),('total','样本评论数'),('hidden_negative_pct','比例 %')),
    'contradiction_rate':(('movie','电影'),('contradiction_count','矛盾标签数'),('total','样本评论数'),('contradiction_pct','比例 %')),
    'movie_similarity':(('movie','电影'),('topic_cosine_similarity','18 主题余弦相似度')),
    'topic_counts':(('topic','主题原名'),('comment_count','样本评论数')),
}


def render_ranked_table(plan_name,rows,definition,evidence_id='g_1'):
    """Preserve database order and values instead of asking a model to restate them."""
    columns=RANK_COLUMNS.get(plan_name)
    if not columns or not rows:
        return ''
    header='| 排名 | '+' | '.join(label for _,label in columns)+' | 证据 |'
    rule='|---:|'+'|'.join('---:' if key not in ('movie','topic') else '---' for key,_ in columns)+'|---|'
    lines=[header,rule]
    for position,row in enumerate(rows,1):
        values=[str(row.get(key,'—')).replace('|','\\|').replace('\n',' ') for key,_ in columns]
        lines.append(f'| {position} | '+' | '.join(values)+f' | [{evidence_id}] |')
    lines+=['',definition+f' [{evidence_id}]']
    return '\n'.join(lines)


@dataclass
class MetricQuery:
    name: str
    query: str
    parameters: dict
    definition: str
    rates: tuple = ()
    comment_selection: bool = False

    def __post_init__(self):
        if 'LIMIT $limit' in self.query:
            limit=self.parameters.pop('limit')
            if type(limit) is not int or not 1<=limit<=50:
                raise ValueError('已审阅统计模板的结果上限必须为1–50整数')
            # Only a validated integer is compiled; all data values stay parameters.
            self.query=self.query.replace('LIMIT $limit',f'LIMIT {limit}')


def requires_statistics(question):
    return bool(re.search(r'数量|多少(?:部|条|个)|几部|最多|最高|最低|排名|排行|比例|占比|正面率|负面率|矛盾率|隐性负面率|分布|趋势|按月|平均|统计|情绪指纹|点赞.{0,3}前|相似.{0,3}电影',question))


def requested_limit(question,default=10):
    match=re.search(r'(?:前|最高的|最多的)\s*(\d+)\s*(?:条|部|名)?',question)
    return min(50,max(1,int(match.group(1)))) if match else default


def metric_query(request,scope):
    q=request.question
    if re.search(r'相似.{0,3}电影|类似.{0,3}电影',q) and len(scope.movies)==1 and not request.filters.model_dump(exclude={'movie'},exclude_none=True):
        # Thesis §5.1 defines movie similarity using the 18-topic proportions.
        # Movie properties "0".."17" are the imported proportions, not counts.
        query='''MATCH (target:Movie {name:$target_movie})
            MATCH (other:Movie) WHERE other<>target AND other.total_reviews>50
            WITH target,other,
                [i IN range(0,17) | coalesce(toFloat(target[toString(i)]),0.0)] AS source_topics,
                [i IN range(0,17) | coalesce(toFloat(other[toString(i)]),0.0)] AS other_topics
            WITH target,other,
                reduce(dot=0.0,i IN range(0,17) | dot+source_topics[i]*other_topics[i]) AS dot,
                reduce(norm=0.0,i IN range(0,17) | norm+source_topics[i]*source_topics[i]) AS source_norm,
                reduce(norm=0.0,i IN range(0,17) | norm+other_topics[i]*other_topics[i]) AS other_norm
            WITH target,other,CASE WHEN source_norm=0 OR other_norm=0 THEN 0.0
                ELSE dot/sqrt(source_norm*other_norm) END AS similarity
            RETURN target.name AS target_movie,other.name AS movie,
                round(similarity,4) AS topic_cosine_similarity
            ORDER BY topic_cosine_similarity DESC,movie LIMIT $limit'''
        return MetricQuery('movie_similarity',query,{'target_movie':scope.movies[0],'limit':requested_limit(q)},
            '按论文第5.1节的18维评论主题比例向量计算余弦相似度；排除目标电影和样本评论不超过50条的电影。0至1表示主题分布相似程度，不是观众评分或题材真值。')
    # Inferred numerator labels must not accidentally restrict the denominator.
    # Explicit UI/API filters remain in force; the returned scope documents them.
    from dataclasses import replace
    base=replace(scope)
    if not request.filters.sentiment and re.search(r'正面.*(?:占比|比例|最多|有多少)|正面率|负面率',q):
        base.sentiment=None
    if not request.filters.consistency and re.search(r'矛盾.*(?:占比|比例)|矛盾率|一致性.*(?:比例|统计)|各占',q):
        base.consistency=None
    where,params=filter_where(base)
    match=f'MATCH (c:Comment)-[:REVIEWS]->(m:Movie) WHERE {where}'
    limit=requested_limit(q)

    if len(scope.movies)==1 and re.search(r'评论过.+作者',q) and re.search(r'(?:还|也).*评论过(?:其他|别的)电影',q):
        # Source comments obey the complete current scope; the author's other
        # movie is intentionally a separate range, not an anonymous bypass.
        query=match+''' MATCH (u:User)-[:POSTED]->(c)
            WITH DISTINCT u
            MATCH (u)-[:POSTED]->(other_comment:Comment)-[:REVIEWS]->(other:Movie)
            WHERE other.name <> $source_movie
            RETURN count(DISTINCT u) AS distinct_authors LIMIT 1'''
        return MetricQuery('authors_other_movies',query,{**params,'source_movie':scope.movies[0]},
            '先找指定电影和当前筛选范围内评论的不同作者，再统计其中也评论过其他电影的不同作者。其他电影评论不继承源电影筛选；范围限于本样本，作者以实际User节点为准。')

    if re.search(r'正面评论(?:数|数量)?最多|最多的正面评论',q):
        query=match+''' WITH m,count(CASE WHEN c.sentiment='正面' THEN 1 END) AS positive_count,
            max(CASE WHEN c.sentiment='正面' THEN c.like_count ELSE 0 END) AS max_positive_like
            WHERE positive_count>0
            WITH m,positive_count,max_positive_like ORDER BY positive_count DESC,max_positive_like DESC,m.name
            WITH collect({movie:m.name,positive_count:positive_count}) AS candidates WHERE size(candidates)>0
            WITH candidates,head(candidates).positive_count AS maximum
            WITH [x IN candidates WHERE x.positive_count=maximum] AS ties
            RETURN head(ties).movie AS movie,head(ties).positive_count AS positive_count,
                size(ties) AS tied_movie_count,[x IN ties | x.movie] AS tied_movies LIMIT 1'''
        return MetricQuery('positive_count_leaders',query,params,
            '在指定样本范围按正面评论数量取最高值，完整保留并列名单；如需解释，只从并列者中选一部代表电影，不称其唯一第一。分层抽样配额会产生并列。')

    if re.search(r'中性(?:评分|评论).*愤怒|愤怒.*中性(?:评分|评论)',q) and not re.search(r'厌恶|失望',q):
        query=match+''' WITH m,count(CASE WHEN c.sentiment='中性' THEN 1 END) AS neutral_total,
            count(CASE WHEN c.sentiment='中性' AND c.emotion='愤怒' THEN 1 END) AS anger_count
            WHERE anger_count>0
            RETURN m.name AS movie,anger_count,neutral_total,
                round(100.0*anger_count/neutral_total,2) AS anger_pct
            ORDER BY anger_count DESC,movie LIMIT $limit'''
        return MetricQuery('neutral_anger',query,{**params,'limit':limit},
            '中性评分为星级派生情感=中性；这里只计预测愤怒，不并入厌恶或失望，不附加未要求的样本数/比例门槛。按愤怒评论数降序，比例分母为该片中性评论数。',
            (('anger_count','neutral_total','anger_pct','中性评分中预测愤怒'),))

    if '隐性负面率' in q:
        query=match+''' WITH m,count(c) AS total,
            count(CASE WHEN c.sentiment='中性' AND c.emotion IN $negative_emotions THEN 1 END) AS hidden_negative
            RETURN m.name AS movie,hidden_negative,total,
                round(100.0*hidden_negative/total,2) AS hidden_negative_pct
            ORDER BY hidden_negative_pct DESC,hidden_negative DESC,movie LIMIT $limit'''
        return MetricQuery('hidden_negative_rate',query,{**params,'negative_emotions':NEGATIVE_EMOTIONS,'limit':limit},
            '隐性负面定义为星级派生情感=中性且预测情绪为失望/愤怒/厌恶。默认分母为该片在指定范围的全部评论，非中性评论数；这是一种公开定义，并非人工负面真值。',
            (('hidden_negative','total','hidden_negative_pct','隐性负面'),))

    if re.search(r'矛盾(?:评论)?(?:占比|比例)|矛盾率',q):
        query=match+''' WITH m,count(c) AS total,
            count(CASE WHEN c.layer_consistency='矛盾' THEN 1 END) AS contradiction_count
            RETURN m.name AS movie,contradiction_count,total,
                round(100.0*contradiction_count/total,2) AS contradiction_pct
            ORDER BY contradiction_pct DESC,movie LIMIT $limit'''
        return MetricQuery('contradiction_rate',query,{**params,'limit':limit},
            '矛盾评论由已有layer_consistency=矛盾标记确定；分母为该片指定范围的全部评论。解释用的评论也须具有矛盾标记，预测情绪不是人工真值。',
            (('contradiction_count','total','contradiction_pct','矛盾评论'),))

    if re.search(r'一致性.*(?:统计|比例)|一致.*矛盾.*可接受.*各占',q):
        query=match+''' WITH collect(c) AS comments,count(c) AS total
            UNWIND comments AS c
            RETURN c.layer_consistency AS consistency,count(c) AS comment_count,total,
                round(100.0*count(c)/total,2) AS consistency_pct
            ORDER BY comment_count DESC,consistency LIMIT 50'''
        return MetricQuery('consistency_distribution',query,params,
            '一致/可接受/矛盾为已有双层分析标记，分母为指定范围全部评论。',
            (('comment_count','total','consistency_pct','双层一致性'),))

    positive_ratio=bool(re.search(r'正面.*(?:占比|比例)|正面率|负面率',q))
    emotion_distribution=bool(re.search(r'情绪(?:分布|指纹)|主要.{0,5}情绪',q))
    if positive_ratio or emotion_distribution:
        common=match+''' WITH m,collect(c) AS comments,count(c) AS total,
            count(CASE WHEN c.sentiment='正面' THEN 1 END) AS positive_count,
            count(CASE WHEN c.sentiment='负面' THEN 1 END) AS negative_count'''
        fields='''m.name AS movie,total,positive_count,negative_count,
                round(100.0*positive_count/total,2) AS positive_pct,
                round(100.0*negative_count/total,2) AS negative_pct'''
        rates=(('positive_count','total','positive_pct','正面评论'),('negative_count','total','negative_pct','负面评论'))
        if emotion_distribution:
            query=common+' UNWIND comments AS c RETURN '+fields+''',c.emotion AS emotion,
                count(c) AS emotion_count,round(100.0*count(c)/total,2) AS emotion_pct
                ORDER BY movie,emotion_count DESC,emotion LIMIT 50'''
            rates+= (('emotion_count','total','emotion_pct','预测情绪'),)
        else:
            query=common+' RETURN '+fields+' ORDER BY positive_pct DESC,movie LIMIT 50'
        return MetricQuery('movie_rates_and_emotions',query,params,
            '正面/负面比例与情绪分布的分母均为该片指定范围的全部评论，先得到分母再按预测情绪分组。星级派生情感与预测情绪分别报告。',rates)

    if re.search(r'主要讨论哪些主题|每个主题.*多少|各主题.*评论(?:数量|数)',q):
        query=match+''' MATCH (c)-[:BELONGS_TO]->(t:Topic)
            RETURN t.topic_id AS topic_id,t.name AS topic,count(c) AS comment_count
            ORDER BY comment_count DESC,topic_id LIMIT 18'''
        return MetricQuery('topic_counts',query,params,
            '对指定范围评论的既有BERTopic归属计数，按评论数降序；别名仅用于显示，不改变聚类归属。')

    semantic=bool(re.search(r'相似|相同意思|说法|措辞|找.{0,8}表达',q))
    comment_request='评论' in q or bool(re.search(r'前\s*[0-9一二三四五六七八九十]+\s*条',q))
    if not semantic and comment_request and re.search(r'最负面|最消极|点赞(?:最高|最多)|矛盾(?:情感|评论)|低评分.*喜爱|5星.*负面情绪',q):
        conditions=[]
        if '矛盾' in q:
            conditions.append("c.layer_consistency='矛盾'")
        if re.search(r'5星.*负面情绪',q):
            conditions.extend(['c.star=5','c.emotion IN $negative_emotions'])
            params['negative_emotions']=NEGATIVE_EMOTIONS
        if re.search(r'低评分.*喜爱',q):
            conditions.extend(['c.star<=2',"c.emotion='喜爱'"])
        negative_order=bool(re.search(r'最负面|最消极',q))
        if negative_order:
            conditions.append("c.sentiment='负面'")
        condition=' AND '+' AND '.join(conditions) if conditions else ''
        order='c.star ASC,c.like_count DESC,c.comment_id' if negative_order else 'c.like_count DESC,c.comment_id'
        query=match+condition+''' RETURN c.comment_id AS comment_id,m.name AS movie,c.star AS star,
            c.sentiment AS sentiment,c.emotion AS emotion,c.like_count AS like_count,
            c.layer_consistency AS layer_consistency,c.topic_name AS topic
            ORDER BY '''+order+' LIMIT $limit'
        definition='按样本点赞数降序选取真实评论，不附加未要求的点赞门槛。'
        if negative_order:
            definition='最负面按已有星级情感筛选负面评论，再以星级升序、点赞降序排序；没有连续情绪强度真值。'
        return MetricQuery('comment_selection',query,{**params,'limit':requested_limit(q,request.top_k)},definition,comment_selection=True)
    return None


def metric_metadata(plan,rows):
    if plan is None:
        return {'method':'text_to_cypher','validation_scope':'查询安全与证据引用；自由生成统计的业务定义仍需复核'}
    facts=[]
    seen=set()
    for row in rows:
        for numerator,denominator,percent,label in plan.rates:
            if all(key in row for key in (numerator,denominator,percent)):
                fact={'movie':row.get('movie'),'label':row.get('emotion',row.get('consistency',label)) if label in ('预测情绪','双层一致性') else label,
                    'numerator':row[numerator],'denominator':row[denominator],'percent':row[percent]}
                key=tuple(fact.values())
                if key not in seen:
                    facts.append(fact)
                    seen.add(key)
    return {'method':'reviewed_metric','metric':plan.name,'definition':plan.definition,
            'parameters':plan.parameters,'rate_facts':facts,'contains_statistics':True,
            'statistics_kind':'ordered_comments' if plan.comment_selection else 'aggregate'}
