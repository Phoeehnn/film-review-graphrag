"""Independent CSV reference facts for human review; no model calls or grades."""
import json
from pathlib import Path

import pandas as pd

from service.settings import Settings


def records(frame,limit=10):
    if isinstance(frame,pd.Series): frame=frame.rename('count').reset_index()
    return json.loads(frame.head(limit).to_json(orient='records',force_ascii=False))


def main():
    s=Settings()
    d=pd.read_csv(s.data_dir/'comments.csv',keep_default_na=False)
    m=pd.read_csv(s.data_dir/'movies.csv',keep_default_na=False)
    topics=pd.read_csv(s.data_dir/'topics.csv',keep_default_na=False)
    users=pd.read_csv(s.data_dir/'users.csv',keep_default_na=False)
    d['Star']=pd.to_numeric(d['Star'])
    d['Like']=pd.to_numeric(d['Like'])
    total=len(d)
    emotion='bert_emotion_7class'
    def film(part): return d[d.Movie_Name.str.contains(part,regex=False)]
    def share(frame,column):
        counts=frame[column].value_counts().rename('count').reset_index()
        counts['percent']=counts['count']*100/len(frame) if len(frame) else 0
        return {'denominator':len(frame),'distribution':records(counts,50)}
    def ranked(frame,metric='count'):
        result=frame.groupby('Movie_Name').agg(count=('comment_id','size'),avg_star=('Star','mean'),avg_like=('Like','mean')).reset_index()
        return records(result.sort_values([metric,'Movie_Name'],ascending=[False,True]))
    def comments(frame,limit=10):
        cols=['comment_id','Movie_Name','Comment','Star','Like','sentiment',emotion,'layer_consistency','topic_name','Date']
        return records(frame.sort_values(['Like','comment_id'],ascending=[False,True])[cols],limit)
    def topic(term): return d[d.topic_name.str.contains(term,regex=False)]
    shaw,boards=film('肖申克'),film('三块广告牌')
    conflict=d[d.layer_consistency.eq('矛盾')]
    negative=['愤怒','失望','厌恶']
    facts={}
    facts[1]={'scope':'历史样本内五分制平均星级与情感分布','basic':ranked(shaw),'sentiment':share(shaw,'sentiment')}
    facts[2]=ranked(d,'avg_star')
    facts[3]=ranked(d)
    facts[4]={'comment_count':len(boards)}
    facts[5]={'avg_star_5point':float(film('乘风破浪').Star.mean())}
    facts[6]=ranked(d,'avg_like')
    facts[7]={'movies':len(m),'nodes':176650,'relationships':618436,'comments':total}
    facts[8]=ranked(film('逆转未来'))
    facts[9]=ranked(film('一出好戏'))
    facts[10]=records(m[m.total_reviews>1000].sort_values('total_reviews',ascending=False)[['movie_name','total_reviews']],50)
    facts[11]={'positive':int(shaw.sentiment.eq('正面').sum()),'total':len(shaw),'positive_percent':float(shaw.sentiment.eq('正面').mean()*100)}
    facts[12]=ranked(d[d.sentiment.eq('负面')])
    facts[13]=share(film('战狼2'),emotion)
    anger=d.groupby('Movie_Name')[emotion].apply(lambda values:float(values.eq('愤怒').mean()*100)).rename('anger_percent').reset_index()
    facts[14]=records(anger.sort_values('anger_percent',ascending=False))
    facts[15]=ranked(d[d[emotion].eq('感动')])
    facts[16]=share(shaw,emotion)
    facts[17]=records(d.groupby(emotion).Like.agg(['mean','count']).reset_index().sort_values('mean',ascending=False),50)
    daughter=film('女儿国');facts[18]=share(daughter[daughter[emotion].isin(negative)],emotion)
    facts[19]=ranked(d[d[emotion].eq('惊喜')])
    facts[20]=share(film('一出好戏'),'sentiment')
    facts[21]=share(shaw,'topic_name')
    facts[22]=ranked(topic('战争')[lambda frame:frame.sentiment.eq('正面')])
    facts[23]=ranked(topic('喜剧'),'avg_star')
    facts[24]=ranked(topic('女人'))
    facts[25]={'definition':'最受欢迎应说明按评论数、正面率、星级还是点赞排序；下表为评论数','rank':ranked(topic('音乐'))}
    controversy=boards.groupby('topic_name').Star.agg(['count','mean','std']).reset_index().sort_values('std',ascending=False)
    facts[26]={'definition':'争议应说明定义，例如星级标准差；少样本主题须注明','star_std_reference':records(controversy,18)}
    facts[27]=ranked(topic('导演'))
    facts[28]=ranked(topic('青春'))
    facts[29]={'topics':len(topics)}
    facts[30]=records(topics[['topic_id','topic_name','count']],18)
    facts[31]=comments(conflict)
    facts[32]=share(d,'layer_consistency')
    hidden=d[d.sentiment.eq('中性')&d[emotion].eq('愤怒')]
    facts[33]=ranked(hidden)
    facts[34]=comments(d[d.Star.eq(5)&d[emotion].isin(negative)])
    facts[35]=comments(conflict)
    hidden_all=d[d.sentiment.eq('中性')&d[emotion].isin(negative)]
    total_by_movie=d.groupby('Movie_Name').size()
    hidden_ratio=(hidden_all.groupby('Movie_Name').size()/total_by_movie*100).fillna(0).rename('hidden_negative_percent_of_all_comments').reset_index()
    facts[36]={'definition':'隐性负面 = 中性星级且预测负面情绪；可选择全片或中性评论作分母，必须明确','all_comment_denominator':records(hidden_ratio.sort_values(hidden_ratio.columns[-1],ascending=False))}
    facts[37]=share(shaw,'layer_consistency')
    facts[38]=share(boards,'layer_consistency')
    facts[39]=comments(d[d.Star.le(2)&d[emotion].eq('喜爱')])
    ratio=(conflict.groupby('Movie_Name').size()/total_by_movie*100).fillna(0).rename('contradiction_percent').reset_index()
    facts[40]=records(ratio.sort_values('contradiction_percent',ascending=False))
    user_relation=pd.read_csv(s.data_dir/'rel_user_comment.csv')
    joined=user_relation.merge(d[['comment_id','Like']],on='comment_id').groupby('user_id').Like.sum().rename('total_likes').reset_index().merge(users[['user_id','username']],on='user_id')
    facts[41]={'definition':'影响力按评论获赞总和；其他定义需说明','total_likes_rank':records(joined.sort_values('total_likes',ascending=False))}
    facts[42]=comments(d[d.Like.gt(1000)],20)
    facts[43]={'definition':'电影相似性没有唯一标准，应说明主题/情绪比例特征与相似度算法，且必须有数据库计算证据；评论向量近邻不能直接当作电影排名。'}
    facts[44]={'肖申克':share(shaw,emotion),'大圣归来':share(film('大圣归来'),emotion)}
    war=topic('战争')
    pivot=war.groupby(['Movie_Name',emotion]).size().unstack(fill_value=0)
    denominators=pivot.sum(axis=1)
    pivot['total']=denominators
    for name in ['感动','愤怒']:
        pivot[name+'_percent']=pivot.get(name,0)*100/denominators
    facts[45]=records(pivot.reset_index(),50)
    facts[46]={'limitation':'主题目录没有名为爱情的主题；需说明用主题关键词/评论文本/其他定义映射爱情，而非凭空使用不存在的标签。','available_topics':records(topics[['topic_name','keywords']],18)}
    facts[47]=comments(d,10)
    facts[48]=records(users.sort_values('review_count',ascending=False)[['user_id','username','review_count']],10)
    monthly=boards.copy();monthly['month']=monthly.Date.str[:7]
    month_table=monthly.groupby('month').agg(total=('comment_id','size'),positive=('sentiment',lambda values:int(values.eq('正面').sum()))).reset_index()
    month_table['positive_percent']=month_table.positive*100/month_table.total
    facts[49]=records(month_table.sort_values('month'),50)
    facts[50]=comments(shaw,10)
    for identifier in (51,52,53):
        facts[identifier]={'semantic_review':'按固定语义标准人工判断评论是否同时满足两个要点；不能仅按 has_results 或词命中判 Y。'}
    facts[54]={'sentiment':share(shaw,'sentiment'),'emotion':share(shaw,emotion),'comments_requirement':'至少两条实际相关评论，引用真实 ID；图谱分母应为 786。'}
    facts[55]={'consistency':share(boards,'layer_consistency'),'contradictory_comments':comments(boards[boards.layer_consistency.eq('矛盾')],10)}
    leader=d[d.sentiment.eq('正面')].Movie_Name.value_counts()
    maximum=int(leader.iloc[0])
    ties=sorted(leader[leader.eq(maximum)].index.tolist())
    name=ties[0]
    facts[56]={'positive_count':maximum,'tied_movie_count':len(ties),'tied_movies':ties,
               'interpretation':'最高数量存在并列；并列电影均满足数量条件。样本有分层配额，不能据此推断全平台热门程度。',
               'example_movie':name,'candidate_comments':comments(d[d.Movie_Name.eq(name)&d.sentiment.eq('正面')],10)}
    target=Path('evaluation/reference-facts.json')
    target.write_text(json.dumps({str(k):v for k,v in facts.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved independent CSV reference facts for 56 questions; no grades assigned.')


if __name__=='__main__':
    main()
