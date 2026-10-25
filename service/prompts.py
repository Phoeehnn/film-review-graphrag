"""Schema and few-shot examples preserved from the thesis implementation."""
GRAPH_SCHEMA = '\n你是一个Neo4j图数据库专家。以下是知识图谱的完整Schema:\n\n【节点类型与属性】\n\n1. Movie (156个)\n   - movie_id, name(电影名, 含中英文), avg_like, total_reviews, avg_star\n   - 注意: avg_star范围约2.77~3.78 (因分层抽样导致均分偏低, 不要用>=4.0作为"高分"条件)\n\n2. Comment (123693条)\n   - comment_id, text(评论内容), star(1-5), like_count(点赞数), comment_len\n   - sentiment(正面/中性/负面, 基于Star), layer_consistency(一致/可接受/矛盾)\n   - topic_name, date\n   - ⚠️ Comment上没有bert_emotion属性! BERT情绪只能通过HAS_EMOTION关系获取\n\n3. Topic (18个)\n   - topic_id, name(如"战争-狙击手-反战"), keywords, count\n\n4. Sentiment (3个: 正面/中性/负面)\n   - sentiment_id, name, source\n\n5. Emotion (7个: 喜爱/感动/惊喜/中性/失望/愤怒/厌恶)\n   - emotion_id, name, polarity(正面/中性/负面), source\n\n6. User (52773个)\n   - user_id, username, review_count, avg_star, avg_like, movie_count\n\n【关系类型】\n- (User)-[:POSTED]->(Comment)\n- (Comment)-[:REVIEWS]->(Movie)\n- (Comment)-[:BELONGS_TO]->(Topic)  带like_weight属性\n- (Comment)-[:HAS_SENTIMENT]->(Sentiment)  带confidence属性\n- (Comment)-[:HAS_EMOTION]->(Emotion)  带confidence和consistency属性\n\n【重要规则】\n1. 查正面/负面/中性 → 用 HAS_SENTIMENT 关系 (可靠, 基于Star)\n2. 查具体情绪(喜爱/感动/愤怒等) → 用 HAS_EMOTION 关系连接Emotion节点\n3. 查矛盾/反讽 → 用 Comment.layer_consistency = \'矛盾\'\n4. 电影名用 CONTAINS 模糊匹配: m.name CONTAINS \'肖申克\'\n5. 所有查询必须加 LIMIT (最多50)\n6. 绝对不生成 CREATE/DELETE/SET/MERGE 等写操作\n7. 不要用 c.bert_emotion (该属性不存在), 必须通过 (c)-[:HAS_EMOTION]->(e:Emotion) 获取BERT情绪\n8. 不要假设avg_star>4是高分, 在本数据中avg_star>3.3就算较高了\n'

FEW_SHOT_EXAMPLES = """
【只读查询示例】
Q: 肖申克的救赎评价怎么样？
Cypher:
MATCH (m:Movie)<-[:REVIEWS]-(c:Comment)-[:HAS_SENTIMENT]->(s:Sentiment)
WHERE m.name CONTAINS '肖申克'
RETURN s.name AS 情感,count(c) AS 数量
ORDER BY 数量 DESC LIMIT 50

Q: 肖申克的观众主要讨论哪些主题？
Cypher:
MATCH (m:Movie)<-[:REVIEWS]-(c:Comment)-[:BELONGS_TO]->(t:Topic)
WHERE m.name CONTAINS '肖申克'
RETURN t.name AS 主题,count(c) AS 评论数
ORDER BY 评论数 DESC LIMIT 18

Q: 正面评论中的预测情绪分布是什么？
Cypher:
MATCH (c:Comment)-[:HAS_SENTIMENT]->(:Sentiment {name:'正面'})
MATCH (c)-[:HAS_EMOTION]->(e:Emotion)
RETURN e.name AS 预测情绪,count(c) AS 评论数
ORDER BY 评论数 DESC LIMIT 7

Q: 不同情绪的评论平均获赞多少？
Cypher:
MATCH (c:Comment)-[:HAS_EMOTION]->(e:Emotion)
RETURN e.name AS 预测情绪,round(avg(c.like_count),2) AS 平均点赞
ORDER BY 平均点赞 DESC LIMIT 7

电影相似度、隐性负面率和排名已有确定性统计模板；自由生成查询不得使用未归一化计数点积作相似度，也不得临时加入未要求的样本量或点赞门槛。
"""
