# 数据格式与启动前提

数据不随仓库提供，也没有自动下载脚本。完整服务需准备以下原格式数据；没有数据时可阅读代码和运行离线测试，不能完成真实问答。

## Neo4j导入CSV

所有CSV使用UTF-8/BOM可读编码、表头及合法引用。文件位于export/neo4j；用户名和评论内容属于私有包，不应提交到仓库。

|文件|列名|
|---|---|
|comments.csv|comment_id, Comment, Star, Like, comment_len, sentiment, bert_emotion_7class, bert_confidence, layer_consistency, topic_id, topic_name, Movie_Name, Username, Date|
|emotions.csv|emotion_id, emotion_name, polarity, source|
|movies.csv|movie_name, avg_like, total_reviews, avg_star, like_std, 喜爱, 感动, 惊喜, 中性, 失望, 愤怒, 厌恶, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, movie_id|
|rel_comment_emotion.csv|comment_id, emotion_id, bert_confidence, layer_consistency|
|rel_comment_movie.csv|comment_id, movie_id|
|rel_comment_sentiment.csv|comment_id, sentiment_id, confidence|
|rel_comment_topic.csv|comment_id, topic_id, like_weight|
|rel_user_comment.csv|user_id, comment_id|
|sentiments.csv|sentiment_id, sentiment_name, source|
|topics.csv|topic_id, topic_name, keywords, count|
|users.csv|user_id, username, review_count, avg_star, avg_like, movie_count|

## 向量与预处理文件

export_gpu/embeddings.npy为123693×384的float32向量，按原评论顺序对应、有限值、已归一化；查询编码器为paraphrase-multilingual-MiniLM-L12-v2。export_gpu/df_preprocessed.csv至少包含Comment、Movie_Name、Star、Date、Username并符合initialize脚本逐行校验。

导入器当前固定预期156部电影、123693条评论、18主题、3倾向、7情绪、52773用户，共176650节点、618436关系；29条评论没有用户关系。它不是任意CSV的通用导入器。替换数据集时需修改并重新验证导入校验、ID/引用和向量顺序，不能套用原实验成绩。

ID是稳定引用，跨文件关系须能定位节点；日期采用ISO格式，星级1–5，情感及情绪使用对应标签表。电影CSV的主题比例列为0至17。节点初始化与索引检查通过后API才报告ready。
