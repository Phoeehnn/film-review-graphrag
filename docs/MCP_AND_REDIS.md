# MCP与可选Redis

MCP适配复用六个工具：get_movie_overview、get_movie_emotions、list_topics、search_movies_by_topic、get_top_movies_by_rating、search_comments。参数与服务工具共享校验；语义检索不能计算全库统计。

安装requirements-mcp后可通过`python -m mcp_server --transport stdio`或`--transport http`启动。需要实际数据库和编码器；stdio只在标准输出传协议消息。客户端能连接并不保证所有业务问题可答。

Redis通过Compose的performance profile启用，负责可配置TTL响应缓存与按IP限流。缓存键包含问题、策略、筛选与版本/源码指纹；评测时默认关闭。缓存命中延迟不能替代首次生成能力或答案正确率。
