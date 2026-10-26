# MCP 工具接入与 Redis 加速

MCP 将系统的六个查询工具开放给兼容客户端；Redis 为重复查询提供缓存，并按配置控制接口请求频率。

## 六个可复用工具

| 工具 | 作用 |
|---|---|
| `get_movie_overview` | 电影概览、样本评论数及星级摘要 |
| `get_movie_emotions` | 电影的预测情绪分布与计数 |
| `list_topics` | 主题目录与显示名称 |
| `search_movies_by_topic` | 按讨论主题查找电影 |
| `get_top_movies_by_rating` | 按样本内评分指标排名 |
| `search_comments` | 按语义及筛选条件查找评论观点 |

MCP 和 API 共用 ToolRegistry 及 Pydantic 参数校验，保持相同的工具语义和证据结构。

## 连接 MCP 客户端

先按项目快速开始配置数据库、数据及编码器，在 Python 环境中安装 MCP 依赖：

```bash
python -m pip install -r requirements-mcp.txt
```

本地客户端可以通过 stdio 启动服务：

```bash
python -m mcp_server --transport stdio
```

支持 Streamable HTTP 的客户端可使用：

```bash
python -m mcp_server --transport http --port 18080
```

HTTP 服务绑定本机；stdio 的标准输出用于协议消息。

## Redis 缓存与限流

缓存键包含问题、策略、筛选和版本/源码指纹，TTL 可配置；限流按 IP 统计请求频率。两个功能通过环境配置分别启用。

在 `.env` 中设置所需功能，例如：

```dotenv
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
```

然后启动 performance 配置：

```bash
docker compose --profile performance up -d
```

本机直接运行 API 时，Redis 默认地址为 `redis://127.0.0.1:16379/0`；Compose 中由服务名连接。正式问答评测默认关闭响应缓存，使每次任务记录对应真实检索与生成过程。
