import re
import json
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict, dataclass, field

from .schemas import Filters
from .topics import canonical_topic, mentioned_topics

STAT = re.compile(r'数量|多少|几部|几条|最多|最高|最低|排名|排行|比例|占比|正面率|负面率|矛盾率|隐性负面率|一致性|分布|趋势|按月|平均|对比|统计|超过|获赞|评分|有哪些|基本信息|评论条数|评论量|分位点|百分之几|汇总')
EXPLAIN = re.compile(r'为什么|原因|解释|结合.{0,8}评论|举.{0,3}例|代表.{0,3}评论|评论.{0,5}说明|(?:给|列).{0,6}(?:例子|评论原文|真实评论|代表评论)|原文例子|帮助理解|结合文本|附.{0,5}原文|原话说明|影评例子|抱怨什么|支持.{0,8}判断.{0,5}影评|评论.{0,4}谈谈')
SEMANTIC = re.compile(r'相似.{0,20}(评论|影评|话)|类似.{0,40}(评论|影评|话)|同义|相同意思|意思接近|语义|表达|找.{0,30}(评论|影评)|说法|措辞|有人.{0,12}说过|原话|原文|原句|真实评论|描写|比喻|怎么形容')
SEMANTIC_COMMENTS = re.compile(r'(?:哪些|什么)(?:评论|影评).{0,12}(?:提到|谈到|说到|表达|认为)|(?:评论|影评).{0,10}(?:提到|谈到|说到|表达)')
STAT_CORE = re.compile(r'数量|有多少|几部|最多|最高|最低|排名|排行|比例|占比|正面率|负面率|矛盾率|隐性负面率|一致性|分布|趋势|按月|平均|对比|统计|超过|获赞|评分|基本信息|评论条数|评论量|分位点|百分之几|汇总')
RANKING = re.compile(r'最负面|最正面|最消极|最积极|前\s*[0-9一二三四五六七八九十]+\s*(条|部|名)|排序')


@lru_cache
def movie_catalog():
    return json.loads(Path(__file__).with_name('movie_catalog.json').read_text(encoding='utf-8'))['movies']


def route(question: str, requested: str, movies=None) -> tuple[str, str]:
    if requested != 'auto':
        return requested, '用户强制指定检索策略'
    # Quoted text is the retrieval target, not the user's requested operation.
    intent = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"|\'[^\'\n]*\'', ' ', question)
    # Titles are targets, not intent. A long bilingual title (e.g. "三块广告牌
    # Three Billboards Outside Ebbing, Missouri") otherwise pushes the words of
    # the request outside the bounded windows below and changes the route.
    intent = re.sub(r'《[^》]*》', '《》', intent)
    known=movie_catalog() if movies is None else movies
    aliases=json.loads(Path(__file__).with_name('movie_aliases.json').read_text(encoding='utf-8'))
    # Mask only verified titles and unique reviewed aliases, never arbitrary
    # multiword English that could contain the actual retrieval target.
    names=list(known)+[a for a,t in aliases.items() if sum(normalize_name(t) in normalize_name(m) for m in known)==1]
    for name in sorted(names,key=len,reverse=True):
        if len(name)>=2:intent=re.sub(re.escape(name),' ',intent,flags=re.I)
    structured=bool(STAT.search(intent) or RANKING.search(intent))
    if structured and EXPLAIN.search(intent):
        return 'hybrid', '同时需要样本统计和评论解释'
    if (SEMANTIC.search(intent) or SEMANTIC_COMMENTS.search(intent)) and not STAT_CORE.search(intent) and not re.search(r'最负面|最正面|最消极|最积极|排序',intent):
        return 'vector', '需要语义相似的评论证据'
    if structured:
        return 'graph', '需要数量、排名、比例、分布或趋势'
    return 'hybrid', '其余问题使用图谱和语义证据融合'


@dataclass
class ResolvedFilters:
    movies: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sentiment: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    explicit: bool = False
    emotion: str | None = None
    consistency: str | None = None
    min_star: int | None = None
    max_star: int | None = None
    emotions: list[str] = field(default_factory=list)

    def params(self):
        return asdict(self)

    def has_scope(self):
        return bool(self.movies or self.topics or self.sentiment or self.date_from or self.date_to or self.emotion or self.emotions or self.consistency or self.min_star is not None or self.max_star is not None)


def normalize_name(value: str) -> str:
    return re.sub(r'[\W_]+', '', value).lower()


def resolve_filters(question: str, filters: Filters, movies: list[str], topics: list[dict]) -> ResolvedFilters:
    q = normalize_name(question)
    movie_names = [filters.movie] if filters.movie else [name for name in movies if len(normalize_name(name)) >= 2 and normalize_name(name) in q]
    matched_aliases = []
    if not filters.movie:
        # Reviewed aliases resolve only to a unique title present in the catalogue.
        from pathlib import Path
        import json
        aliases=json.loads(Path(__file__).with_name('movie_aliases.json').read_text(encoding='utf-8'))
        for alias,title in aliases.items():
            if normalize_name(alias) in q:
                matches=[name for name in movies if normalize_name(title) in normalize_name(name)]
                if len(matches)==1:movie_names.extend(matches);matched_aliases.append(alias)
    # Unique distinctive title prefixes allow shortened movie names, e.g. 肖申克.
    # Run for every question, not only when nothing matched yet: in
    # "对比肖申克和大圣归来" the alias 大圣归来 must not hide 肖申克. Spans already
    # matched by a full title or alias are masked first so they cannot match twice.
    if not filters.movie:
        rest = q
        for matched in sorted({normalize_name(n) for n in movie_names}|{normalize_name(a) for a in matched_aliases},key=len,reverse=True):
            rest = rest.replace(matched, '|')
        for name in movies:
            if name in movie_names:
                continue
            norm = normalize_name(name)
            for width in range(min(6, len(norm)), 2, -1):
                part = norm[:width]
                if part in rest and sum(part in normalize_name(other) for other in movies) == 1:
                    movie_names.append(name)
                    break
    topic_names = [canonical_topic(filters.topic)] if filters.topic else mentioned_topics(question)
    if not topic_names and re.search(r'主题|话题|讨论|谈到|涉及|相关',question):
        for topic in topics:
            terms = [part for part in topic['name'].split('_')[-1].split('-') if len(part) >= 2]
            # Generic words are too broad to identify a topic reliably.
            if any(term in question for term in terms if term not in {'电影', '喜欢', '故事', '好看', '剧情', '演技', '不错', '演员', '评分', '中文', '生活', '中国'}):
                topic_names.append(topic['name'])
        if not topic_names and re.search(r'性别(?:主题|话题|议题|讨论)',question):
            topic_names=[canonical_topic('t_6')]
    sentiment = filters.sentiment
    if not sentiment:
        for phrase, value in [('正面评论', '正面'), ('负面评论', '负面'), ('中性评论', '中性')]:
            if phrase in question:
                sentiment = value
                break
    scope=ResolvedFilters(sorted(set(movie_names)),sorted(set(topic_names)),sentiment,filters.date_from,filters.date_to,bool(filters.model_dump(exclude_none=True)),filters.emotion,filters.consistency,filters.min_star,filters.max_star)
    if not scope.consistency and re.search(r'矛盾(?:情感)?(?:的)?评论|不一致的评论|评分.{0,10}预测情绪.{0,6}不一致',question):
        scope.consistency='矛盾'
    rejects_five=bool(re.search(r'(?:不该|不值|不给|不愿|凭什么).{0,10}(?:五星|5\s*星)',question))
    if scope.min_star is None and scope.max_star is None and re.search(r'五星|5\s*星',question) and not rejects_five and (SEMANTIC.search(question) or re.search(r'BERT|预测|情绪',question)):
        scope.min_star=scope.max_star=5
    if not scope.emotion and re.search(r'BERT.{0,12}负面|预测.{0,10}负面情绪',question):
        scope.emotions=['失望','愤怒','厌恶']
    if re.search(r'低评分.{0,10}喜爱',question):
        scope.emotion=scope.emotion or '喜爱'
        if scope.max_star is None:
            scope.max_star=2
    return scope


def comment_scope_violations(comment,scope):
    problems=[]
    for values,key in ((scope.movies,'movie'),(scope.topics,'topic')):
        if values and not any(str(value).lower() in str(comment.get(key,'')).lower() for value in values):
            problems.append(key)
    for expected,key in ((scope.sentiment,'sentiment'),(scope.emotion,'emotion'),(scope.consistency,'layer_consistency')):
        if expected and comment.get(key)!=expected:
            problems.append(key)
    if scope.emotions and comment.get('emotion') not in scope.emotions:
        problems.append('emotion')
    for bound,operator in ((scope.min_star,'min'),(scope.max_star,'max')):
        if bound is not None and (comment.get('star') is None or (comment['star']<bound if operator=='min' else comment['star']>bound)):
            problems.append(operator+'_star')
    if scope.date_from and (not comment.get('date') or comment['date']<scope.date_from):
        problems.append('date_from')
    if scope.date_to and (not comment.get('date') or comment['date']>scope.date_to):
        problems.append('date_to')
    return problems


def rrf(channels: dict[str, list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    merged = {}
    for channel, candidates in channels.items():
        seen = set()
        for rank, candidate in enumerate(candidates, 1):
            identifier = candidate['comment_id']
            if identifier in seen:
                continue
            seen.add(identifier)
            item = merged.setdefault(identifier, {**candidate, 'rrf_score': 0.0, 'ranks': {}})
            item['rrf_score'] += 1 / (k + rank)
            item['ranks'][channel] = rank
    return sorted(merged.values(), key=lambda item: (-item['rrf_score'], item['comment_id']))[:top_k]
