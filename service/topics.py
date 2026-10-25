"""User-approved display labels without changing BERTopic assignments."""
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def topic_config():
    config=json.loads(Path(__file__).with_name('topic_aliases.json').read_text(encoding='utf-8'))
    definitions=config['topics']
    if len(definitions)!=18 or {d['id'] for d in definitions}!={f't_{i}' for i in range(18)}:
        raise ValueError('主题别名配置必须完整覆盖t_0至t_17')
    names=[d['display_name'] for d in definitions]
    if len(set(names))!=18:
        raise ValueError('主题显示名称必须唯一')
    return config


def topic_definition(value):
    text=str(value).strip()
    match=re.fullmatch(r'(?:t_|主题\s*)(\d+)',text)
    identifier=f't_{int(match.group(1))}' if match else text
    for entry in topic_config()['topics']:
        if identifier==entry['id'] or text==entry['raw_name'] or text in entry['aliases']:
            return entry
    # Preserve unambiguous legacy keyword selectors (e.g. 战争/青春). Broad
    # words shared by multiple clusters still require an explicit topic choice.
    matches=[entry for entry in topic_config()['topics'] if len(text)>=2 and text in entry['raw_name'].split('_')[-1].split('-')]
    if len(matches)==1:
        return matches[0]
    return None


def canonical_topic(value):
    entry=topic_definition(value)
    return entry['raw_name'] if entry else value


def topic_display(value):
    entry=topic_definition(value)
    return entry['display_name'] if entry else value


def decorate_topic(row):
    entry=topic_definition(row.get('id',row.get('name','')))
    if not entry:
        return row
    if row['name']!=entry['raw_name']:
        raise ValueError(f"主题{entry['id']}原始名称与别名配置不符，不能悄悄改变映射")
    return {**row,'id':entry['id'],'raw_name':row['name'],
            'display_name':entry['display_name'],'aliases':entry['aliases'],
            'description':entry['description']}


def mentioned_topics(question):
    found=[]
    for entry in topic_config()['topics']:
        if any(alias in question for alias in entry['aliases']) or re.search(r'主题\s*'+entry['id'][2:]+r'(?!\d)',question):
            found.append(entry['raw_name'])
    return found


def friendly_text(text):
    # Only render labels; do not pass this through Cypher or quoted comment text.
    for entry in topic_config()['topics']:
        text=text.replace(entry['raw_name'],entry['display_name'])
    return text


def friendly_value(value):
    if isinstance(value,str):
        return topic_display(value)
    if isinstance(value,list):
        return [friendly_value(v) for v in value]
    if isinstance(value,dict):
        return {k:friendly_value(v) for k,v in value.items()}
    return value


GENRES=r'爱情|科幻|恐怖|悬疑|动作|犯罪|动画|纪录|冒险|家庭|历史|战争|喜剧|音乐|武侠'


def unsupported_scope_reason(question, filters=None):
    if filters and filters.topic and not topic_definition(filters.topic):
        return f'当前主题目录没有“{filters.topic}”。请从18个评论主题选择；主题名称与电影题材不同。'
    # Talking about genres in comment content is allowed; classifying films is not.
    semantic_comment=bool(re.search(r'(?:找|检索|相似|类似|表达|提到|说过).*(?:评论|影评|的话)',question))
    statistical=bool(re.search(r'比例|占比|正面率|负面率|排名|最多|最高|最低|多少部|哪些电影|电影有哪些',question))
    genre_selection=re.search(r'(?:找|检索).*(?:'+GENRES+r')(?:题材|类型)(?:的)?(?:电影|片)(?:的)?(?:评论|影评)',question)
    if semantic_comment and not statistical and not genre_selection:
        return None
    explicit=re.search(r'('+GENRES+r')(?:题材|类型)(?:的)?(?:片|电影)?',question)
    shorthand=re.search(r'('+GENRES+r')(?:片|电影)(?:中|里|的(?:正面|负面|评分|排名)|有哪些)',question)
    missing=explicit or shorthand
    if missing:
        label=missing.group(1)
        return f'当前数据库没有电影“{label}”题材标签，不能确认这类电影的完整范围。18个主题描述评论讨论角度，不能替代电影题材分类。请选择具体电影或评论主题后再查询。'
    if re.search(r'电影(?:属于|是|的)?(?:什么|哪种|哪些)?(?:题材|类型)',question):
        return '当前数据库没有电影题材或类型字段；不能据评论主题推断电影所属类型。可以查询具体电影的评论主题分布。'
    return None
