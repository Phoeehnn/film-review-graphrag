"""API-backed chat with a quiet home page and progressive disclosure."""
import json
import os
from uuid import uuid4
from datetime import date

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from service.topics import friendly_text,friendly_value,topic_display
from ui.presentation import heading,hero,metrics

load_dotenv()
API=os.getenv('API_BASE_URL','http://127.0.0.1:18000').rstrip('/')
STRATEGIES={'自动选择（推荐）':'auto','图谱统计':'graph','语义评论':'vector','融合检索':'hybrid'}
STRATEGY_NAMES={'auto':'自动选择','graph':'图谱统计','vector':'语义评论','hybrid':'融合检索','agent':'Agent 助手'}
STATUS_NAMES={'ok':'已完成','partial':'部分完成','cannot_answer':'无法回答','insufficient_evidence':'证据不足','no_results':'暂无结果','error':'运行出错','timeout':'请求超时'}
FILTER_DEFAULTS={'filter_movie':'不限','filter_topic':'不限','filter_sentiment':'不限',
    'filter_emotion':'不限','filter_consistency':'不限','filter_stars':(1,5),
    'filter_date_from':'','filter_date_to':''}


@st.cache_data(ttl=30)
def catalog_stats():
    with httpx.Client(timeout=20) as client:
        catalog=client.get(API+'/api/v1/catalog');catalog.raise_for_status()
        stats=client.get(API+'/api/v1/stats');stats.raise_for_status()
    return catalog.json(),stats.json()


def reset_filters():
    st.session_state.update(FILTER_DEFAULTS)


def recall_changed():
    st.session_state['recall_count_changed']=True


def queue_example(question,entry):
    # Runs as a callback before widgets render, so examples start in a known scope.
    reset_filters()
    st.session_state['query_strategy']='自动选择（推荐）'
    st.session_state['recall_count']=5
    st.session_state['entry_mode']=entry
    st.session_state['pending_question']=question
    st.session_state.pop('conversation_session',None)


def clear_history():
    entry=st.session_state.get('entry_mode','检索问答')
    st.session_state['history_'+entry]=[]
    st.session_state.pop('pending_question',None)
    if entry!='Agent':
        identifier=st.session_state.pop('conversation_session',None)
        if identifier:
            try:httpx.delete(API+'/api/v1/sessions/'+identifier,timeout=5).raise_for_status()
            except httpx.HTTPError:st.session_state['session_clear_warning']='本地对话已清空，服务端会话清除暂未成功，将在到期时清理。'


def sidebar_controls(catalog,stats):
    with st.sidebar:
        st.divider()
        st.session_state.setdefault('filter_stars',(1,5))
        st.session_state.setdefault('recall_count',5)
        entry=st.radio('问答方式',['检索问答','Agent'],key='entry_mode',
            format_func=lambda value:{'检索问答':'直接问答','Agent':'Agent 助手'}[value],
            help='直接问答适合单个问题。Agent 会分步调用工具，并能理解当前会话最近六轮追问。')
        if entry=='Agent':
            st.caption('把电影、主题等条件直接写在问题里。助手会自行选择工具。')
        else:
            st.caption('默认自动检索。直接输入问题，或点击首页示例。')
        disabled=entry=='Agent' or catalog is None
        with st.expander('限定评论范围',expanded=False):
            st.caption('不选就搜索全部样本。多个条件同时生效。')
            movie=st.selectbox('电影',['不限']+(catalog['movies'] if catalog else []),key='filter_movie',disabled=disabled,
                help='只检索这部电影的评论；不选时使用全部样本。')
            topic_names={t['id']:t.get('display_name',t['name']) for t in catalog['topics']} if catalog else {}
            topic=st.selectbox('讨论主题',['不限']+list(topic_names),key='filter_topic',disabled=disabled,
                format_func=lambda value:topic_names.get(value,value),help='评论在谈什么，例如演技、剧情或性别议题。它不是电影题材。')
            if topic!='不限' and catalog:
                st.caption(next(t for t in catalog['topics'] if t['id']==topic).get('description',''))
            sentiment=st.selectbox('星级态度',['不限','正面','中性','负面'],key='filter_sentiment',disabled=disabled,
                help='由评论星级归类，不代表文字情绪。具体对应规则可在使用指南查看。')
        with st.expander('更多筛选条件',expanded=False):
            emotion=st.selectbox('文字情绪（模型预测）',['不限','喜爱','感动','惊喜','中性','失望','愤怒','厌恶'],key='filter_emotion',disabled=disabled,
                help='按已保存的模型预测筛选。预测可能出错，阅读时仍以评论原文为准。')
            consistency=st.selectbox('评分与预测情绪的关系',['不限','一致','矛盾','可接受'],key='filter_consistency',disabled=disabled,
                help='原项目给出的双层标签。矛盾表示评分态度和预测情绪存在明显差异，不一定是观众自相矛盾。')
            stars=st.slider('评论星级范围',1,5,value=None,key='filter_stars',disabled=disabled,help='1星最低，5星最高。拖动两端限制范围。')
            date_from=st.text_input('起始日期',key='filter_date_from',placeholder='例如 2018-01-01',disabled=disabled,help='填写年-月-日，留空表示没有起始限制。')
            date_to=st.text_input('结束日期',key='filter_date_to',placeholder='例如 2018-12-31',disabled=disabled,help='留空表示没有结束限制。结束日期应不早于起始日期。')
        filters={k:v for k,v in {'movie':None if movie=='不限' else movie,'topic':None if topic=='不限' else topic,
            'sentiment':None if sentiment=='不限' else sentiment,'emotion':None if emotion=='不限' else emotion,
            'consistency':None if consistency=='不限' else consistency,'min_star':stars[0] if stars!=(1,5) else None,
            'max_star':stars[1] if stars!=(1,5) else None,'date_from':date_from or None,'date_to':date_to or None}.items() if v is not None}
        if entry!='Agent' and filters:
            st.caption(f'已设置 {len(filters)} 项筛选条件。')
            st.button('重置筛选',key='reset_filters',on_click=reset_filters,width='stretch')
        with st.expander('检索设置 · 对照实验',expanded=False):
            mode=st.selectbox('回答依据',list(STRATEGIES),key='query_strategy',disabled=entry=='Agent',
                help='一般保持自动选择。统计选图谱，找相似表达选语义评论，统计加解释选融合。强制模式只走所选路径。')
            top_k=st.slider('展示评论条数',1,20,value=None,key='recall_count',disabled=entry=='Agent',on_change=recall_changed,help='显示多少条评论证据，默认5条。会话追问会沿用本轮条件里的条数，拖动此处可修改；问题明确指定条数时以问题为准。不是数据库中符合条件的总数。')
            st.caption('不确定时，保持自动选择和默认条数即可。')
        st.button('清空当前对话',key='clear_history',on_click=clear_history,width='stretch',
            help='只清空当前问答方式的页面对话，不删除数据库、评测结果或另一种方式的对话。')
        st.divider()
        st.caption('● 数据服务已连接' if stats else '○ 数据服务暂未就绪')
        if stats:st.caption(f"实际节点 {stats['node_total']:,} · 样本内结果")
    return entry,STRATEGIES[mode],top_k,filters


def show_graph(subgraph):
    if not subgraph['nodes']:
        st.info('本次回答依据是统计表，没有返回评论节点。可以换一个找评论的问题查看关系图。')
        return
    nodes=subgraph['nodes'][:80];ids={node['id'] for node in nodes}
    quote=lambda value:json.dumps(str(value),ensure_ascii=False)
    colors={'Movie':'#d4e6dd','Comment':'#edf2ed','Topic':'#e8e3f1','Sentiment':'#e5edda','Emotion':'#f1e4d7','User':'#eae5d5'}
    names={'Movie':'电影','Comment':'评论','Topic':'主题','Sentiment':'星级态度','Emotion':'预测情绪','User':'观众'}
    edges={'REVIEWS':'评价','BELONGS_TO':'讨论主题','HAS_SENTIMENT':'星级归类','HAS_EMOTION':'情绪预测','POSTED':'发表'}
    lines=['digraph G { graph [rankdir=LR,bgcolor="transparent",pad="0.2"]; node [shape=box,style="rounded,filled",fontname="sans-serif",color="#d2ddd4",fontcolor="#294c51"]; edge [fontname="sans-serif",fontsize=10,color="#a9bdb0",fontcolor="#7a8d83"];']
    for node in nodes:
        prop=node['properties'];kind=node['labels'][0]
        label=prop.get('name',prop.get('username',prop.get('comment_id',names.get(kind,kind))))
        if 'Topic' in node['labels']:label=topic_display(label)
        label=names.get(kind,kind)+' · '+str(label)[:24]
        lines.append(quote(node['id'])+' [label='+quote(label)+',fillcolor='+quote(colors.get(kind,'#edf2ed'))+'];')
    for edge in subgraph['relationships']:
        if edge['source'] in ids and edge['target'] in ids:
            lines.append(quote(edge['source'])+' -> '+quote(edge['target'])+' [label='+quote(edges.get(edge['type'],edge['type']))+'];')
    st.graphviz_chart('\n'.join(lines+['}']),width='stretch')
    st.caption(f"本次实际返回 {len(subgraph['nodes'])} 个节点、{len(subgraph['relationships'])} 条关系；图中最多展示前 80 个节点。")


def show_result(result):
    strategy=STRATEGY_NAMES.get(result['strategy'],result['strategy'])
    state=STATUS_NAMES.get(result['status'],result['status'])
    st.caption(f"{strategy} · {state} · {result['elapsed_seconds']:.2f} 秒 · {len(result['evidence'])} 项依据")
    for warning in result.get('warnings',[]):st.info(friendly_text(warning))
    if result.get('error'):st.warning('本次处理遇到问题，具体原因可在技术详情查看。')
    with st.expander('查看回答依据 · 原文、统计与关系图',expanded=False):
        labels=['原始评论','统计依据','关系图']
        if result.get('tool_calls'):labels.append('助手执行过程')
        labels.append('技术详情（可选）')
        tabs=st.tabs(labels)
        with tabs[0]:
            comments=[e for e in result['evidence'] if e['kind']=='comment']
            if not comments:st.info('本次没有独立评论证据。部分图谱查询会把文本保存在统计行中，可查看统计依据。')
            for ordinal,item in enumerate(comments,1):
                data=item['data']
                with st.container(border=True):
                    st.markdown('**第'+str(ordinal)+'条 · ['+item['id']+'] '+data.get('movie','未提供电影')+'**')
                    st.write(data.get('text',''))
                    st.caption(f"{data.get('star','—')} 星 · 星级态度 {data.get('sentiment','—')} · 预测情绪 {data.get('emotion','—')}")
                    st.caption(f"主题 {data.get('topic_display_name') or topic_display(data.get('topic'))} · 日期 {data.get('date','—')} · 点赞 {data.get('like_count','—')}")
                    st.caption(f"评分与预测关系 {data.get('layer_consistency','未提供')} · 情绪来源 {data.get('emotion_source','未提供')}")
        with tabs[1]:
            aggregates=[e for e in result['evidence'] if e['kind']=='aggregate']
            for item in aggregates:
                st.markdown('**['+item['id']+'] '+item['data'].get('scope','样本内查询结果')+'**')
                rows=friendly_value(item['data'].get('rows',[]))
                if rows:
                    try:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
                    except (ValueError,TypeError):st.json(rows)
                if item['data'].get('definition'):st.caption(item['data']['definition'])
            if not aggregates:st.info('本次只检索评论原文。几条评论不能用于计算全部样本的比例。')
        with tabs[2]:show_graph(result['subgraph'])
        if result.get('tool_calls'):
            with tabs[3]:
                tool_names={'get_movie_overview':'查看电影概况','get_movie_emotions':'查看电影情绪分布','list_topics':'查看主题目录','search_movies_by_topic':'查找主题下的电影','get_top_movies_by_rating':'查看评分排名','search_comments':'查找相关评论'}
                for index,tool in enumerate(result['tool_calls'],1):
                    name=tool.get('name',tool.get('tool','工具'))
                    st.markdown(f"**步骤 {index} · {tool_names.get(name,name)}**")
                    st.json(friendly_value(tool))
        with tabs[-1]:
            st.caption('供开发和复现使用，不需要看懂这些内容才能提问。')
            st.write('请求编号：'+result['request_id'])
            with st.expander('Cypher：实际执行的数据库查询'):
                for query in result.get('cypher',[]):st.code(query,language='cypher')
                if not result.get('cypher'):st.caption('本次没有生成图谱查询。')
            with st.expander('模型用量、耗时与证据检查'):
                st.json({'服务版本':result.get('service_version'),'模型用量':result['model_usage'],
                    '耗时（秒）':result.get('stage_timings',{}),'证据检查':result.get('evidence_checks',{}),'错误':result.get('error')})
            with st.expander('完整关系数据（保留数据库原名）'):st.json(result['subgraph'])
            if result.get('cypher_attempts'):
                with st.expander('查询生成与修复记录'):
                    if result.get('adopted_attempt') is not None:
                        st.caption(f"最终采用第 {result['adopted_attempt']} 次查询。结果为空也会保留；执行成功仍需核对查询是否回答了原问题。")
                    st.caption('unsafe 表示触犯安全规则；scope 表示无法可靠施加筛选，两者都不会执行。repeated 表示重复失败候选，未再次执行。')
                    for attempt in result['cypher_attempts']:
                        st.caption(f"第 {attempt['attempt']} 次 · {attempt['status']} · {attempt.get('failure_type','')} · {attempt['elapsed_seconds']:.2f} 秒")
                        query=attempt.get('safe_cypher') or attempt.get('cypher')
                        if query:st.code(query,language='cypher')
                        else:st.caption('模型调用未产生查询候选。')
                        if attempt.get('error'):st.caption(attempt['error'])
            if result.get('conversation'):
                with st.expander('本轮追问改写与会话条件'):st.json(result['conversation'])
            st.download_button('下载本次完整结果',json.dumps(result,ensure_ascii=False,indent=2),file_name=result['request_id']+'.json',mime='application/json',key='download_'+result['request_id'],icon=':material/download:')


def example_cards(entry):
    if entry=='Agent':
        examples=[('找评论，再解释','找两条肖申克中关于希望和自由的评论，解释它们表达了什么。','分步检索'),
            ('比较两部电影','比较肖申克和大圣归来的情绪分布，再找代表评论。','对比与证据'),
            ('探索性别议题','找几条性别议题下讨论女性自主选择的评论，并概括观点。','主题探索')]
    else:
        examples=[('看看电影评价','肖申克的救赎评价怎么样？','综合了解'),
            ('数一数评论','肖申克的救赎有多少条评论？','样本统计'),
            ('寻找相似表达','找几条表达希望和自由的相似评论。','语义检索'),
            ('探索性别议题','性别议题下，观众在讨论什么？','主题探索'),
            ('统计之后，看看原因','肖申克的正面占比是多少？结合评论解释喜欢它的原因。','统计与解释'),
            ('看看评分与情绪的差异','三块广告牌的矛盾评论占比是多少？结合至少两条评论解释原因。','双层情感')]
    heading('从一个问题开始','点击即可提问')
    st.caption('示例会恢复默认筛选；直接问答示例使用自动选择。')
    for offset in range(0,len(examples),3):
        for index,(col,example) in enumerate(zip(st.columns(3),examples[offset:offset+3]),offset):
            title,question,category=example
            with col:
                with st.container(border=True):
                    st.caption(category)
                    st.button(title,key=f'example_{entry}_{index}',on_click=queue_example,args=(question,entry),width='stretch',icon=':material/arrow_outward:')
                    st.caption(question)


def session_controls(entry):
    if entry=='Agent':return False,None
    with st.sidebar:
        enabled=st.toggle('记住当前会话',value=True,key='remember_context',help='理解“那正面的呢”等追问。条件可固定或撤销，摘要不会修改固定条件；会话当天有效。关闭后每个问题独立处理。')
        if not enabled:return False,None
        identifier=st.session_state.get('conversation_session')
        if not identifier:
            st.caption('首次发送后创建会话；电影、主题等条件会在回答中显示。')
            return True,None
        try:
            response=httpx.get(API+'/api/v1/sessions/'+identifier,timeout=8)
            if response.status_code==404:
                st.session_state.pop('conversation_session',None)
                st.info('上一会话已过期，本次将重新开始。');return True,None
            response.raise_for_status();state=response.json()
        except httpx.HTTPError:
            st.warning('暂时无法读取会话状态，恢复服务后再试。');return True,None
        with st.expander('当前会话条件',expanded=bool(state.get('active_conditions'))):
            names={'movie':'电影','topic':'主题','sentiment':'星级态度','date_from':'开始日期','date_to':'结束日期','emotion':'预测情绪','consistency':'一致性','min_star':'最低星级','max_star':'最高星级','top_k':'评论条数'}
            active=state.get('active_conditions',{})
            if active:
                for key,value in active.items():
                    if key=='topic':value=topic_display(value)
                    st.caption(names.get(key,key)+'：'+str(value)+(' · 已固定' if state.get('pinned',{}).get(key)==active[key] else ' · 本轮条件'))
                if st.button('固定本轮条件',key='pin_conditions'):
                    save_conditions(identifier,{'expected_version':state['version'],'updates':active})
                chosen=st.multiselect('要撤销哪些条件',list(active),format_func=lambda key:names.get(key,key),key='unpin_fields')
                if st.button('撤销所选条件',key='unpin_conditions',disabled=not chosen):
                    save_conditions(identifier,{'expected_version':state['version'],'unset':chosen})
            else:st.caption('还没有生效条件。')
        return True,state


def save_conditions(identifier,payload):
    try:
        r=httpx.patch(API+'/api/v1/sessions/'+identifier+'/conditions',json=payload,timeout=8)
        r.raise_for_status()
    except httpx.HTTPError:
        st.warning('条件未保存，可能是连接中断或会话已更新。请刷新会话后重试。')
        return
    st.rerun()


def render_explorer():
    try:catalog,stats=catalog_stats()
    except (httpx.HTTPError,ValueError):catalog,stats=None,None
    entry,strategy,top_k,filters=sidebar_controls(catalog,stats)
    remember,session_state=session_controls(entry)
    history=st.session_state.setdefault('history_'+entry,[])
    if not history:
        hero();metrics(stats)
        with st.form('first_question_form',clear_on_submit=True,border=False):
            first_question=st.text_input('输入你的问题',key='first_question_text',
                placeholder='想了解哪部电影，或寻找怎样的评论？在这里输入…')
            submitted=st.form_submit_button('发送问题',type='primary',key='send_first_question',icon=':material/send:')
        typed=first_question.strip() if submitted else None
        if submitted and not typed:
            st.warning('请先输入一个问题。')
        example_cards(entry)
        st.markdown('<div class="note-strip">所有统计描述历史样本；文字情绪是模型预测。答案中的 [c_…] 和 [g_…] 是来源编号，可以展开回答依据核对。</div>',unsafe_allow_html=True)
        if not stats:st.warning('数据服务尚未就绪。请先启动 Docker Desktop 和本项目容器，稍后刷新。')
    else:
        st.markdown('<div class="eyebrow">'+('AGENT WORKSPACE' if entry=='Agent' else 'FILM REVIEW Q&A')+'</div>',unsafe_allow_html=True)
        st.markdown('<div class="compact-heading">'+('一起探索，分步找证据。' if entry=='Agent' else '把问题交给评论，把判断留给你。')+'</div>',unsafe_allow_html=True)
        st.caption('Agent 可以理解当前会话的最近六轮追问。' if entry=='Agent' else '会话模式可继承条件；“第二条”按回答依据中评论卡片的序号定位。' if remember else '每个问题独立检索；请写明电影或主题。')
    if message:=st.session_state.pop('session_clear_warning',None):st.warning(message)
    for message in history:
        with st.chat_message(message['role']):
            st.markdown(friendly_text(message['content']) if message['role']=='assistant' else message['content'])
            if message.get('result'):show_result(message['result'])
    if history:
        typed=st.chat_input('问一部电影、找一种表达，或探索一个主题…',key='question_input')
    question=st.session_state.pop('pending_question',None) or typed
    if not question:return
    if entry!='Agent':
        try:
            for key in ('date_from','date_to'):
                if key in filters:date.fromisoformat(filters[key])
            if filters.get('date_from') and filters.get('date_to') and filters['date_from']>filters['date_to']:
                raise ValueError('结束日期早于起始日期')
        except ValueError:
            st.error('请检查日期：使用 YYYY-MM-DD 格式，结束日期不能早于起始日期。')
            return
    with st.chat_message('user'):st.markdown(question)
    with st.chat_message('assistant'):
        output,progress=st.empty(),st.empty()
        try:
            with httpx.Client(timeout=135) as client:
                if entry=='Agent':
                    progress.caption('助手正在分步寻找证据…')
                    recent=[{'role':m['role'],'content':m['content'][:2000]} for m in history[-12:]]
                    response=client.post(API+'/api/v1/agent',json={'question':question,'history':recent});response.raise_for_status()
                    result=response.json()
                else:
                    parts,result=[],None
                    payload={'question':question,'strategy':strategy,'top_k':top_k,'filters':filters}
                    endpoint='/api/v1/query/stream'
                    if remember:
                        identifier=st.session_state.get('conversation_session')
                        if not identifier:
                            created=client.post(API+'/api/v1/sessions');created.raise_for_status()
                            session_state=created.json();identifier=session_state['id']
                            st.session_state['conversation_session']=identifier
                        payload['client_request_id']=str(uuid4())
                        payload['expected_version']=session_state['version'] if session_state else None
                        if not st.session_state.pop('recall_count_changed',False) and session_state:
                            payload['top_k']=session_state.get('active_conditions',{}).get('top_k',top_k)
                        endpoint='/api/v1/sessions/'+identifier+'/query/stream'
                    with client.stream('POST',API+endpoint,json=payload) as response:
                        response.raise_for_status();event=None
                        for line in response.iter_lines():
                            if line.startswith('event: '):event=line[7:]
                            elif line.startswith('data: '):
                                data=json.loads(line[6:])
                                if event=='delta':
                                    parts.append(data['text']);output.markdown(friendly_text(''.join(parts))+' ▌')
                                elif event=='progress':progress.caption({'routing':'正在选择检索方式…','retrieval':'正在查找统计和评论…','generation':'正在整理有来源的回答…'}.get(data['stage'],'正在处理…'))
                                elif event=='correction':
                                    output.markdown(friendly_text(data['answer']));progress.caption('证据检查发现冲突，已改为可以确认的内容。')
                                elif event=='done':result=data
                    if result is None:raise RuntimeError('连接结束前未收到最终结果')
            progress.empty();output.markdown(friendly_text(result['answer']));show_result(result)
            history.extend([{'role':'user','content':question},{'role':'assistant','content':result['answer'],'result':result}])
            st.rerun()
        except (httpx.HTTPError,ValueError,RuntimeError,KeyError) as exc:
            progress.empty();output.error('暂时未能完成请求，请检查服务状态后重试。')
            with st.expander('查看错误详情'):st.code(str(exc))
