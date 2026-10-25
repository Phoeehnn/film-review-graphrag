"""Local visual assets and the beginner-friendly presentation pages."""
import html
from pathlib import Path

import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).resolve().parent/'assets'


def apply_theme():
    st.markdown('<style>'+ (ASSETS/'theme.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)


def brand():
    st.markdown('<div class="brand"><div class="brand-symbol">◈</div><div><b>影评知图</b><small>FILM REVIEW EXPLORER</small></div></div>',unsafe_allow_html=True)


def hero():
    visual=(ASSETS/'hero_graph.svg').read_text(encoding='utf-8')
    st.markdown('<section class="hero"><div class="hero-copy"><div class="eyebrow">影评 · 主题 · 情绪 · 知识图谱</div><h1>从一句影评，<br>看到电影的另一面。</h1><p>看看观众讨论了什么、为什么喜欢或失望。把样本统计与评论原文放在一起，让每个回答都有来源。</p><div class="hero-chips"><span>样本统计</span><span>语义找评论</span><span>Agent 助手</span><span>证据可追溯</span></div></div><div class="hero-visual">'+visual+'<div class="hero-visual-caption">关系结构示意 · 实际子图随回答展示</div></div></section>',unsafe_allow_html=True)


def metrics(stats):
    nodes=stats.get('nodes',{}) if stats else {}
    values=[('样本电影',nodes.get('Movie'),'不同作品的观众反馈'),('评论原文',nodes.get('Comment'),'每条评论都能追溯'),('讨论主题',nodes.get('Topic'),'按讨论角度整理'),('图谱关系',stats.get('relationship_total') if stats else None,'连接电影、评论与情绪')]
    cards=[]
    for title,value,note in values:
        value=f'{value:,}' if value is not None else '—'
        cards.append(f'<div class="metric-tile"><small>{title}</small><strong>{value}</strong><span>{note}</span></div>')
    st.markdown('<div class="metric-row">'+''.join(cards)+'</div>',unsafe_allow_html=True)


def heading(title,note=''):
    st.markdown(f'<div class="section-heading"><h2>{html.escape(title)}</h2><span>{html.escape(note)}</span></div>',unsafe_allow_html=True)


def intro_card(number,title,body):
    st.markdown(f'<div class="intro-card"><span class="number">{html.escape(number)}</span><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',unsafe_allow_html=True)


def render_about():
    from ui.explorer import catalog_stats
    st.markdown('<div class="eyebrow">ABOUT THE PROJECT</div>',unsafe_allow_html=True)
    st.title('电影之外，还有观众的故事。')
    st.markdown('<div class="page-lead">这个项目将影视评论中的主题、评分态度和预测情绪整理成知识图谱，再提供统计问答、语义检索与能调用工具的 Agent 助手。你可以从一部电影出发，也可以从一句意思相近的话出发。</div>',unsafe_allow_html=True)
    try:
        catalog,stats=catalog_stats()
    except Exception:
        catalog,stats=None,None
        st.info('数据库暂未就绪，规模信息将在服务恢复后显示。')
    metrics(stats)
    heading('一张图，怎样连接观众的声音？','结构示意，不代表某次真实查询结果')
    st.graphviz_chart('''digraph FilmReview {
      graph [rankdir=LR, bgcolor="transparent", pad="0.35", nodesep="0.5", ranksep="1.0"];
      node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=14, margin="0.22,0.14", color="#d7e3da", fontcolor="#254a50"];
      edge [fontname="sans-serif", fontsize=11, color="#a6bcb1", fontcolor="#7a8b84"];
      user [label="观众", fillcolor="#eae5d5"];
      comment [label="评论原文\\n星级 · 日期 · 点赞", fillcolor="#e4eee8"];
      movie [label="电影", fillcolor="#d4e6dd"];
      topic [label="讨论主题\\n如：剧情叙事", fillcolor="#e8e3f1"];
      sentiment [label="星级态度\\n正面 · 中性 · 负面", fillcolor="#e5edda"];
      emotion [label="文字情绪（模型预测）\\n喜爱 · 感动 · 愤怒等", fillcolor="#f1e4d7"];
      user -> comment [label="发表"];
      comment -> movie [label="评价"];
      comment -> topic [label="属于"];
      comment -> sentiment [label="星级归类"];
      comment -> emotion [label="情绪预测"];
    }''',width='stretch')
    st.markdown('<div class="note-strip">知识图谱可以理解为“有连接的资料库”：从一条评论，就能找到它评价的电影、讨论的主题和情绪标签。回答中的关系图展示的是本次证据对应的真实节点。</div>',unsafe_allow_html=True)
    heading('三种检索，各自解决什么问题？')
    cols=st.columns(3)
    cards=[('01','图谱：把数量算清楚','适合问多少、哪部最多、占比多少。直接查询样本数据，不用几条评论猜总体比例。'),('02','向量：按意思找评论','即使没有完全相同的词，也能寻找表达类似意思的评论，例如“困境中仍保留希望”。'),('03','融合：统计加上解释','把实际统计与代表评论放在一起，回答“有多少”及“为什么”。默认由系统判断该选哪种方式。')]
    for col,card in zip(cols,cards):
        with col:intro_card(*card)
    heading('18 个主题，18 种讨论角度')
    st.caption('显示名是对已有评论聚类的暂定解释。电影题材与评论主题是不同概念。')
    if catalog:
        names=[topic.get('display_name',topic['name']) for topic in catalog['topics']]
        st.markdown('<div class="hero-chips">'+''.join('<span>'+html.escape(name)+'</span>' for name in names)+'</div>',unsafe_allow_html=True)
        with st.expander('查看每个主题的范围说明'):
            for topic in catalog['topics']:
                st.markdown('**'+topic.get('display_name',topic['name'])+'** · '+topic.get('description',''))
    heading('从毕设产物，到可运行的应用')
    st.markdown('原项目完成评论整理、主题建模、双层情感分析与图谱构建。升级复用这些数据和向量，加入独立 API、三种检索方式、工具调用、流式回答、容器部署与对照评测。')
    st.markdown('**Agent 助手**会根据问题调用工具，例如先看电影的情绪分布，再寻找代表评论。你能展开看到它做了什么；它支持当前会话中的短期追问。')
    st.info('数据是历史分层样本，统计只能描述样本内结果；文字情绪由模型预测。数据没有完整的电影题材标签，因此无法给出可靠的“全部爱情片”等题材集合统计。')
    st.page_link('pages/4_智能问答.py',label='回到问答，试一个问题',icon=':material/chat_bubble:')


def render_guide():
    st.markdown('<div class="eyebrow">QUICK START & USER GUIDE</div>',unsafe_allow_html=True)
    st.title('第一次使用，从这里开始。')
    path=ROOT/'docs/USER_GUIDE.md'
    if not path.is_file():
        st.info('操作说明暂未随界面部署，请更新界面容器后重试。')
        return
    content=path.read_text(encoding='utf-8')
    st.markdown(content)
    st.download_button('下载完整操作说明（Markdown）',content,file_name='影评知图_操作说明.md',mime='text/markdown',key='download_user_guide',icon=':material/download:')
    for filename,title in [('PROJECT_DESIGN.md','理解项目与设计思路'),('GRADING.md','人工评分：拿不准时怎么判断')]:
        with st.expander(title):
            document=ROOT/'docs'/filename
            if not document.is_file():
                st.info('这份补充说明暂未随界面部署，其他功能可以照常使用。')
                continue
            text=document.read_text(encoding='utf-8')
            st.download_button('下载'+title,text,file_name=filename,mime='text/markdown',key=filename)
            st.markdown(text)
