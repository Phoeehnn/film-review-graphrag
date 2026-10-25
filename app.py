"""Streamlit presentation shell; application data is retrieved through the API."""
import streamlit as st

from ui.presentation import apply_theme,brand

st.set_page_config(page_title='影评知图 · 影视评论知识图谱',page_icon='◈',layout='wide')
apply_theme()

with st.sidebar:
    brand()
    menu=st.container()
    research=st.toggle('显示研究工具',key='research_tools',value=False,
        help='打开后显示问答、检索质量和 Agent 人工复核。默认演示界面隐藏它们；已有评分和答案不会删除。')

home=st.Page('pages/4_智能问答.py',title='智能问答',icon=':material/chat_bubble:',default=True)
about=st.Page('pages/0_项目与图谱.py',title='项目与图谱',icon=':material/hub:',url_path='about')
guide=st.Page('pages/3_使用指南.py',title='使用指南',icon=':material/menu_book:',url_path='guide')
pages=[home,about,guide]
if research:
    pages.extend([
        st.Page('pages/1_评测复核.py',title='问答评测复核',icon=':material/fact_check:',url_path='answer-review'),
        st.Page('pages/2_检索质量复核.py',title='检索质量复核',icon=':material/checklist:',url_path='retrieval-review'),
        st.Page('pages/5_Agent任务复核.py',title='Agent任务复核',icon=':material/assignment_turned_in:',url_path='agent-review'),
        st.Page('pages/6_多轮与修复复核.py',title='多轮与修复复核',icon=':material/compare_arrows:',url_path='conversation-review')])
navigation=st.navigation(pages,position='hidden')
with menu:
    for page in pages[:3]:st.page_link(page)
    if research:
        st.caption('研究与实验')
        for page in pages[3:]:st.page_link(page)
navigation.run()
