"""Chinese presentation of the already-frozen rubric; never assigns a grade."""
from pathlib import Path

import streamlit as st


def manual_link(kind='answer'):
    path=Path(__file__).resolve().parents[1]/'docs/GRADING.md'
    operations=path.with_name('GRADING.md')
    if operations.exists():
        with st.expander('新版人工复核：一步步操作'):
            st.markdown(operations.read_text(encoding='utf-8'))
    with st.expander('拿不准怎么选？先看评分小抄',expanded=False):
        if kind=='retrieval':
            st.markdown('**2：全部要点相关且满足筛选。1：只满足部分要点。0：无关、意思相反或违反筛选。**')
            st.caption('1不是“不确定”。拿不准可以先留空，读评论原文并对照本题的全部要点。')
        else:
            st.markdown('**Y：重要要求都完成且有依据。P：确实答对一部分，但没完成。N：关键错误或没有有效答案。**')
            st.caption('P不是中立；不确定可以先留空。语言生硬不等于事实错误。回答是否完成与拒答是否合理分开评价。')
            st.info('常见 vector 情况：问全库数量、平均分或占比，只因证据不足而合理拒答，且未完成其他子任务 → 回答 N，拒答 Y。确实完成了题目要求的一部分才考虑 P；拒答合理本身不会让回答变成 Y。完整指南有逐项例子。')
        if path.exists():
            content=path.read_text(encoding='utf-8')
            st.download_button('下载人工测评判断指南',content,file_name='GRADING.md',mime='text/markdown',key='download_scoring_guide')
            with st.expander('在页面中阅读完整指南'):st.markdown(content)


def translated_rubric(question):
    if question['id']<=50:
        return {'Y':'回答完整、准确，由样本证据和真实引用支持；所需统计必须来自聚合查询。',
            'P':'部分正确，或明确说明证据缺失，同时提供了一些有用且有依据的内容。',
            'N':'错误或无依据的结论、范围错误、运行错误、超时，或没有有用的回答。'}
    y={51:'至少两条评论表达困境中的希望或自由；引用真实评论ID，不声称全库普遍如此。',
        52:'至少两条评论同时提到剧情薄弱/散乱和演技好；真实ID且忠实解释。仅夸演技属于部分满足。',
        53:'至少两条评论同时涉及缺点/批评和五星或强烈认可；真实ID，不把预测情绪当人工判断。',
        54:'正确的样本正面占比及分母、预测情绪分布、至少两条相关评论解释喜欢原因；不能由召回的几条评论估算总体统计。',
        55:'正确的电影样本矛盾评论占比，至少两条相关评论；解释评分与预测差异，不把弱监督情绪当真实人工标签。',
        56:'正确的样本正面评论数最高电影及数量，至少两条来自同一目标电影的评论；不能换成正面率或总评分最高。'}
    return {'Y':y[question['id']],
        'P':'只满足统计或语义要求的一部分，或只有一条有用的相关评论，并如实说明局限。',
        'N':'统计错误、解释无依据、证据不相关、运行错误或超时。'}
