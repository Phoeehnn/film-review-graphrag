"""Human relevance annotations for the independent CPU retrieval baseline."""
import csv
import json
from pathlib import Path

import streamlit as st

from scripts.report_retrieval import summarize
from ui.grading_help import manual_link
from ui.run_selection import retrieval_complete


st.set_page_config(page_title='检索质量复核',layout='wide')
st.title('评论检索质量 · 人工复核')
st.caption('独立检查评论是否相关，不评价大模型答案。2=满足全部要点，1=部分相关，0=无关；硬筛选违规计0。')
manual_link('retrieval')
root=Path(__file__).resolve().parents[1]
runs=sorted((root/'evaluation/retrieval/runs').glob('*'),reverse=True)
runs=[run for run in runs if retrieval_complete(run)]
if not runs:
    st.info('独立检索基线尚未完成，完成后可在这里复核。')
    st.stop()
run=st.selectbox('运行日期',runs,format_func=lambda path:path.name)
config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
st.caption('本次评分对应服务版本：'+config.get('service_version','历史基线')+'。诊断与未完成运行不会进入列表。')
questions=json.loads((run/'questions.json').read_text(encoding='utf-8'))['questions']
results=json.loads((run/'results.json').read_text(encoding='utf-8'))
with (run/'relevance.csv').open(encoding='utf-8-sig',newline='') as handle:
    labels=list(csv.DictReader(handle))
scored=sum(row['relevance']!='' for row in labels)
st.progress(scored/len(labels) if labels else 0,text=f'已标注 {scored}/{len(labels)} 个结果位置，同题同评论会同步到两个组件。')
reviewer=st.text_input('评分人（保存时必填）')
question=st.selectbox('问题',questions,format_func=lambda q:q['id']+' · '+q['query'])
st.subheader(question['query'])
st.write('完整相关必须同时满足：'+'；'.join(question['criteria']))
st.json({'筛选':question['filters'],'分组':question['group']})
st.caption('模型预测情绪是元数据，不能作为评论实际文字语义的真值。')
index={(row['question_id'],row['comment_id']):row for row in labels}
updates={}
seen=set()
for result in (r for r in results if r['question_id']==question['id']):
    st.markdown('### '+{'vector':'向量候选前5','hybrid_candidates':'RRF融合评论候选前5'}[result['mode']])
    st.caption(f"向量组件 {result['vector_retrieval_seconds']:.3f} 秒；图候选 {result['graph_candidates_seconds']:.3f} 秒")
    if not result['comments']:
        st.info('本题本组件没有评论，缺失位置按0计入P@5。')
    for candidate in result['comments']:
        comment=candidate['comment'];identifier=comment['comment_id']
        with st.expander(f"{candidate['rank']} · [{identifier}] {comment['movie']} · {comment.get('star')}星",expanded=True):
            st.write(comment['text'])
            st.caption(f"主题 {comment.get('topic_display_name')} · 预测情绪 {comment.get('emotion')} · 一致性 {comment.get('layer_consistency')} · 日期 {comment.get('date')}")
            if candidate['facet_violations']:
                st.warning('硬筛选违规：'+', '.join(candidate['facet_violations'])+'；汇总时按0计。')
            if identifier in seen:
                st.caption('同题同一评论，使用上面已经填写的评分。')
                continue
            seen.add(identifier)
            old=index[(question['id'],identifier)]
            choices=['未评分','2','1','0']
            mark=st.selectbox('相关性',choices,index=choices.index(old['relevance'] or '未评分'),key=f"relevance_{run.name}_{question['id']}_{identifier}",format_func=lambda value:{'未评分':'未评分 · 待核实','2':'2 · 全部要点相关','1':'1 · 部分相关','0':'0 · 无关或违反条件'}[value])
            note=st.text_input('判断依据',value=old['notes'],key=f"notes_{run.name}_{question['id']}_{identifier}")
            updates[identifier]={'relevance':'' if mark=='未评分' else mark,'notes':note}
if st.button('保存本题相关性',type='primary'):
    if not reviewer.strip():
        st.error('请填写评分人，保存的是人工标注。')
    else:
        for row in labels:
            if row['question_id']==question['id']:
                row.update(updates[row['comment_id']]);row['reviewer']=reviewer.strip()
        target=run/'relevance.csv';temporary=target.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(labels[0]));writer.writeheader();writer.writerows(labels)
        temporary.replace(target)
        summarize(run)
        st.success('已保存，组件指标已更新。')
        st.rerun()
with st.expander('当前组件报告'):
    st.markdown((run/'report.md').read_text(encoding='utf-8'))
st.caption('nDCG仅使用已返回候选作理想集合；未穷举全库相关评论，不能宣称真实全库Recall。')
