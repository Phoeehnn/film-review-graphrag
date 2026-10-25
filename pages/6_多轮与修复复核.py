"""Human semantic review of the new ablations; automatic scope checks stay separate."""
import csv
import json
from pathlib import Path

import streamlit as st

from ui.explorer import show_result

ROOT=Path(__file__).resolve().parents[1]
st.title('多轮与查询修复 · 人工复核')
st.caption('这里评价最终任务是否完成。条件自动匹配、执行成功和最终回答正确是不同指标。原来的三类复核保留。')
kind=st.radio('实验类别',['conversation','repair','long-conversation'],format_func=lambda s:{'conversation':'四轮追问','repair':'Cypher自纠错','long-conversation':'长对话压缩'}[s])
runs=sorted((ROOT/'evaluation/v3/runs'/kind).glob('*'),reverse=True)
runs=[r for r in runs if (r/'progress.json').exists() and json.loads((r/'progress.json').read_text(encoding='utf-8')).get('state')=='complete' and (r/'human-scores.csv').exists()]
if not runs:
    st.info('这组新版真实对照尚未完成，完成后才能在此评分。');st.stop()
run=st.selectbox('运行目录',runs,format_func=lambda p:p.name)
config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
st.caption('服务版本：'+config['service_version']+'；不要混用其他版本的答案。')
records=json.loads((run/'full-results.json').read_text(encoding='utf-8'))
target=run/'human-scores.csv'
rows=list(csv.DictReader(target.open(encoding='utf-8-sig')));old={r['record_id']:r for r in rows}
complete=sum(r['answer_correct'] in ('Y','P','N') for r in rows)
st.progress(complete/len(rows),text=f'已评分 {complete}/{len(rows)}')
reviewer=st.text_input('评分人',key='v3_reviewer')
pending=st.checkbox('仅显示未评分',value=True)
available=[r for r in records if old[r['record_id']]['answer_correct'] not in ('Y','P','N')] if pending else records
if not available:available=records
r=st.selectbox('记录',available,format_func=lambda v:v['record_id']+' · '+v['question'])
st.subheader(r['question']);st.caption('对照组：'+r['mode'])
if 'gold' in r:
    with st.expander('预期独立问题与条件',expanded=True):
        st.write(r['gold']['gold_independent_question']);st.json(r['gold']['expected_filters'])
        if r['gold'].get('expected_reference'):st.json(r['gold']['expected_reference'])
    st.caption('自动条件匹配：'+str(r.get('conditions_correct'))+'；只是条件检查，不能代替答案评分。')
else:
    references=ROOT/'evaluation/v3/reference-facts.json'
    if references.exists():
        facts=json.loads(references.read_text(encoding='utf-8'))['facts']
        if str(r['question_id']) in facts:
            with st.expander('CSV独立计算的参考事实'):st.json(facts[str(r['question_id'])])
    if isinstance(r['question_id'],int):
        facts=json.loads((ROOT/'evaluation/reference-facts.json').read_text(encoding='utf-8'))
        if str(r['question_id']) in facts:
            with st.expander('原题参考事实'):st.json(facts[str(r['question_id'])])
with st.expander('本题冻结标准'):st.json(r['rubric'])
st.markdown(r['result']['answer']);show_result(r['result'])
score=old[r['record_id']]
choices=['未评分','Y','P','N']
grade=st.selectbox('回答评分',choices,index=choices.index(score['answer_correct'] or '未评分'),key='v3_grade_'+r['record_id'])
refusal_choices=['未评价','Y','N','NA']
refusal=st.selectbox('拒答是否正确',refusal_choices,index=refusal_choices.index(score['refusal_correct'] or '未评价'),key='v3_refusal_'+r['record_id'],format_func=lambda v:{'未评价':'未评价','Y':'Y · 合理','N':'N · 不合理','NA':'不适用'}[v])
notes=st.text_area('依据或失败原因',value=score['notes'],key='v3_notes_'+r['record_id'])
if st.button('保存这一项',type='primary'):
    if not reviewer.strip():st.error('请填写评分人。')
    else:
        score.update(answer_correct='' if grade=='未评分' else grade,refusal_correct='' if refusal=='未评价' else refusal,reviewer=reviewer.strip(),notes=notes)
        temporary=target.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(target)
        from scripts.evaluate_v3 import summarize
        summarize(run);st.rerun()
st.download_button('下载本组评分CSV',target.read_bytes(),file_name=kind+'_'+run.name+'_human-scores.csv',mime='text/csv')
