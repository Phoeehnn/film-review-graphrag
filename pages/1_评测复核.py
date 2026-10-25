"""Explicit human marking. No answer is automatically labeled Y/P/N."""
import csv
import json
from pathlib import Path

import streamlit as st
from service.topics import friendly_text,friendly_value
from ui.grading_help import manual_link,translated_rubric

st.set_page_config(page_title='人工评测复核',layout='wide')
st.title('三路检索 · 人工评测复核')
st.caption('逐题比较真实回答与证据。Y=完整正确，P=部分有效，N=错误或无效；未评分时不发布准确率。')
manual_link()
root=Path(__file__).resolve().parents[1]
runs=sorted((root/'evaluation'/'runs').glob('*'),reverse=True)
runs=[p for p in runs if (p/'human-scores.csv').exists() and (p/'progress.json').exists()
      and json.loads((p/'progress.json').read_text(encoding='utf-8')).get('state')=='complete'
      and len(list((p/'responses').glob('*.json')))==168]
if not runs:
    st.info('168 次真实问答正在准备或运行；完成后可在此逐题评分。')
    st.stop()
run=st.selectbox('实验日期目录',runs,format_func=lambda p:p.name)
config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
st.caption('本次评分对应服务版本：'+config.get('service_version','1.0.0 基线')+'。旧实验原始回答保持不变，主题别名仅用于显示。')
if run != runs[0]:
    st.warning('你选的是旧实验。若要报告当前服务成绩，请切回上方最新的完整运行。')
fixture=json.loads((run/'questions.json').read_text(encoding='utf-8'))
questions=fixture['questions']
with (run/'human-scores.csv').open(encoding='utf-8-sig',newline='') as file:
    score_rows=list(csv.DictReader(file))
scores={(int(row['id']),row['strategy']):row for row in score_rows}
complete=sum(r['answer_correct'] in ('Y','P','N') for r in score_rows)
st.progress(complete/168,text=f'已评分 {complete}/168')
reviewer=st.text_input('评分人（保存时必填）',key='human_reviewer')
unfinished=[q for q in questions if any(scores[(q['id'],s)]['answer_correct'] not in ('Y','P','N') for s in ('graph','vector','hybrid'))]
only_pending=st.checkbox('仅显示未完成的题目',value=True)
available=unfinished if only_pending and unfinished else questions
question=st.selectbox('题目',available,format_func=lambda q:f"{q['id']} · {q['suite']} · {q['question']}")
st.subheader(question['question'])
with st.expander('本题在运行前固定的评分标准',expanded=True):
    for score,description in translated_rubric(question).items():
        st.markdown(f'**{score}**：{description}')
    st.caption('以上为原标准的中文说明。以原冻结标准为准，不因模型输出调整门槛。')
    with st.expander('原始评分标准（英文）'):st.json(question['rubric'])
reference_path=root/'evaluation'/'reference-facts.json'
if reference_path.exists():
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    if str(question['id']) in reference:
        with st.expander('从 CSV 独立计算的参考事实'):
            st.json(friendly_value(reference[str(question['id'])]))
choices=['未评分','Y','P','N']
columns=st.columns(3)
new_scores={}
for column,strategy in zip(columns,('graph','vector','hybrid')):
    with column:
        record=json.loads((run/'responses'/f"{question['id']:02d}_{strategy}.json").read_text(encoding='utf-8'))
        result=record['result']
        st.markdown('### '+strategy)
        st.caption(f"{result['status']} · {result['elapsed_seconds']} 秒")
        st.markdown(friendly_text(result['answer']))
        for warning in result.get('warnings',[]):
            st.caption(warning)
        with st.expander('证据、查询和错误'):
            for evidence in result['evidence']:
                st.markdown('**['+evidence['id']+'] '+evidence['kind']+'**')
                if evidence['kind']=='comment':
                    st.write(evidence['data']['text'])
                    st.json(friendly_value({k:v for k,v in evidence['data'].items() if k not in ('text','nodes','relationships')}))
                else:
                    st.json(friendly_value(evidence['data']))
                    st.code(evidence.get('cypher',''),language='cypher')
            if result.get('error'):
                st.json(result['error'])
        old=scores[(question['id'],strategy)]
        mark=st.selectbox('回答评分',choices,index=choices.index(old['answer_correct'] or '未评分'),key=f"grade_{run.name}_{question['id']}_{strategy}",format_func=lambda value:{'未评分':'未评分 · 待核实','Y':'Y · 完整正确','P':'P · 部分有效','N':'N · 错误或无效'}[value])
        refusal_options=['未评价','Y','N','不适用']
        old_refusal={'':'未评价','Y':'Y','N':'N','NA':'不适用'}.get(old['refusal_correct'],'未评价')
        refusal=st.selectbox('拒答是否正确（另记）',refusal_options,index=refusal_options.index(old_refusal),key=f"refusal_{run.name}_{question['id']}_{strategy}",format_func=lambda value:{'未评价':'未评价 · 待核实','Y':'Y · 拒答合理','N':'N · 拒答不合理','不适用':'不适用 · 没有拒答'}[value])
        notes=st.text_area('依据 / 失败原因',value=old['notes'],key=f"notes_{run.name}_{question['id']}_{strategy}")
        new_scores[strategy]={'answer_correct':'' if mark=='未评分' else mark,'refusal_correct':{'未评价':'','不适用':'NA','Y':'Y','N':'N'}[refusal],'notes':notes}
if st.button('保存本题三路评分',type='primary'):
    if not reviewer.strip():
        st.error('请填写评分人，保存的是人工标注。')
    else:
        for row in score_rows:
            if int(row['id'])==question['id']:
                row.update(new_scores[row['strategy']])
                row['reviewer']=reviewer.strip()
        target=run/'human-scores.csv'
        temporary=target.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['id','strategy','answer_correct','refusal_correct','reviewer','notes'])
            writer.writeheader()
            writer.writerows(score_rows)
        temporary.replace(target)
        from scripts.report import summarize
        summarize(run)
        st.success('已保存；报告已更新。')
        st.rerun()
with st.expander('当前报告'):
    report=run/'report.md'
    if report.exists():
        st.markdown(report.read_text(encoding='utf-8'))
st.download_button('下载人工评分 CSV',(run/'human-scores.csv').read_bytes(),file_name=run.name+'_human-scores.csv',mime='text/csv')
