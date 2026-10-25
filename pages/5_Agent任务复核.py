"""Human review of final Agent task outcomes; trace metrics remain separate."""
import csv
import json
from pathlib import Path
from ui.run_selection import agent_complete

import streamlit as st

from scripts.report_agent_human import report
from ui.grading_help import manual_link


st.set_page_config(page_title='Agent任务复核', layout='wide')
st.title('Agent · 最终任务人工复核')
st.caption('20 个问题各运行 3 次。这里只判断最终任务是否完成；工具选择正确不等于答案正确。')
manual_link()
root = Path(__file__).resolve().parents[1]
runs = sorted((root / 'evaluation/agent/runs').glob('*'), reverse=True)
runs = [path for path in runs if agent_complete(path)]
if not runs:
    st.info('暂无可用的 Agent 评测数据。运行结果与评分记录不随公开仓库提供；准备好完整数据后可在此复核。')
    st.stop()
fixture = json.loads((root / 'evaluation/agent/questions.json').read_text(encoding='utf-8'))
run = st.selectbox('实验目录', runs, format_func=lambda path: path.name)
config = json.loads((run / 'run-config.json').read_text(encoding='utf-8'))
st.caption(f"本次回答对应服务版本：{config['service_version']}；请核对最新运行目录后再评分。")
if run != runs[0]:
    st.warning('目前选择的是旧运行。')
with (run / 'human-task-scores.csv').open(encoding='utf-8-sig', newline='') as handle:
    labels = list(csv.DictReader(handle))
scored = sum(row['task_success'] in ('Y', 'N') for row in labels)
st.progress(scored / 60, text=f'已评分 {scored}/60 次最终任务')
reviewer = st.text_input('评分人（保存时必填）', key='agent_reviewer')
questions = fixture['questions']
pending = {row['question_id'] for row in labels if row['task_success'] == ''}
only_pending = st.checkbox('只显示未完成问题', value=True)
available = [q for q in questions if q['id'] in pending] if only_pending and pending else questions
question = st.selectbox('问题', available, format_func=lambda q: f"{q['id']} · {q['question']}")
st.subheader(question['question'])
st.caption('预期工具由开发者在运行前填写，供参考；请自行判断预期工具是否合理。最终分数仍以答案能否完成问题为准。')
st.json({'预期必须调用': question['required'], '允许调用': question['allowed']})
st.info('Y：最终回答完整解决问题，范围、关键数字和引用证据可信。N：没有完成、核心事实错误或关键结论缺证据。暂时拿不准就留空，写下疑点。')
columns = st.columns(3)
updates = {}
for repeat, column in enumerate(columns, 1):
    with column:
        record = json.loads((run / 'responses' / f"{question['id']}_{repeat}.json").read_text(encoding='utf-8'))
        result = record['result']
        old = next(row for row in labels if row['question_id'] == question['id'] and int(row['repetition']) == repeat)
        st.markdown(f'### 第 {repeat} 次')
        st.caption(f"状态：{result.get('status')}；自动工具轨迹：{'合格' if record['trace_score']['tool_selection_correct'] else '需检查'}")
        st.write(result.get('answer', ''))
        with st.expander('工具、证据与错误'):
            for call in result.get('tool_calls', []):
                st.json(call)
            for item in result.get('evidence', []):
                st.json(item)
            if result.get('error'):
                st.json(result['error'])
        choices = ['未评分', 'Y', 'N']
        mark = st.selectbox('任务是否完成', choices, index=choices.index(old['task_success'] or '未评分'),
                            key=f"agent_score_{run.name}_{question['id']}_{repeat}",
                            format_func=lambda value: {'未评分': '未评分 · 待核实', 'Y': 'Y · 完成', 'N': 'N · 未完成'}[value])
        note = st.text_area('判断依据 / 失败原因', value=old['notes'],
                            key=f"agent_note_{run.name}_{question['id']}_{repeat}")
        updates[repeat] = {'task_success': '' if mark == '未评分' else mark, 'notes': note}
if st.button('保存本题三次评分', type='primary'):
    if not reviewer.strip():
        st.error('请先填评分人。')
    else:
        for row in labels:
            if row['question_id'] == question['id']:
                row.update(updates[int(row['repetition'])])
                row['reviewer'] = reviewer.strip()
        target = run / 'human-task-scores.csv'
        temporary = target.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['question_id', 'repetition', 'task_success', 'reviewer', 'notes'])
            writer.writeheader()
            writer.writerows(labels)
        temporary.replace(target)
        report(run)
        st.success('已保存，Agent 任务报告已更新。')
        st.rerun()
st.download_button('下载 Agent 人工评分 CSV', (run / 'human-task-scores.csv').read_bytes(),
                   file_name=run.name + '_human-task-scores.csv', mime='text/csv')
