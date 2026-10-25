"""Resume-safe real Agent tool-trajectory evaluation; never grades answer semantics."""
import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
from scripts.evaluation_failures import require_available

from service.version import SERVICE_VERSION, runtime_source_hashes


ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'evaluation/agent/questions.json'


def score_trace(result,question):
    calls=result.get('tool_calls',[])
    successful={c['tool'] for c in calls if c.get('ok')}
    unexpected=sorted({c['tool'] for c in calls}-{*question['allowed']})
    missing=sorted(set(question['required'])-successful)
    failed=[c for c in calls if not c.get('ok')]
    status=result.get('status')
    budget=any('达到五轮' in warning or '十次工具' in warning for warning in result.get('warnings',[]))
    correct=not (missing or unexpected or failed or budget or status in ('error','timeout'))
    if any(c.get('error',{}).get('code')=='INVALID_ARGUMENTS' for c in failed):stage='参数抽取'
    elif failed:stage='执行'
    elif missing or unexpected:stage='工具选择'
    elif budget or status in ('error','timeout','partial','insufficient_evidence','no_results'):stage='回复或终态'
    else:stage=None
    return {'tool_selection_correct':correct,'missing_required':missing,'unexpected_tools':unexpected,
            'failed_tools':[{'tool':c['tool'],'error':c.get('error')} for c in failed],
            'model_rounds':result.get('model_usage',{}).get('calls',0),'tool_executions':len(calls),
            'budget_triggered':budget,'automatic_failure_stage':stage,
            'answer_semantics':'待独立人工复核；工具轨迹正确不代表最终答案正确'}


def summarize(run,questions):
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    records=[]
    for question in questions:
        for repeat in range(1,4):
            path=run/'responses'/f"{question['id']}_{repeat}.json"
            if path.exists():records.append(json.loads(path.read_text(encoding='utf-8')))
    complete=len(records)==len(questions)*3
    by_id={question['id']:[r for r in records if r['question_id']==question['id']] for question in questions}
    pass3=sum(len(rows)==3 and all(r['trace_score']['tool_selection_correct'] for r in rows) for rows in by_id.values()) if complete else None
    summary={'service_version':config['service_version'],'question_count':len(questions),'planned_calls':len(questions)*3,
        'completed_calls':len(records),'complete':complete,
        'tool_selection_correct_rate':sum(r['trace_score']['tool_selection_correct'] for r in records)/len(records) if complete else None,
        'mean_model_rounds':sum(r['trace_score']['model_rounds'] for r in records)/len(records) if complete else None,
        'mean_tool_executions':sum(r['trace_score']['tool_executions'] for r in records)/len(records) if complete else None,
        'budget_trigger_rate':sum(r['trace_score']['budget_triggered'] for r in records)/len(records) if complete else None,
        'pass_power_3_tool_trajectory':pass3/len(questions) if complete else None,
        'failure_stage_counts':dict(Counter(r['trace_score']['automatic_failure_stage'] for r in records if r['trace_score']['automatic_failure_stage'])),
        'total_model_tokens':sum(r['result'].get('model_usage',{}).get('total_tokens',0) for r in records),
        'task_success_pass_power_3':None,'task_success_note':'回答语义和任务终态未经独立人工标注；不能把工具轨迹pass^3冒充任务pass^3。',
        'fixture_note':'期望工具由项目开发者在运行前编写，尚非独立人工标注。'}
    (run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fmt=lambda value:'待完成' if value is None else f'{value:.1%}'
    lines=['# Agent 六工具真实评测','',f"服务版本：{config['service_version']}；已完成 {len(records)}/{len(questions)*3} 次真实请求。",
        f"工具轨迹选择正确率：{fmt(summary['tool_selection_correct_rate'])}；三次都选对（工具轨迹 pass^3）：{fmt(summary['pass_power_3_tool_trajectory'])}。",
        f"平均模型轮数：{summary['mean_model_rounds'] if summary['mean_model_rounds'] is not None else '待完成'}；平均工具执行数：{summary['mean_tool_executions'] if summary['mean_tool_executions'] is not None else '待完成'}；预算触发率：{fmt(summary['budget_trigger_rate'])}。",
        '', '这些指标只评估工具调用轨迹。回答是否答对、任务是否真正完成仍需人工复核，因此当前不发布任务成功率或任务 pass^3。',
        '预期工具由项目开发者按六种工具能力预先填写，不作为独立人工真值；完整请求、响应、失败原因在 responses/ 中。']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


async def evaluate(run,api):
    fixture_bytes=FIXTURE.read_bytes();fixture=json.loads(fixture_bytes)
    questions=fixture['questions'];assert len(questions)==20 and fixture['repetitions']==3
    run.mkdir(parents=True,exist_ok=True);(run/'responses').mkdir(exist_ok=True)
    config_path=run/'run-config.json'
    config={'started_at':datetime.now().astimezone().isoformat(),'service_version':SERVICE_VERSION,
            'fixture_sha256':hashlib.sha256(fixture_bytes).hexdigest(),'runtime_source_sha256':runtime_source_hashes(),
            'api':api,'repetitions':3,'judgment_scope':'tool trajectory only; human task success pending'}
    if config_path.exists():
        previous=json.loads(config_path.read_text(encoding='utf-8'))
        for key in ('service_version','fixture_sha256','runtime_source_sha256','api'):
            assert config[key]==previous[key],f'Cannot resume with changed {key}'
    else:
        config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    async with httpx.AsyncClient(timeout=145) as client:
        ready=(await client.get(api+'/ready')).json()
        assert ready['ready'],'API not ready'
        assert ready['database']['manifest']['nodes']['Comment']==123693
        assert (await client.get(api+'/health')).json()['runtime_source_sha256']==runtime_source_hashes(),'Deployed API source differs'
        for question in questions:
            for repeat in range(1,4):
                target=run/'responses'/f"{question['id']}_{repeat}.json"
                if target.exists():continue
                response=await client.post(api+'/api/v1/agent',json={'question':question['question']})
                response.raise_for_status()
                result=response.json()
                require_available(result,run,f"{question['id']}_{repeat}")
                record={'question_id':question['id'],'repetition':repeat,'question':question['question'],
                    'expected_required':question['required'],'expected_allowed':question['allowed'],
                    'run_date':datetime.now().astimezone().isoformat(),'result':result,
                    'trace_score':score_trace(result,question)}
                target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                summary=summarize(run,questions)
                print(f"{question['id']} {repeat}/3 status={result['status']} tool_correct={record['trace_score']['tool_selection_correct']} progress={summary['completed_calls']}/60",flush=True)
    from scripts.report_agent_human import report
    report(run)
    (run/'review-status.json').write_text(json.dumps({'reviewable':True}),encoding='utf-8')
    return summarize(run,questions)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/f'evaluation/agent/runs/{datetime.now():%Y-%m-%d}_{SERVICE_VERSION}')
    parser.add_argument('--api',default='http://127.0.0.1:18000')
    args=parser.parse_args()
    asyncio.run(evaluate(args.output,args.api))
