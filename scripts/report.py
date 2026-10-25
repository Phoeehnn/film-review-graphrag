"""Report operational metrics immediately; accuracy only after human Y/P/N marks."""
import argparse
import csv
import json
import hashlib
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def frozen_route(run):
    """Replay the route used by this run, not a later edited service version."""
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    release=Path(__file__).resolve().parents[1]/'releases/2026-09-29_v1.1/source.zip'
    with zipfile.ZipFile(release) as archive:
        source=archive.read('service/routing.py')
    assert hashlib.sha256(source).hexdigest()==config['runtime_source_sha256']['service/routing.py']
    scope={'__name__':'service.routing_frozen','__package__':'service'}
    exec(compile(source,'frozen service/routing.py','exec'),scope)
    return scope['route']


def auto_route_replay(run,records,scores):
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    decisions=config.get('auto_route_decisions')
    legacy_route=frozen_route(run) if decisions is None else None
    questions=json.loads((run/'questions.json').read_text(encoding='utf-8'))['questions']
    by_key={(r['id'],r['requested_strategy']):r for r in records}
    output=[]
    for suite in ('original_50','supplementary_6'):
        selected=[];best=[];latencies=[];choices=Counter()
        for question in (q for q in questions if q['suite']==suite):
            identifier=question['id']
            strategy=decisions[str(identifier)] if decisions is not None else legacy_route(question['question'],'auto')[0]
            choices[strategy]+=1
            marks=[scores[(identifier,mode)]['answer_correct'] for mode in ('graph','vector','hybrid')]
            selected.append(scores[(identifier,strategy)]['answer_correct'])
            best.append('Y' if 'Y' in marks else 'P' if 'P' in marks else 'N' if all(m in ('Y','P','N') for m in marks) else '')
            latencies.append(by_key[(identifier,strategy)]['result']['elapsed_seconds'])
        selected_complete=all(m in ('Y','P','N') for m in selected)
        oracle_complete=all(m in ('Y','P','N') for m in best)
        output.append({'suite':suite,'questions':len(selected),'route_counts':dict(choices),
            'auto_human_scored':sum(m in ('Y','P','N') for m in selected),
            'auto_strict_accuracy':selected.count('Y')/len(selected) if selected_complete else None,
            'auto_effective_rate':sum(m in ('Y','P') for m in selected)/len(selected) if selected_complete else None,
            'oracle_strict_upper_bound':best.count('Y')/len(best) if oracle_complete else None,
            'oracle_effective_upper_bound':sum(m in ('Y','P') for m in best)/len(best) if oracle_complete else None,
            'selected_forced_mode_mean_seconds':float(np.mean(latencies)),
            'definition':f"运行前冻结的{config.get('service_version','1.1.0')}规则路由选择本次强制模式答案；离线反事实估计，并非重新执行auto请求。理想上限假设每题事后知道三路中人工分数最高者。"})
    (run/'auto-route-replay.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    return output


def summarize(run):
    records=json.loads((run/'full-results.json').read_text(encoding='utf-8'))
    scores={}
    with (run/'human-scores.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['answer_correct'] not in ('','Y','P','N'):
                raise ValueError('人工评分仅允许 Y/P/N 或留空')
            scores[(int(row['id']),row['strategy'])]=row
    summaries=[]
    for suite in ('original_50','supplementary_6'):
        for strategy in ('graph','vector','hybrid'):
            rows=[r for r in records if r['suite']==suite and r['requested_strategy']==strategy]
            latencies=[r['result']['elapsed_seconds'] for r in rows]
            marks=[scores.get((r['id'],strategy),{}).get('answer_correct','') for r in rows]
            human_complete=all(mark in ('Y','P','N') for mark in marks)
            status=Counter(r['result']['status'] for r in rows)
            refused=[r for r in rows if r['result']['status'] in ('cannot_answer','insufficient_evidence','no_results')]
            refusal_marks=[scores.get((r['id'],strategy),{}).get('refusal_correct','') for r in refused]
            summary={'suite':suite,'strategy':strategy,'total':len(rows),'human_scored':sum(m in ('Y','P','N') for m in marks),'human_complete':human_complete,'Y':marks.count('Y'),'P':marks.count('P'),'N':marks.count('N'),'strict_accuracy':marks.count('Y')/len(rows) if human_complete else None,'effective_answer_rate':(marks.count('Y')+marks.count('P'))/len(rows) if human_complete else None,'mean_latency_seconds':float(np.mean(latencies)),'p95_latency_seconds':float(np.percentile(latencies,95)),'error_rate':status['error']/len(rows),'timeout_rate':status['timeout']/len(rows),'statuses':dict(status),'prompt_tokens':sum(r['result']['model_usage'].get('prompt_tokens',0) for r in rows),'completion_tokens':sum(r['result']['model_usage'].get('completion_tokens',0) for r in rows),'total_tokens':sum(r['result']['model_usage'].get('total_tokens',0) for r in rows),'refusal_count':len(refused),'refusal_human_scored':sum(m in ('Y','N') for m in refusal_marks),'refusal_correct_rate':refusal_marks.count('Y')/len(refused) if refused and all(m in ('Y','N') for m in refusal_marks) else None}
            stages=sorted({stage for record in rows for stage in record['result'].get('stage_timings',{})})
            summary['stage_timings']={stage:{'requests':sum(stage in record['result'].get('stage_timings',{}) for record in rows),
                'mean_over_all_requests_seconds':float(np.mean([record['result'].get('stage_timings',{}).get(stage,0) for record in rows])),
                'p95_over_all_requests_seconds':float(np.percentile([record['result'].get('stage_timings',{}).get(stage,0) for record in rows],95))} for stage in stages}
            summary['model_calls']=sum(record['result']['model_usage'].get('calls',0) for record in rows)
            summary['zero_model_call_requests']=sum(record['result']['model_usage'].get('calls',0)==0 for record in rows)
            summary['corrected_answers']=sum(record['result'].get('evidence_checks',{}).get('corrected_answer',False) for record in rows)
            summary['model_identifiers']=sorted({model for record in rows for model in record['result']['model_usage'].get('models',[])})
            summaries.append(summary)
    (run/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    replay=auto_route_replay(run,records,scores)
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    lines=['# 三路检索真实评测','',f"运行目录：{run.name}；服务版本：{config.get('service_version','1.1.0')}；源码指纹见 run-config.json。",'', '原 50 题与新增 6 题分开报告；严格准确率 Y/总题数，有效回答率 (Y+P)/总题数。未完成人工评分时不发布准确率。','', '| 题集 | 策略 | 人工评分 | 严格准确率 | 有效率 | 平均/P95 秒 | 错误/超时率 | tokens |','|---|---|---|---|---|---|---|---|']
    for row in summaries:
        accuracy='待人工评分' if row['strict_accuracy'] is None else f"{row['strict_accuracy']:.1%}"
        effective='待人工评分' if row['effective_answer_rate'] is None else f"{row['effective_answer_rate']:.1%}"
        lines.append(f"| {row['suite']} | {row['strategy']} | {row['human_scored']}/{row['total']} | {accuracy} | {effective} | {row['mean_latency_seconds']:.2f}/{row['p95_latency_seconds']:.2f} | {row['error_rate']:.1%}/{row['timeout_rate']:.1%} | {row['total_tokens']} |")
    lines += ['', '拒答正确性在 summary.json 中单独记录，不等同于回答准确率。', '', '论文原有 98% 是有效回答率；原严格回答准确率为 76%。新运行不沿用旧人工标注。', '', '测试替身只用于自动化回归，没有进入以上真实回答成绩。', '', '分阶段耗时、模型调用次数、零调用范围拒答次数、字段冲突后的回答修正次数见 summary.json。未执行的阶段在全题平均/P95中计为0；它们不代表单独阶段运行时的平均。']
    lines+=['','## 自动路由的离线反事实估计','',
        f"使用本次运行时冻结的{config.get('service_version','1.1.0')}路由决定，从已完成的三种强制模式里选答案。没有重新执行auto请求；理想上限是假定事后知道每题三路人工分数的上界。未完成人工标注时不发布效果数字。", '',
        '| 题集 | 规则选择图/向量/融合 | 已标注 | 自动路由严格/有效 | 理想上限严格/有效 | 被选模式平均秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in replay:
        count=row['route_counts'];choice=f"{count.get('graph',0)}/{count.get('vector',0)}/{count.get('hybrid',0)}"
        fmt=lambda v:'待人工评分' if v is None else f'{v:.1%}'
        lines.append(f"| {row['suite']} | {choice} | {row['auto_human_scored']}/{row['questions']} | {fmt(row['auto_strict_accuracy'])}/{fmt(row['auto_effective_rate'])} | {fmt(row['oracle_strict_upper_bound'])}/{fmt(row['oracle_effective_upper_bound'])} | {row['selected_forced_mode_mean_seconds']:.2f} |")
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    summarize(args.run_dir)
    print('Wrote report.md and summary.json; unscored accuracy stays null.')
