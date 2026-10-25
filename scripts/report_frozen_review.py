"""Read saved 1.4.1 answers/labels; report follow-ups without modifying inputs."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = '2026-10-03_v1.4.1_final'


def summarize_group(records, labels):
    grades = [labels.get(r['record_id'], {}).get('answer_correct', '') for r in records]
    invalid = sorted(set(grades) - {'', 'Y', 'P', 'N'})
    if invalid:
        raise ValueError('Invalid human grades: ' + ','.join(invalid))
    total = len(records)
    scored = sum(g in ('Y', 'P', 'N') for g in grades)
    complete = total > 0 and scored == total
    return {'required': total, 'scored': scored, 'remaining': total-scored,
            'Y': grades.count('Y'), 'P': grades.count('P'), 'N': grades.count('N'),
            'human_complete': complete,
            'strict_accuracy': grades.count('Y')/total if complete else None,
            'effective_answer_rate': (grades.count('Y')+grades.count('P'))/total if complete else None}


def selected_records(records, kind, repair_modes=('0','2')):
    if kind == 'repair':
        return [r for r in records if r['mode'] in repair_modes]
    return [r for r in records if int(r['turn']) >= 2]


def build_report(tag=TAG, repair_modes=('0','2')):
    output = {'tag':tag, 'version':'1.4.1', 'created_at':datetime.now().astimezone().isoformat(),
              'read_only_inputs':True, 'model_calls':0, 'repair_modes':list(repair_modes),
              'repair_comparison_coverage':set(repair_modes)>={'0','2'}, 'experiments':{}, 'input_sha256':{},
              'notes':[
                  'Only turns >=2 are scored for follow-up accuracy; none/full/compact all remain.',
                  'Repair modes 0 and 2 are the minimum for the v3 human accuracy comparison; mode 1 is optional.',
                  'A mode-2-only report gives that arm accuracy, not the requested 0-vs-2 human comparison.',
                  'No retry occurred in the formal run; different outputs do not establish repair gains.',
                  'D10-D12 turn 4: expected_filters includes positive sentiment; abbreviated independent question omits it. Use the frozen conditions as an explanatory annotation; do not edit the gold.',
                  'Do not copy 1.4.1 accuracy or human grades onto later modified answers.'
              ]}
    for kind in ('repair','conversation','long-conversation'):
        directory=ROOT/'evaluation/v3/runs'/kind/tag
        config=json.loads((directory/'run-config.json').read_text(encoding='utf-8'))
        if config['service_version']!='1.4.1': raise ValueError('This rubric selection is for frozen 1.4.1 only')
        paths=[directory/'full-results.json', directory/'human-scores.csv', directory/'run-config.json']
        records=json.loads(paths[0].read_text(encoding='utf-8'))
        with paths[1].open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
        labels={r['record_id']:r for r in rows}
        if len(labels)!=len(rows): raise ValueError('Duplicate human record IDs')
        chosen=selected_records(records,kind,repair_modes)
        if not all(r['record_id'] in labels for r in chosen): raise ValueError('Missing human score rows')
        modes=sorted(set(r['mode'] for r in chosen))
        groups={m:summarize_group([r for r in chosen if r['mode']==m],labels) for m in modes}
        output['experiments'][kind]={'groups':groups, 'selected_record_ids':[r['record_id'] for r in chosen],
            'omitted_first_turns':sum(r.get('turn')==1 for r in records) if kind!='repair' else 0,
            'required':len(chosen),'scored':sum(g['scored'] for g in groups.values()),
            'human_complete':all(g['human_complete'] for g in groups.values())}
        for path in paths: output['input_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def write_report(report, directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'followup-and-repair-human.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['kind','mode','required','scored','remaining','Y','P','N','human_complete','strict_accuracy','effective_answer_rate']
    with (directory/'followup-and-repair-human.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({'kind':k,'mode':m,**g} for k,e in report['experiments'].items() for m,g in e['groups'].items())
    lines=['# 1.4.1 人工追问与修复组评分进度','',
        '从冻结答案及人类CSV只读计算，不重问模型，不修改原始评分或原summary。未评齐时准确率留空。',
        '', '| 实验 | 组别 | 已评/需评 | 严格准确率 | 有效回答率 |','|---|---|---:|---:|---:|']
    for kind,e in report['experiments'].items():
        for mode,g in e['groups'].items():
            value=lambda key: '待人工完成' if g[key] is None else f'{g[key]:.2%}'
            lines.append(f"| {kind} | {mode} | {g['scored']}/{g['required']} | {value('strict_accuracy')} | {value('effective_answer_rate')} |")
    lines+=['','0/2两组人工比较覆盖：'+str(report['repair_comparison_coverage']),
            '正式批次重写触发0次，不能把两组差异归因于自纠错。',
            'D10–D12第4轮按expected_filters中的正面条件评判，简略完整问题漏写该条件不作为系统失败。']
    (directory/'followup-and-repair-human.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default=TAG)
    parser.add_argument('--repair-modes',nargs='+',choices=['0','1','2'],default=['0','2'])
    parser.add_argument('--output',type=Path,default=ROOT/'results/v1.4.1')
    args=parser.parse_args()
    report=build_report(args.tag,tuple(dict.fromkeys(args.repair_modes)))
    write_report(report,args.output)
    print(json.dumps({'tag':args.tag,'groups':{k:{m:[g['scored'],g['required']] for m,g in e['groups'].items()} for k,e in report['experiments'].items()},'inputs_modified':False,'model_calls':0},ensure_ascii=False))
