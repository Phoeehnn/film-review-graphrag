"""Real bounded-repair and conversation ablations; semantic grades stay blank."""
import argparse
import asyncio
import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

import httpx
from scripts.evaluation_failures import require_available,ProviderUnavailable

from service.version import SERVICE_VERSION,runtime_source_hashes

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def equal_conditions(actual,expected):
    if set(actual)!=set(expected):return False
    for key,value in expected.items():
        if key=='movie':
            if str(value) not in str(actual[key]):return False
        elif actual[key]!=value:return False
    return True


def adopted_success(result):
    attempts=result.get('cypher_attempts') or []
    adopted=result.get('adopted_attempt')
    return any(a.get('attempt')==adopted and a.get('adopted') and a.get('executed')
        and a.get('result_kind') in ('rows','empty') for a in attempts)


def summarize(run):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/'responses').glob('*.json'))]
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    score_path=run/'human-scores.csv'
    fields=['record_id','answer_correct','refusal_correct','reviewer','notes']
    existing={}
    if score_path.exists():
        existing={r['record_id']:r for r in csv.DictReader(score_path.open(encoding='utf-8-sig'))}
    temporary=score_path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(existing.get(r['record_id'],{'record_id':r['record_id']}) for r in records)
    temporary.replace(score_path)
    groups={}
    for mode in sorted({r['mode'] for r in records}):
        subset=[r for r in records if r['mode']==mode]
        seconds=sorted(r['result']['elapsed_seconds'] for r in subset)
        attempts=[a for r in subset for a in r['result'].get('cypher_attempts',[])]
        generated=[r for r in subset if r['result'].get('cypher_attempts')]
        groups[mode]={'completed':len(subset),'mean_seconds':statistics.mean(seconds),'p50_seconds':statistics.median(seconds),
            'p95_seconds':seconds[min(len(seconds)-1,int(.95*len(seconds)))],
            'model_calls':sum(r['result'].get('model_usage',{}).get('calls',0) for r in subset),
            'total_tokens':sum(r['result'].get('model_usage',{}).get('total_tokens',0) for r in subset),
            'input_tokens':sum(r['result'].get('model_usage',{}).get('prompt_tokens',0) for r in subset),
            'errors':sum(r['result']['status']=='error' for r in subset),'timeouts':sum(r['result']['status']=='timeout' for r in subset),
            'generated_query_tasks':len(generated),'first_attempt_execution_success':sum(r['result']['cypher_attempts'][0].get('status') in ('ok','empty') for r in generated),
            # Final = the result the system actually adopted, not "any attempt ran".
            'final_execution_success':sum(adopted_success(r['result']) for r in generated),
            'any_attempt_execution_success':sum(any(a.get('status') in ('ok','empty') for a in r['result']['cypher_attempts']) for r in generated),
            'repeated_candidates':sum(a.get('status')=='repeated' for a in attempts),
            'scope_blocked':sum(a.get('failure_type')=='scope' for a in attempts),
            'scope_executed':sum(a.get('failure_type')=='scope' and a.get('executed') for a in attempts),
            'template_tasks':sum(any(e.get('data',{}).get('method')=='reviewed_metric' for e in r['result'].get('evidence',[])) for r in subset),
            'unsupported_before_generation':sum(r['result']['status']=='cannot_answer' and not r['result'].get('cypher_attempts') for r in subset),
            'conversation_prompt_tokens':sum(r['result'].get('model_usage',{}).get('conversation_prompt_tokens',0) for r in subset),
            'rewrite_prompt_tokens':sum(r['result'].get('model_usage',{}).get('conversation_phases',{}).get('rewrite',{}).get('prompt_tokens',0) for r in subset),
            'summary_prompt_tokens':sum(r['result'].get('model_usage',{}).get('conversation_phases',{}).get('summary',{}).get('prompt_tokens',0) for r in subset),
            'rewrite_calls':sum(r['result'].get('model_usage',{}).get('conversation_phases',{}).get('rewrite',{}).get('calls',0) for r in subset),
            'summary_calls':sum(r['result'].get('model_usage',{}).get('conversation_phases',{}).get('summary',{}).get('calls',0) for r in subset),
            'discarded_history_turns':sum(r['result'].get('conversation',{}).get('discarded_turns',0) for r in subset),
            'kept_history_turns':sum(r['result'].get('conversation',{}).get('kept_turns',0) for r in subset),
            'mean_attempts_generated_subset':statistics.mean(len(r['result']['cypher_attempts']) for r in generated) if generated else None,
            'unsafe_blocked':sum(a.get('failure_type')=='unsafe' for a in attempts),
            'unsafe_executed':sum(a.get('failure_type')=='unsafe' and a.get('executed') for a in attempts),
            'condition_match_count':sum(r.get('conditions_correct',False) for r in subset),
            'human_scored':sum(existing.get(r['record_id'],{}).get('answer_correct') in ('Y','P','N') for r in subset),
            'answer_accuracy':None}
        human=[existing.get(r['record_id'],{}).get('answer_correct','') for r in subset]
        for phase in ('rewrite','summary'):
            for key in ('calls','prompt_tokens','completion_tokens','total_tokens','elapsed_seconds'):
                groups[mode][phase+'_'+key]=sum(r['result'].get('model_usage',{}).get('conversation_phases',{}).get(phase,{}).get(key,0) for r in subset)
        if all(g in ('Y','P','N') for g in human):groups[mode]['answer_accuracy']=human.count('Y')/len(human)
    complete=len(records)==config['planned_requests']
    save(run/'summary.json',{'service_version':config['service_version'],'complete':complete,'completed':len(records),'planned':config['planned_requests'],'groups':groups,
        'labels_note':'No model or program assigns final answer grades; condition checks do not prove answer correctness.',
        'comparison_note':'Same questions/configuration, separate real generations at temperature 0; first Cypher can differ, so no claim of identical initial candidates.'})
    save(run/'full-results.json',records)
    if groups:
        with (run/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['mode']+list(next(iter(groups.values()))))
            writer.writeheader();writer.writerows({'mode':m,**g} for m,g in groups.items())
    lines=['# 多轮与自纠错真实对照','',f"服务{config['service_version']}，完成{len(records)}/{config['planned_requests']}。最终答案准确率待人工完整复核。",
        '', '| 组别 | 完成 | 平均/P95秒 | 条件匹配 | 模型输入tokens | 总tokens | 错误/超时 |', '|---|---:|---|---:|---:|---:|---|']
    for mode,g in groups.items():lines.append(f"| {mode} | {g['completed']} | {g['mean_seconds']:.2f}/{g['p95_seconds']:.2f} | {g['condition_match_count']} | {g['input_tokens']} | {g['total_tokens']} | {g['errors']}/{g['timeouts']} |")
    if config['kind']=='repair':
        lines+=['','| 组别 | 生成任务 | 模板/直接拒答 | 首次/采用/任一次执行成功 | 安全/范围拦截 | 重复候选 |','|---|---:|---|---|---|---:|']
        for mode,g in groups.items():lines.append(f"| {mode} | {g['generated_query_tasks']} | {g['template_tasks']}/{g['unsupported_before_generation']} | {g['first_attempt_execution_success']}/{g['final_execution_success']}/{g['any_attempt_execution_success']} | {g['unsafe_blocked']}/{g['scope_blocked']} | {g['repeated_candidates']} |")
    else:
        lines+=['','| 组别 | 改写调用/输入/输出 | 摘要调用/输入/输出 | 改写/摘要秒 | 保留/丢弃历史轮数总和 |','|---|---|---|---|---|']
        for mode,g in groups.items():lines.append(f"| {mode} | {g['rewrite_calls']}/{g['rewrite_prompt_tokens']}/{g['rewrite_completion_tokens']} | {g['summary_calls']}/{g['summary_prompt_tokens']}/{g['summary_completion_tokens']} | {g['rewrite_elapsed_seconds']:.2f}/{g['summary_elapsed_seconds']:.2f} | {g['kept_history_turns']}/{g['discarded_history_turns']} |")
    lines+=['','模板题与生成查询分开统计。执行过查询并不证明查询语义正确；空结果可能是正确结果。全量历史也受相同上下文预算约束，丢弃轮数逐轮记录。短对话与长对话分别运行、分别报告。']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')


async def evaluate(args):
    fixture=ROOT/'evaluation/v3'/('cypher-hard.json' if args.kind=='repair' else 'long-dialogues.json' if args.long else 'dialogues.json')
    raw=fixture.read_bytes();data=json.loads(raw)
    original=json.loads((ROOT/'evaluation/questions.json').read_text(encoding='utf-8'))['questions'][:50]
    questions=original+data['questions'] if args.kind=='repair' else data['groups']
    modes=['0','1','2'] if args.kind=='repair' else ['none','full','compact']
    planned=len(questions)*len(modes) if args.kind=='repair' else sum(len(g['turns']) for g in questions)*len(modes)
    run=args.output;run.mkdir(parents=True,exist_ok=True);(run/'responses').mkdir(exist_ok=True)
    async with httpx.AsyncClient(base_url=args.api,timeout=115) as client:
        health=(await client.get('/health')).json()
        if health.get('version')!=SERVICE_VERSION or health.get('runtime_source_sha256')!=runtime_source_hashes():raise ValueError('Deployed source mismatch')
        ready=await client.get('/ready');ready.raise_for_status()
        if ready.json().get('redis',{}).get('enabled'):raise ValueError('Disable Redis during evaluation')
        config={'service_version':SERVICE_VERSION,'runtime_source_sha256':runtime_source_hashes(),'fixture_sha256':hashlib.sha256(raw).hexdigest(),
                'started_at':datetime.now().astimezone().isoformat(),'kind':args.kind,'long_dialogues':args.long,'planned_requests':planned,
                'modes':modes,'readiness':ready.json(),'test_doubles':False,'cache_enabled':False,'temperature':0,'model':'deepseek-flash',
                'original_questions_sha256':hashlib.sha256((ROOT/'evaluation/questions.json').read_bytes()).hexdigest(),
                'references_sha256':hashlib.sha256((ROOT/'evaluation/v3/reference-facts.json').read_bytes()).hexdigest(),
                'context_budget_bytes':health.get('conversation_config',{}).get('context_budget_bytes'),
                'context_protocol':'Identical confirmations/questions; none ignores prior state; full is budgeted original history, compact is recent three original turns plus summary. Both obey identical UTF-8 byte budget; original history can be truncated.'}
        config_path=run/'run-config.json'
        if config_path.exists():
            old=json.loads(config_path.read_text(encoding='utf-8'))
            for key in ('runtime_source_sha256','fixture_sha256','original_questions_sha256','references_sha256'):
                if old[key]!=config[key]:raise ValueError('Frozen run changed; use a new directory')
        else:save(config_path,config)
        save(run/'fixture.json',data)
        save(run/'progress.json',{'state':'running','total':planned})
        sem=asyncio.Semaphore(args.concurrency)
        async def repair_one(q):
            async with sem:
                for mode in modes:
                    identifier=f"{q['id']}_{mode}";target=run/'responses'/(identifier+'.json')
                    if not target.exists():
                        response=await client.post('/api/v1/query',json={'question':q['question'],'strategy':'graph','max_cypher_rewrites':int(mode)})
                        response.raise_for_status();result=response.json()
                        require_available(result,run,identifier)
                        assert result['strategy']=='graph'
                        save(target,{'record_id':identifier,'question_id':q['id'],'question':q['question'],'mode':mode,'rubric':q['rubric'],'result':result})
                    print('repair',identifier,len(list((run/'responses').glob('*.json'))),'/',planned,flush=True)
        async def dialogue_one(group,mode):
            async with sem:
                targets=[run/'responses'/f"{group['id']}_{mode}_{i}.json" for i in range(1,len(group['turns'])+1)]
                if all(p.exists() for p in targets):return
                scores=run/'human-scores.csv'
                if scores.exists() and any(r.get('answer_correct') in ('Y','P','N') for r in csv.DictReader(scores.open(encoding='utf-8-sig'))):
                    raise ValueError('Human review has started; do not regenerate a partially completed conversation run')
                # A partially recorded group must be replayed to reconstruct state.
                response=await client.post('/api/v1/sessions');response.raise_for_status();session=response.json()
                try:
                    for i,turn in enumerate(group['turns'],1):
                        payload={'question':turn['question'],'client_request_id':f"{group['id']}_{mode}_{i}",'context_mode':mode,'confirm_conditions':turn['confirm_conditions']}
                        response=await client.post('/api/v1/sessions/'+session['id']+'/query',json=payload)
                        response.raise_for_status();result=response.json()
                        require_available(result,run,f"{group['id']}_{mode}_{i}")
                        expected=turn['expected_filters'];actual=result.get('conversation',{}).get('effective_conditions',{})
                        identifier=f"{group['id']}_{mode}_{i}"
                        record={'record_id':identifier,'group_id':group['id'],'turn':i,'mode':mode,'question':turn['question'],
                            'gold':turn,'conditions_correct':equal_conditions(actual,expected),'result':result,
                            'rubric':{'Y':'Correct referent and effective conditions, supported final answer or correct predefined empty/missing-author outcome.','P':'Useful supported requested subtask only.','N':'Wrong referent/scope, unsupported claims, task not completed, runtime failure.'}}
                        save(targets[i-1],record)
                        print('dialogue',identifier,result['status'],'scope',record['conditions_correct'],len(list((run/'responses').glob('*.json'))),'/',planned,flush=True)
                finally:await client.delete('/api/v1/sessions/'+session['id'])
        try:
            if args.kind=='repair':await asyncio.gather(*(repair_one(q) for q in questions))
            else:await asyncio.gather(*(dialogue_one(g,m) for g in questions for m in modes))
        except Exception as exc:
            save(run/'progress.json',{'state':'waiting_for_provider' if isinstance(exc,ProviderUnavailable) else 'failed','error_type':type(exc).__name__,'total':planned,'completed':len(list((run/'responses').glob('*.json')))})
            summarize(run);raise
        summarize(run)
        save(run/'progress.json',{'state':'complete','total':planned,'completed':planned,'completed_at':datetime.now().astimezone().isoformat()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['repair','conversation']);p.add_argument('--long',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--api',default='http://127.0.0.1:18000');p.add_argument('--concurrency',type=int,default=2,choices=[1,2,3])
    asyncio.run(evaluate(p.parse_args()))
