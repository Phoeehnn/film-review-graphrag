"""Run 56 frozen questions x 3 forced strategies with real API/model calls."""
import argparse
import asyncio
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import httpx

from service.settings import ROOT, Settings
from service.version import SERVICE_VERSION, runtime_source_hashes
from service.routing import route
from scripts.evaluation_failures import require_available,ProviderUnavailable


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def source_hashes():
    paths = sorted([*(ROOT/'service').glob('*.py'),*(ROOT/'service').glob('*.json'),*(ROOT/'pages').glob('*.py')])
    paths += [ROOT/name for name in ['requirements-service.txt','requirements-dev.txt','compose.yaml','Dockerfile','app.py','config.py','graph_chain.py','scripts/initialize.py','scripts/evaluate.py','scripts/report.py']]
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


async def evaluate(args):
    settings = Settings()
    fixture_path = ROOT/'evaluation/questions.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    run = args.output or ROOT/'evaluation'/'runs'/datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run.mkdir(parents=True,exist_ok=True)
    responses = run/'responses'
    responses.mkdir(exist_ok=True)
    async with httpx.AsyncClient(base_url=settings.api_base_url,timeout=settings.query_timeout+20) as client:
        health=await client.get('/health')
        health.raise_for_status()
        deployed=health.json()
        if deployed.get('version')!=SERVICE_VERSION or deployed.get('runtime_source_sha256')!=runtime_source_hashes():
            raise ValueError('部署版本与当前服务源码不同，请先完成部署再启动新版本评测')
        readiness = await client.get('/ready')
        readiness.raise_for_status()
        state = readiness.json()
        config = {'started_at':datetime.now().astimezone().isoformat(),'python':platform.python_version(),'platform':platform.platform(),'question_fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),'source_sha256':source_hashes(),'strategies':['graph','vector','hybrid'],'temperature':settings.generation_temperature,'max_tokens':settings.generation_max_tokens,'thinking':'disabled','configured_model':settings.deepseek_model,'embedding_model':settings.embedding_model,'embedding_dimension':384,'normalized_embeddings':True,'vector_backend':settings.vector_backend,'top_k':5,'rrf_k':60,'rrf_weights':{'graph':1,'vector':1},'graph_candidate_limit':20,'vector_candidate_limit':20,'timeout_seconds':settings.query_timeout,'database_timeout_seconds':settings.database_timeout,'concurrency':args.concurrency,'readiness':state,'human_scores':'not yet collected','test_doubles':False}
        config['embedding_revision']=settings.embedding_revision
        config['service_version']=SERVICE_VERSION
        config['runtime_source_sha256']=deployed['runtime_source_sha256']
        config['auto_route_decisions']={str(q['id']):route(q['question'],'auto')[0] for q in fixture['questions']}
        config['metric_policy']='reviewed generic metric templates with Text-to-Cypher for remaining questions'
        config['evidence_validation']='bounded deterministic checks; no automatic Y/P/N grades'
        config['scope_policy']='movie genres unavailable; explicit scope refusal without model call'
        old_path = run/'run-config.json'
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding='utf-8'))
            if old['source_sha256']!=config['source_sha256'] or old['question_fixture_sha256']!=config['question_fixture_sha256']:
                raise ValueError('代码/题集变化，请使用新的日期运行目录，不可混入已有实验')
            config = old
        else:
            write_json(old_path,config)
            write_json(run/'questions.json',fixture)
        write_json(ROOT/'var'/'evaluation-current.json',{'run_dir':str(run),'started_at':config['started_at']})
        progress = {'state':'running','total':168,'completed':0,'errors':0,'timeouts':0,'run_dir':str(run)}
        semaphore = asyncio.Semaphore(args.concurrency)
        results = []
        async def one(question,strategy):
            async with semaphore:
                filename = f"{question['id']:02d}_{strategy}.json"
                target = responses/filename
                if target.exists():
                    record = json.loads(target.read_text(encoding='utf-8'))
                else:
                    response = await client.post('/api/v1/query',json={'question':question['question'],'strategy':strategy,'top_k':5,'filters':{}})
                    response.raise_for_status()
                    result = response.json()
                    require_available(result,run,filename)
                    if result['strategy']!=strategy:
                        raise ValueError(f'强制策略不一致: {strategy} -> {result["strategy"]}')
                    record = {'id':question['id'],'suite':question['suite'],'type':question['type'],'question':question['question'],'requested_strategy':strategy,'run_date':config['started_at'],'result':result}
                    write_json(target,record)
                results.append(record)
                progress['completed'] += 1
                progress['errors'] += int(record['result']['status']=='error')
                progress['timeouts'] += int(record['result']['status']=='timeout')
                write_json(run/'progress.json',progress)
                print(f"{progress['completed']}/168 id={question['id']} {strategy} {record['result']['status']} {record['result']['elapsed_seconds']}s",flush=True)
        try:
            await asyncio.gather(*(one(q,strategy) for q in fixture['questions'] for strategy in config['strategies']))
        except Exception as exc:
            progress.update(state='waiting_for_provider' if isinstance(exc,ProviderUnavailable) else 'failed',failure=type(exc).__name__+': '+str(exc)[:500])
            write_json(run/'progress.json',progress)
            raise
        results.sort(key=lambda r:(r['id'],r['requested_strategy']))
        write_json(run/'full-results.json',results)
        fields=['id','suite','type','strategy','question','status','answer','evidence_ids','elapsed_seconds','service_version','stage_timings','model','model_calls','prompt_tokens','completion_tokens','total_tokens','error','answer_correct','refusal_correct','reviewer','notes']
        with (run/'results.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            for record in results:
                result,usage=record['result'],record['result']['model_usage']
                writer.writerow({'id':record['id'],'suite':record['suite'],'type':record['type'],'strategy':record['requested_strategy'],'question':record['question'],'status':result['status'],'answer':result['answer'],'evidence_ids':','.join(e['id'] for e in result['evidence']),'elapsed_seconds':result['elapsed_seconds'],'service_version':result.get('service_version'),'stage_timings':json.dumps(result.get('stage_timings',{}),ensure_ascii=False),'model':','.join(usage.get('models',[])),'model_calls':usage.get('calls',0),'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),'total_tokens':usage.get('total_tokens',0),'error':json.dumps(result.get('error'),ensure_ascii=False) if result.get('error') else ''})
        score_path = run/'human-scores.csv'
        if not score_path.exists():
            with score_path.open('w',encoding='utf-8-sig',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=['id','strategy','answer_correct','refusal_correct','reviewer','notes'])
                writer.writeheader()
                writer.writerows({'id':r['id'],'strategy':r['requested_strategy']} for r in results)
        progress.update(state='complete',completed_at=datetime.now().astimezone().isoformat())
        write_json(run/'progress.json',progress)
        from scripts.report import summarize
        summarize(run)
        print('Real evaluation complete: '+str(run),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--concurrency',type=int,default=2,choices=[1,2,3])
    asyncio.run(evaluate(parser.parse_args()))
