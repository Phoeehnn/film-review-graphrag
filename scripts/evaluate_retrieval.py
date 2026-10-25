"""CPU-only retrieval components. Relevance is left for a human, not word rules."""
import asyncio
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from service.graph import GraphStore
from service.retrieval import Encoder,Retriever
from service.routing import comment_scope_violations,resolve_filters,rrf
from service.schemas import Filters
from service.settings import Settings
from service.topics import topic_config
from service.version import SERVICE_VERSION,runtime_source_hashes


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


async def main():
    settings=Settings()
    fixture_path=Path('evaluation/retrieval/questions.json')
    fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
    output=Path('evaluation/retrieval/runs')/datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output.mkdir(parents=True)
    started=time.monotonic()
    config={'started_at':datetime.now().astimezone().isoformat(),'service_version':SERVICE_VERSION,
        'fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        'source_sha256':runtime_source_hashes(),'embedding_model':settings.embedding_model,
        'embedding_revision':settings.embedding_revision,'device':'cpu','normalized_embeddings':True,
        'vector_backend':settings.vector_backend,'llm_calls':0,'reranker':None,
        'candidate_limit':20,'top_k':5,'rrf_k':60,'weights':{'graph':1,'vector':1},
        'component_protocol':'Vector top20 prefix@5 compared with scoped-liked+vector20 RRF@5; no Text-to-Cypher, generation or statistics.',
        'ann_note':'无筛选为原生ANN。独立组件取20条后截5，与完整API向量模式直接取5的候选宽度不同，不能用本试验代替完整问答评测。',
        'human_relevance_labels':'pending; never inferred from has_results, scores or term matches'}
    save(output/'run-config.json',config)
    save(output/'questions.json',fixture)
    save(output/'topic-aliases.json',topic_config())
    graph=GraphStore(settings)
    encoder=Encoder(settings)
    retriever=Retriever(settings,graph,encoder)
    results=[]
    labels=[]
    try:
        with httpx.Client(base_url=settings.api_base_url,timeout=20) as client:
            state=client.get('/ready');state.raise_for_status()
        config['readiness']=state.json()
        movies,topics=await graph.catalog()
        warm=time.monotonic()
        await encoder.load()
        config['model_load_seconds']=round(time.monotonic()-warm,3)
        for q in fixture['questions']:
            scope=resolve_filters(q['query'],Filters.model_validate(q['filters']),movies,topics)
            begin=time.monotonic()
            vector=await retriever.vector(q['query'],scope,20)
            vector_seconds=time.monotonic()-begin
            begin=time.monotonic()
            liked=await graph.liked_candidates(scope,20) if scope.has_scope() else []
            graph_seconds=time.monotonic()-begin
            for mode,chosen in [('vector',vector[:5]),('hybrid_candidates',rrf({'graph':liked,'vector':vector},5))]:
                comments,_=await graph.comments([candidate['comment_id'] for candidate in chosen])
                hydrated={c['comment_id']:c for c in comments}
                items=[]
                for rank,candidate in enumerate(chosen,1):
                    data=hydrated[candidate['comment_id']]
                    violations=comment_scope_violations(data,scope)
                    comment={k:v for k,v in data.items() if k not in ('nodes','relationships')}
                    items.append({'rank':rank,'candidate':candidate,'comment':comment,'facet_violations':violations})
                    labels.append({'question_id':q['id'],'mode':mode,'rank':rank,'comment_id':data['comment_id'],
                        'relevance':'','reviewer':'','notes':'','movie':data['movie'],'star':data.get('star'),
                        'topic_display_name':data.get('topic_display_name'),'text':data.get('text'),
                        'facet_violations':','.join(violations)})
                result={'question_id':q['id'],'group':q['group'],'mode':mode,'query':q['query'],'criteria':q['criteria'],
                    'resolved_filters':scope.params(),'vector_retrieval_seconds':round(vector_seconds,4),
                    'graph_candidates_seconds':round(graph_seconds,4) if mode=='hybrid_candidates' else 0,
                    'graph_channel_degraded':mode=='hybrid_candidates' and not scope.has_scope(),'comments':items}
                results.append(result)
            save(output/'results.json',results)
            print(f"{q['id']}: two retrieval components complete",flush=True)
        fields=list(labels[0])
        with (output/'relevance.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(labels)
        config['total_wall_seconds']=round(time.monotonic()-started,3)
        config['completed_at']=datetime.now().astimezone().isoformat()
        save(output/'run-config.json',config)
        save(Path('var/retrieval-evaluation-current.json'),{'run_dir':str(output),'human_relevance_complete':False})
        from scripts.report_retrieval import summarize
        summarize(output)
        print('Saved CPU retrieval baseline: '+str(output),flush=True)
    finally:
        await graph.close()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main())
