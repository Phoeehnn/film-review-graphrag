"""Macro precision@5/nDCG@5 only when every returned item is human-labelled.

Missing result slots count as zero relevance. A globally estimated recall@k
cannot be calculated without a complete relevant set; do not invent it.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def summarize(run):
    results=json.loads((run/'results.json').read_text(encoding='utf-8'))
    config=json.loads((run/'run-config.json').read_text(encoding='utf-8'))
    with (run/'relevance.csv').open(encoding='utf-8-sig',newline='') as handle:
        labels=list(csv.DictReader(handle))
    index={}
    for row in labels:
        if row['relevance'] not in ('','0','1','2'):
            raise ValueError('相关性仅允许0/1/2或留空')
        if row['relevance'] and not row['reviewer'].strip():
            raise ValueError('人工相关性标注需要评分人')
        index[(row['question_id'],row['mode'],row['comment_id'])]=row
    summary=[]
    for mode in ('vector','hybrid_candidates'):
        subset=[r for r in results if r['mode']==mode]
        positions=[(r,c) for r in subset for c in r['comments']]
        marks=[index[(r['question_id'],mode,c['comment']['comment_id'])]['relevance'] for r,c in positions]
        complete=bool(marks) and all(mark!='' for mark in marks)
        p5=[];ndcg=[]
        if complete:
            for r in subset:
                values=[int(index[(r['question_id'],mode,c['comment']['comment_id'])]['relevance']) if not c['facet_violations'] else 0 for c in r['comments']]
                values+= [0]*(5-len(values))
                p5.append(sum(v==2 for v in values)/5)
                # This is candidate-pool nDCG; IDCG comes from this returned set,
                # not a known corpus-wide relevant set. Label the limit explicitly.
                dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values))
                ideal=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(sorted(values,reverse=True)))
                ndcg.append(dcg/ideal if ideal else 0)
        row={'mode':mode,'question_count':len(subset),'returned_positions':len(positions),
            'human_scored':sum(mark!='' for mark in marks),'human_complete':complete,
            'fully_relevant_precision_at_5':sum(p5)/len(p5) if complete else None,
            'candidate_pool_ndcg_at_5':sum(ndcg)/len(ndcg) if complete else None,
            'corpus_recall_at_5':None,'recall_limitation':'未穷举全库相关评论，无法计算真实全库Recall@5。',
            'facet_violation_positions':sum(bool(c['facet_violations']) for r,c in positions),
            'empty_questions':sum(not r['comments'] for r in subset),
            'mean_vector_component_seconds':sum(r['vector_retrieval_seconds'] for r in subset)/len(subset),
            'mean_graph_candidate_seconds':sum(r['graph_candidates_seconds'] for r in subset)/len(subset)}
        summary.append(row)
    summary_data={'run_dir':str(run),'config':config,'summary':summary,
        'unique_query_comment_pairs_to_review':len({(row['question_id'],row['comment_id']) for row in labels}),
        'human_complete':all(row['human_complete'] for row in summary)}
    fixture=json.loads((run/'questions.json').read_text(encoding='utf-8'))
    protocol=fixture.get('protocol',{}).get('five_semantic_acceptance')
    semantic=[]
    if protocol:
        for mode in ('vector','hybrid_candidates'):
            questions=[]
            for identifier in protocol['question_ids']:
                result=next(r for r in results if r['mode']==mode and r['question_id']==identifier)
                items=result['comments']
                marks=[index[(identifier,mode,c['comment']['comment_id'])]['relevance'] for c in items]
                checked=all(m!='' for m in marks)
                count=sum(m=='2' and not c['facet_violations'] for m,c in zip(marks,items))
                questions.append({'question_id':identifier,'labels_complete':checked,
                    'fully_relevant_count':count if checked else None,
                    'reasonable':count>=protocol['reasonable_min_fully_relevant_comments_in_top5'] if checked else None})
            complete=all(q['labels_complete'] for q in questions)
            reasonable=sum(q['reasonable'] is True for q in questions)
            semantic.append({'mode':mode,'questions':questions,'human_complete':complete,
                'reasonable_questions':reasonable if complete else None,
                'passed':reasonable>=protocol['pass_min_reasonable_questions'] if complete else None})
    summary_data['five_semantic_acceptance']=semantic
    (run/'summary.json').write_text(json.dumps(summary_data,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 独立检索组件基线','',f"CPU总用时（含模型加载）：{config['total_wall_seconds']}秒；模型加载{config['model_load_seconds']}秒；大模型调用0次；未加入重排。",'',
        '20个预先固定问题，各模式返回前5。vector为语义候选20条截取前5；hybrid_candidates为实际RRF评论融合组件。没有运行图谱统计与回答生成，不替代168次完整问答评测。', '',
        '| 组件 | 已标注/返回位置 | 完全相关P@5 | 候选集合nDCG@5 | 硬筛选违规 | 空结果题 | 平均向量/图候选秒 |',
        '|---|---|---|---|---|---|---|']
    for r in summary:
        p='待人工复核' if r['fully_relevant_precision_at_5'] is None else f"{r['fully_relevant_precision_at_5']:.1%}"
        n='待人工复核' if r['candidate_pool_ndcg_at_5'] is None else f"{r['candidate_pool_ndcg_at_5']:.3f}"
        lines.append(f"| {r['mode']} | {r['human_scored']}/{r['returned_positions']} | {p} | {n} | {r['facet_violation_positions']} | {r['empty_questions']} | {r['mean_vector_component_seconds']:.3f}/{r['mean_graph_candidate_seconds']:.3f} |")
    lines+=['',f"去重后需人工阅读的题目/评论组合：{summary_data['unique_query_comment_pairs_to_review']}。相关性0/1/2规则已在运行前冻结，尚未自动打分。",'',
        '完全相关P@5只把2分计为相关；缺失位置计0。nDCG的理想排序由已返回候选构成，仅反映该候选集合的排序，不声称全库nDCG或Recall。', '',
        '先看多条件语义的失误原因，再决定是否在相同候选和同一人工标签上进行CPU小模型重排对照。不得据时间、非空结果或词命中宣称重排提高了相关性。']
    if semantic:
        lines+=['','## 前五道语义题的最低验收线','',
            '使用运行前固定的规则：R01–R05每题前5条至少2条达到人工2分，至少3题满足。两个组件分别统计，不合并取最好结果。']
        for row in semantic:
            status='待人工标注' if row['passed'] is None else ('通过' if row['passed'] else '未通过')
            count='待确定' if row['reasonable_questions'] is None else str(row['reasonable_questions'])
            lines.append(f"- {row['mode']}：合理题数{count}/5；{status}。")
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary_data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path)
    summarize(parser.parse_args().run_dir)
