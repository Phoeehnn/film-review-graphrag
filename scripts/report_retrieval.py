"""Report candidate-pool retrieval metrics while preserving score provenance."""
import argparse
import csv
import json
import math
from pathlib import Path


def _review_status(row):
    explicit = row.get('review_status', '')
    if explicit:
        return explicit
    reviewer = row.get('reviewer', '')
    if reviewer and '待用户复核' not in reviewer:
        return 'human_confirmed'  # compatibility with old, human-labelled runs
    return 'unreviewed'


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
            raise ValueError('已有分数必须记录评分来源')
        row['_review_status']=_review_status(row)
        index[(row['question_id'],row['mode'],row['comment_id'])]=row

    summary=[]
    for mode in ('vector','hybrid_candidates'):
        subset=[r for r in results if r['mode']==mode]
        positions=[(r,c) for r in subset for c in r['comments']]
        position_rows=[index[(r['question_id'],mode,c['comment']['comment_id'])] for r,c in positions]
        marks=[row['relevance'] for row in position_rows]
        scores_complete=bool(marks) and all(mark!='' for mark in marks)
        p5=[];ndcg=[]
        if scores_complete:
            for result in subset:
                values=[int(index[(result['question_id'],mode,c['comment']['comment_id'])]['relevance']) if not c['facet_violations'] else 0 for c in result['comments']]
                values += [0] * (5-len(values))
                p5.append(sum(v==2 for v in values)/5)
                dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values))
                ideal=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(sorted(values,reverse=True)))
                ndcg.append(dcg/ideal if ideal else 0)
        pending=sum(row['_review_status']=='ai_pending' for row in position_rows)
        confirmed=sum(row['_review_status']=='human_confirmed' and row['relevance']!='' for row in position_rows)
        human_complete=bool(position_rows) and all(row['_review_status']=='human_confirmed' and row['relevance'] for row in position_rows)
        if human_complete:
            review_state='人工复核完成'
        elif scores_complete and pending==len(position_rows):
            review_state='AI初评（待人工复核）'
        elif scores_complete and pending:
            review_state='人工复核进行中（未复核项暂用AI建议）'
        elif scores_complete:
            review_state='分数已填，来源待核对'
        else:
            review_state='仍有未评分项'
        summary.append({
            'mode':mode,'question_count':len(subset),'returned_positions':len(positions),
            'human_scored':confirmed,'scored_positions':sum(mark!='' for mark in marks),
            'ai_pending':pending,'review_state':review_state,
            'human_complete':human_complete,'scores_complete':scores_complete,
            'fully_relevant_precision_at_5':sum(p5)/len(p5) if scores_complete else None,
            'candidate_pool_ndcg_at_5':sum(ndcg)/len(ndcg) if scores_complete else None,
            'corpus_recall_at_5':None,'recall_limitation':'未穷举全库相关评论，无法计算真实全库Recall@5。',
            'facet_violation_positions':sum(bool(c['facet_violations']) for r,c in positions),
            'empty_questions':sum(not r['comments'] for r in subset),
            'mean_vector_component_seconds':sum(r['vector_retrieval_seconds'] for r in subset)/len(subset),
            'mean_graph_candidate_seconds':sum(r['graph_candidates_seconds'] for r in subset)/len(subset),
        })

    summary_data={
        'run_dir':str(run),'config':config,'summary':summary,
        'unique_query_comment_pairs_to_review':len({(row['question_id'],row['comment_id']) for row in labels}),
        'human_complete':all(row['human_complete'] for row in summary),
        'scores_complete':all(row['scores_complete'] for row in summary),
        'has_ai_pending':any(row['ai_pending'] for row in summary),
    }
    fixture=json.loads((run/'questions.json').read_text(encoding='utf-8'))
    protocol=fixture.get('protocol',{}).get('five_semantic_acceptance')
    semantic=[]
    if protocol:
        for mode in ('vector','hybrid_candidates'):
            questions=[]
            for identifier in protocol['question_ids']:
                result=next(r for r in results if r['mode']==mode and r['question_id']==identifier)
                items=result['comments']
                item_rows=[index[(identifier,mode,c['comment']['comment_id'])] for c in items]
                marks=[row['relevance'] for row in item_rows]
                scores_ready=all(mark!='' for mark in marks)
                human_ready=scores_ready and all(row['_review_status']=='human_confirmed' for row in item_rows)
                count=sum(mark=='2' and not c['facet_violations'] for mark,c in zip(marks,items))
                questions.append({
                    'question_id':identifier,'labels_complete':scores_ready,'human_confirmed':human_ready,
                    'fully_relevant_count':count if scores_ready else None,
                    'reasonable':count>=protocol['reasonable_min_fully_relevant_comments_in_top5'] if scores_ready else None,
                })
            human_complete=all(q['labels_complete'] and q['human_confirmed'] for q in questions)
            scores_available=all(q['labels_complete'] for q in questions)
            reasonable=sum(q['reasonable'] is True for q in questions)
            semantic.append({
                'mode':mode,'questions':questions,'human_complete':human_complete,
                'scores_available':scores_available,
                'reasonable_questions':reasonable if scores_available else None,
                'passed':reasonable>=protocol['pass_min_reasonable_questions'] if human_complete else None,
            })
    summary_data['five_semantic_acceptance']=semantic
    (run/'summary.json').write_text(json.dumps(summary_data,ensure_ascii=False,indent=2),encoding='utf-8')

    if summary_data['human_complete']:
        title='# 独立检索组件基线（人工复核完成）'
        batch_confirmed=any(row.get('confirmation_source')=='user_chat_batch_acceptance' for row in labels)
        provenance=('评分由Codex/Sol提供，用户于2026年10月19日在聊天中确认整批接受，按AI辅助、人工确认口径报告；不是132项独立逐条人工评分，也未记录为网页逐项签认。' if batch_confirmed else '相关性分数已由评分人保存确认；若使用AI辅助，须保留对应评分来源。')
    elif summary_data['has_ai_pending']:
        title='# 独立检索组件基线（AI初评，待人工复核）'
        provenance='当前P@5/nDCG为AI建议分计算的预览值，不能表述为人工结果；逐题确认后报告会更新。'
    else:
        title='# 独立检索组件基线（人工复核进行中）'
        provenance='指标基于当前已填分数；尚未确认的项目仍可能包含AI建议，不能表述为人工最终结果。'
    lines=[title,'',provenance,
        f"CPU总用时（含模型加载）：{config['total_wall_seconds']}秒；模型加载{config['model_load_seconds']}秒；大模型调用0次；未加入重排。",'',
        '20个预先固定问题，各模式返回前5。vector为语义候选20条截取前5；hybrid_candidates为实际RRF评论融合组件。没有运行图谱统计与回答生成，不替代168次完整问答评测。','',
        '| 组件 | 已评分/返回位置 | 人工确认 | 状态 | 完全相关P@5 | 候选集合nDCG@5 | 硬筛选违规 | 空结果题 | 平均向量/图候选秒 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in summary:
        p='暂无完整评分' if row['fully_relevant_precision_at_5'] is None else f"{row['fully_relevant_precision_at_5']:.1%}"
        n='暂无完整评分' if row['candidate_pool_ndcg_at_5'] is None else f"{row['candidate_pool_ndcg_at_5']:.3f}"
        lines.append(f"| {row['mode']} | {row['scored_positions']}/{row['returned_positions']} | {row['human_scored']}/{row['returned_positions']} | {row['review_state']} | {p} | {n} | {row['facet_violation_positions']} | {row['empty_questions']} | {row['mean_vector_component_seconds']:.3f}/{row['mean_graph_candidate_seconds']:.3f} |")
    lines += [
        '',f"去重题目/评论组合：{summary_data['unique_query_comment_pairs_to_review']}；200个候选位置各自保留来源，评分表区分网页保存和用户聊天整批确认。",'',
        '完全相关P@5只把2分计为相关；缺失位置计0。nDCG的理想排序由已返回候选构成，仅反映该候选集合的排序，不声称全库nDCG或Recall。','',
        '先看多条件语义的失误原因，再决定是否在相同候选和同一标签上进行CPU小模型重排对照。不得据时间、非空结果或词命中宣称重排提高了相关性。',
    ]
    if semantic:
        lines += ['', '## 前五道语义题的最低验收线','',
            '运行前固定规则：R01–R05每题前5条至少2条达到2分，至少3题满足。两个组件分别统计，不合并取最好结果；仅人工复核完成后，才形成正式验收结论。']
        for row in semantic:
            status='待人工复核' if row['passed'] is None else ('通过' if row['passed'] else '未通过')
            count='暂无评分' if row['reasonable_questions'] is None else str(row['reasonable_questions'])
            lines.append(f"- {row['mode']}：当前分数下合理题数{count}/5；正式验收{status}。")
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary_data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path)
    summarize(parser.parse_args().run_dir)
