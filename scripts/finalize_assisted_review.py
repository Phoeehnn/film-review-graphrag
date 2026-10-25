"""User-requested small random audit; keep all AI uncertainty visible and preserve prior human grades."""
import csv
import json
from collections import Counter
from datetime import datetime

from scripts.assisted_review import ROOT, TAG, OUT, digest, file_hash, read, summarize


def freeze_sampling():
    path = OUT / 'sampling-revision.json'
    if path.exists():
        return read(path)
    records = read(OUT / 'dataset.json')
    sources = [(ROOT / f'evaluation/runs/{TAG}/human-scores.csv', 'qa'),
               (ROOT / f'evaluation/agent/runs/{TAG}/human-task-scores.csv', 'agent')]
    sources += [(ROOT / f'evaluation/v3/runs/{k}/{TAG}/human-scores.csv', k) for k in ('repair', 'conversation', 'long-conversation')]
    prior = {}
    for source, kind in sources:
        for row in csv.DictReader(source.open(encoding='utf-8-sig')):
            grade = row.get('answer_correct', row.get('task_success', ''))
            if grade not in ('Y', 'P', 'N') or not row.get('reviewer', '').strip():
                continue
            if kind == 'qa':
                key = f"qa/{int(row['id']):02d}_{row['strategy']}"
            elif kind == 'agent':
                key = f"agent/{row['question_id']}_{row['repetition']}"
            else:
                key = f"{kind}/{row['record_id']}"
            if key in {r['key'] for r in records}:
                prior[key] = {'grade': grade, 'refusal_correct': row.get('refusal_correct', ''),
                              'reviewer': row['reviewer'], 'note': row.get('notes', ''),
                              'source_path': source.relative_to(ROOT).as_posix(), 'source_sha256': file_hash(source),
                              'provenance': 'existing_human_label_preserved'}
    # Labels/grades never enter ranking. User reduced the earlier 56-sample plan during execution.
    # Record that timing explicitly rather than claiming this revision preceded all grading.
    eligible = [r for r in records if r['key'] not in prior]
    rank = lambda r: digest('user-small-random-audit-2026-10-17-v1:' + r['key'])
    chosen = []
    for group in sorted({(r['kind'], r['mode']) for r in eligible}):
        group_rows = sorted((r for r in eligible if (r['kind'], r['mode']) == group), key=rank)
        chosen.extend(r['key'] for r in group_rows[:2])
    remaining = sorted((r for r in eligible if r['key'] not in chosen), key=rank)
    chosen.extend(r['key'] for r in remaining[:30-len(chosen)])
    if len(chosen) != 30 or len(set(chosen)) != 30:
        raise ValueError('Expected 30 unique samples')
    revision = {'created_at': datetime.now().astimezone().isoformat(), 'reason': '用户明确要求尽量由AI全部评分，只少量随机人工审核。',
                'selection_timing': 'User revision during the first scoring pass; selection uses keys/group only, never grade/confidence/outcome.',
                'sample_size': 30, 'method': 'SHA256固定种子分层随机：每个kind/mode先取2份，余下按同一随机序补足30；排除已有人工标签。',
                'seed': 'user-small-random-audit-2026-10-17-v1', 'sample_keys': sorted(chosen),
                'dataset_sha256': digest(records), 'existing_human_labels': prior,
                'policy': 'AI completes all 518 suggestions; prior human scores take precedence at signoff; sample 30 required, other uncertainty visible for optional inspection. AI agreement is not independent human accuracy.'}
    path.write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding='utf-8')
    return revision


def finalize():
    revision = freeze_sampling()
    labels = summarize(OUT)
    assessments = read(OUT / 'codex-audit.json') if (OUT / 'codex-audit.json').exists() else {}
    for row in labels:
        row['ai_uncertainty_flags'] = row['review_reasons']
        row['review_reasons'] = ['30份分层随机人工抽查'] if row['key'] in revision['sample_keys'] else []
        if assessments.get(row['key'], {}).get('requires_user_decision'):
            row['review_reasons'].append('AI无法裁定，需用户决定')
        row['human_review_required'] = bool(row['review_reasons'])
        row['existing_human'] = revision['existing_human_labels'].get(row['key'])
    (OUT / 'ai-scores.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['key', 'kind', 'mode', 'question', 'ai_grade', 'judge_grade', 'refusal_correct', 'confidence', 'reason', 'human_review_required', 'existing_human_grade', 'human_grade', 'human_reviewer']
    with (OUT / 'ai-scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({**r, 'existing_human_grade': (r['existing_human'] or {}).get('grade', '')} for r in labels)
    summary = read(OUT / 'summary.json')
    summary.update(needs_human=sum(r['human_review_required'] for r in labels), active_random_sample=30,
                   existing_human_preserved=len(revision['existing_human_labels']), codex_audited=len(assessments),
                   sampling_policy='sampling-revision.json', human_accuracy=None, human_complete=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 1.4.1 AI辅助评分与30份人工抽查', '',
             f"AI初评 {summary['ai_scored']}/518；已有人工评分 {summary['existing_human_preserved']} 份原样保留；本次需随机审核30份。", '',
             '评分来自DeepSeek初评与Codex定向检查；尚未由用户完成本次抽查和最终签认。不得称为全量逐题人工测评。', '',
             '抽查不按分数或是否成功挑选。按实验类别/模式分层，种子与记录清单已固定；用户缩小抽查范围发生在评分进行中，已如实登记。', '',
             '若抽查发现实质错误，先检查同类评分并修正；必要时扩展该类抽查，不为凑一致率改人工判断。30份只能作为抽查验证，不能证明其余所有分数正确。', '',
             '原有123份人工分数优先保留于最终综合表；AI不同判断另列，原标签不覆盖。其余未逐题复核的建议分数，只有用户签认后才标为接受的AI评分。', '',
             '推荐论文口径：“采用AI辅助评分与人工审核相结合的综合评测，对新增AI评分随机抽查30份，并保留既有人工标签。”明确报告各部分数量，综合分数不冒充独立人工准确率。', '',
             '## AI建议分布', '', '|组别|Y|P|N|未评|', '|---|---:|---:|---:|---:|']
    for group, counts in summary['by_group'].items():
        lines.append(f"|{group}|{counts.get('Y',0)}|{counts.get('P',0)}|{counts.get('N',0)}|{counts.get('pending',0)}|")
    lines += ['', '## 本次仅需随机审核的30份', '']
    for row in labels:
        if row['key'] in revision['sample_keys']:
            lines.append(f"- **{row['key']} · {row['ai_grade'] or '待评分'}**：{row['question']} — {row['reason']}")
    lines += ['', '其余AI疑问和初评理由均保留于ai-scores.json，辅助页面“AI疑问（可选）”可查看。', '']
    (OUT / 'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('ai_scored','needs_human','active_random_sample','existing_human_preserved','codex_audited')}, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    if '--freeze-sampling' in sys.argv:
        value = freeze_sampling()
        print(json.dumps({'sample': len(value['sample_keys']), 'prior_human': len(value['existing_human_labels']), 'sample_keys': value['sample_keys']}, ensure_ascii=False))
    else:
        finalize()
