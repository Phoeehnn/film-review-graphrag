"""Create/aggregate human Agent task scores without changing raw responses."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'evaluation/agent/runs/2026-09-30_1.3.0'
FIELDS = ['question_id', 'repetition', 'task_success', 'reviewer', 'notes']


def report(run):
    questions = json.loads((ROOT / 'evaluation/agent/questions.json').read_text(encoding='utf-8'))['questions']
    expected = [(q['id'], n) for q in questions for n in range(1, 4)]
    score_path = run / 'human-task-scores.csv'
    if not score_path.exists():
        with score_path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({'question_id': q, 'repetition': n, 'task_success': '', 'reviewer': '', 'notes': ''}
                             for q, n in expected)
    with score_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected), 'Agent score CSV must contain 60 rows'
    actual = [(r['question_id'], int(r['repetition'])) for r in rows]
    assert actual == expected, 'Agent score row order or IDs changed'
    assert all(r['task_success'] in ('', 'Y', 'N') for r in rows), 'task_success must be Y/N/blank'
    scores = {(r['question_id'], int(r['repetition'])): r['task_success'] for r in rows}
    marked = sum(bool(x) for x in scores.values())
    complete = marked == len(expected)
    successful = sum(x == 'Y' for x in scores.values())
    pass3 = sum(all(scores[q['id'], n] == 'Y' for n in range(1, 4)) for q in questions)
    summary = {'question_count': len(questions), 'total_trials': len(expected), 'human_scored': marked,
               'human_complete': complete, 'task_success_rate': successful / len(expected) if complete else None,
               'task_success_pass_power_3': pass3 / len(questions) if complete else None,
               'pass3_count': pass3 if complete else None,
               'failure_count_by_question': dict(Counter(r['question_id'] for r in rows if r['task_success'] == 'N')),
               'definition': 'Y means the final answer completed the requested task with supported, correctly scoped facts; N otherwise. Three independent trials per question must all be Y for pass^3.'}
    (run / 'human-task-report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(report(args.run), ensure_ascii=False, indent=2))
