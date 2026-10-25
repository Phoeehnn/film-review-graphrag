"""Evaluate the deterministic router on a clearly labelled development set."""
import json
import argparse
from collections import Counter
from pathlib import Path

from service.routing import route


ROOT=Path(__file__).resolve().parents[1]


def main(fixture=None,target=None):
    fixture=fixture or ROOT/'evaluation/router/development_cases.json'
    cases=json.loads(fixture.read_text(encoding='utf-8'))['cases']
    rows=[{'question':case['question'],'expected':case['expected'],'actual':route(case['question'],'auto')[0]} for case in cases]
    counts=Counter(row['expected'] for row in rows)
    result={'total':len(rows),'expected_counts':dict(counts),'correct':sum(row['actual']==row['expected'] for row in rows),
            'development_accuracy':sum(row['actual']==row['expected'] for row in rows)/len(rows),
            'misroutes':[row for row in rows if row['actual']!=row['expected']],
            'scope_note':'人工编写的开发/回归集；这不是独立留出集，也不是问答正确率。'}
    target=target or ROOT/'evaluation/router/development_report.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    main(args.cases,args.output)
