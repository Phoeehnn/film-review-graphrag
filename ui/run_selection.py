"""Hide incomplete and explicitly superseded runs without hiding valid baselines."""
import json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def excluded(run):
    marker=run/'review-status.json'
    if marker.exists() and not load(marker).get('reviewable',True):return True
    progress=run/'progress.json'
    return progress.exists() and load(progress).get('state')!='complete'


def retrieval_complete(run):
    try:
        return not excluded(run) and (run/'relevance.csv').is_file() and bool(load(run/'run-config.json').get('completed_at')) and len(load(run/'results.json'))==40
    except (OSError,ValueError,TypeError):return False


def agent_complete(run):
    try:
        summary=load(run/'summary.json')
        return not excluded(run) and (run/'human-task-scores.csv').is_file() and summary.get('complete') is True and summary.get('completed_calls')==60 and len(list((run/'responses').glob('*.json')))==60
    except (OSError,ValueError,TypeError):return False
