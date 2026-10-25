import json
from pathlib import Path

from scripts.auto_judge import cohen_kappa, parse_grade
from scripts.report_agent_human import report
from service.redis_layer import RedisLayer, replay_cached
from service.schemas import Filters, QueryRequest, QueryResult
from service.settings import Settings


def test_advisory_grade_parser_and_kappa():
    assert parse_grade('{"grade":"P","reason":"遗漏比例","evidence_ids":["g_1"]}')['grade'] == 'P'
    assert cohen_kappa([('Y', 'Y'), ('P', 'P'), ('N', 'N')]) == 1
    assert cohen_kappa([]) is None


def test_cache_key_scope_and_replayed_usage():
    layer = RedisLayer(Settings(_env_file=None, cache_enabled=False))
    first = QueryRequest(question='剧情感人', strategy='vector')
    second = QueryRequest(question='剧情感人', strategy='vector', filters=Filters(movie='肖申克'))
    assert layer.cache_key(first, 'dataset1') != layer.cache_key(second, 'dataset1')
    assert layer.cache_key(first, 'dataset1') != layer.cache_key(first, 'dataset2')
    old = QueryResult(request_id='original', strategy='vector', status='ok', answer='有 [c_1]',
                      model_usage={'calls': 1, 'total_tokens': 123})
    import time
    hit = replay_cached(old.model_dump_json(), time.monotonic())
    assert hit.answer == old.answer and hit.request_id != old.request_id
    assert hit.model_usage['total_tokens'] == 0 and hit.cache['origin_model_usage']['total_tokens'] == 123


def test_agent_task_pass3_requires_all_human_marks(tmp_path):
    first = report(tmp_path)
    assert first['human_scored'] == 0 and first['task_success_pass_power_3'] is None
    file = tmp_path / 'human-task-scores.csv'
    lines = file.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines) == 61
    import csv
    with file.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['task_success'] = 'Y' if not (row['question_id'] == 'A01' and row['repetition'] == '3') else 'N'
    with file.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = report(tmp_path)
    assert summary['human_complete'] and summary['task_success_pass_power_3'] == 19 / 20
