import json

import pytest

from scripts import assisted_review as review


def assessment(**patch):
    value = {'grade': 'Y', 'refusal_correct': 'NA', 'confidence': 'high', 'needs_human': False,
             'reason': '数字和范围与证据相符', 'uncertainties': [], 'evidence_ids': ['g_1']}
    return json.dumps({**value, **patch}, ensure_ascii=False)


def test_unknown_judge_citation_requires_review_and_agent_partial_rejected():
    record = {'kind': 'agent', 'result': {'evidence': [{'id': 'g_1'}]}}
    parsed = review.parse(assessment(evidence_ids=['invented']), record)
    assert parsed['needs_human'] and parsed['uncertainties']
    with pytest.raises(ValueError):
        review.parse(assessment(grade='P'), record)
    with pytest.raises(ValueError):
        review.parse(assessment(needs_human='false'), record)


def test_complete_evidence_kept_and_posted_identity_preserved():
    text = '这是真实评论' * 1000
    rows = [{'value': i} for i in range(200)]
    result = {'answer': text, 'evidence': [{'id': 'c_1', 'data': {'text': text, 'rows': rows,
        'nodes': [{'_element_id': 'u', 'user_id': 'u_42', 'username': 'name'},
                  {'_element_id': 'c', 'comment_id': 'c_1'}],
        'relationships': [{'_type': 'POSTED', 'source': 'u', 'target': 'c'}]}}]}
    projected = review.project_result(result)
    data = projected['evidence'][0]['data']
    assert projected['answer'] == text and data['text'] == text and data['rows'] == rows
    assert data['posted_relations'] == [{'user_id': 'u_42', 'comment_id': 'c_1'}]
    assert 'nodes' not in data and 'relationships' not in data


def test_review_cannot_become_human_accuracy_or_overwrite_human_file(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    protected = tmp_path / 'human-scores.csv'
    protected.write_text('id,grade\n1,N\n', encoding='utf-8')
    out = tmp_path / 'ai'
    out.mkdir()
    record = {'key': 'qa/01_graph', 'kind': 'qa', 'mode': 'graph', 'question_id': '1',
              'question': '问题', 'result': {'evidence': []}}
    def write(name, value):
        (out / name).write_text(json.dumps(value), encoding='utf-8')
    write('dataset.json', [record])
    write('protocol.json', {'dataset_sha256': review.digest([record]), 'blind_sample_keys': []})
    write('protected-human-files.json', {'human-scores.csv': review.file_hash(protected)})
    (out / 'responses').mkdir()
    write('responses/qa__01_graph.json', {'input_sha256': review.digest(record),
                                         'assessment': json.loads(assessment()), 'model_usage': {'calls': 1}})
    before = protected.read_bytes()
    review.summarize(out)
    summary = review.read(out / 'summary.json')
    assert summary['human_accuracy'] is None and summary['human_complete'] is False
    assert protected.read_bytes() == before
    protected.write_text('external edit', encoding='utf-8')
    with pytest.raises(ValueError, match='human files changed'):
        review.summarize(out)


def test_real_frozen_scope_and_history_are_complete():
    if not (review.OUT / 'dataset.json').exists():
        pytest.skip('Local frozen review inputs are intentionally not committed')
    records = review.read(review.OUT / 'dataset.json')
    assert len(records) == len({r['key'] for r in records}) == 518
    assert len([r for r in records if r['kind'] == 'qa']) == 168
    assert len([r for r in records if r['kind'] == 'repair']) == 140
    assert {r['mode'] for r in records if r['kind'] == 'repair'} == {'0', '2'}
    eighth = next(r for r in records if r['key'] == 'long-conversation/L01_compact_8')
    assert [r['turn'] for r in eighth['history']] == list(range(1, 8))
    assert all(r['history'] for r in records if r['kind'] in ('conversation', 'long-conversation'))
