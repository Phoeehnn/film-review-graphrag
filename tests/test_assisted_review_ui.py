import shutil
from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest

from scripts.assisted_review import ROOT, OUT, read
from scripts.finalize_assisted_review import freeze_sampling

pytestmark = pytest.mark.skipif(not (OUT / 'ai-scores.json').exists(), reason='Local review data are intentionally not committed')


def test_sample_is_small_fixed_grouped_and_excludes_existing_human_labels():
    revision = freeze_sampling()
    assert len(revision['sample_keys']) == len(set(revision['sample_keys'])) == 30
    assert not (set(revision['sample_keys']) & set(revision['existing_human_labels']))
    records = read(OUT / 'dataset.json')
    sample = [r for r in records if r['key'] in revision['sample_keys']]
    assert {(r['kind'], r['mode']) for r in sample} == {(r['kind'], r['mode']) for r in records}
    assert len(revision['existing_human_labels']) == 123
    assert revision == freeze_sampling()


def test_ui_requires_actual_click_name_and_all_samples_before_signoff(tmp_path, monkeypatch):
    for name in ('protocol.json', 'dataset.json', 'ai-scores.json', 'ai-scores.csv'):
        shutil.copy2(OUT / name, tmp_path / name)
    monkeypatch.setenv('ASSISTED_REVIEW_DIR', str(tmp_path))
    app = AppTest.from_file(str(ROOT / 'scripts/assisted_review_ui.py')).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '518/518'
    assert app.metric[1].value == '30'
    assert app.metric[2].value == '0/30'
    assert next(b for b in app.button if b.label == '生成最终确认记录').disabled
    next(b for b in app.button if b.label == '认可本项AI分数与拒答判断').click().run(timeout=30)
    assert not (tmp_path / 'human-final-review.json').exists()
    app.text_input[0].set_value('isolated-ui-test').run(timeout=30)
    next(b for b in app.button if b.label == '认可本项AI分数与拒答判断').click().run(timeout=30)
    assert not app.exception
    state = read(tmp_path / 'human-final-review.json')
    assert len(state['decisions']) == 1 and 'signoff' not in state
    assert app.metric[2].value == '1/30'
    assert next(b for b in app.button if b.label == '生成最终确认记录').disabled
