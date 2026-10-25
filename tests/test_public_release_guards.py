"""Publication guards: environment-only credentials and empty review pages."""
import ast
import json
from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from service.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PAGES = ['1_评测复核.py', '2_检索质量复核.py', '5_Agent任务复核.py', '6_多轮与修复复核.py']


def test_config_has_no_literal_password_or_key_fallback(monkeypatch):
    tree = ast.parse((ROOT / 'config.py').read_text(encoding='utf-8'))
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    for name, attribute in [('NEO4J_PASSWORD', 'neo4j_password'), ('DEEPSEEK_API_KEY', 'deepseek_api_key')]:
        assert isinstance(assignments[name], ast.Attribute)
        assert assignments[name].attr == attribute
        assert isinstance(assignments[name].value, ast.Name) and assignments[name].value.id == 'settings'
    assert Settings.model_fields['neo4j_password'].default == ''
    assert Settings.model_fields['deepseek_api_key'].default == ''
    monkeypatch.setenv('NEO4J_PASSWORD', 'synthetic-password-for-unit-test')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'synthetic-key-for-unit-test')
    settings = Settings(_env_file=None)
    assert settings.neo4j_password == 'synthetic-password-for-unit-test'
    assert settings.deepseek_api_key == 'synthetic-key-for-unit-test'


def test_env_example_is_explicitly_nonfunctional():
    fields = dict(line.split('=', 1) for line in (ROOT / '.env.example').read_text(encoding='utf-8').splitlines() if '=' in line and not line.startswith('#'))
    assert fields['NEO4J_PASSWORD'] == 'your-neo4j-password'
    assert fields['DEEPSEEK_API_KEY'] == 'your-deepseek-api-key'


@pytest.mark.parametrize('page', REVIEW_PAGES)
@pytest.mark.parametrize('include_question_fixtures', [False, True])
def test_review_pages_without_private_runs_show_no_data(tmp_path, page, include_question_fixtures):
    app_root = tmp_path / 'empty-application'
    (app_root / 'pages').mkdir(parents=True)
    target = app_root / 'pages' / page
    shutil.copy2(ROOT / 'pages' / page, target)
    if include_question_fixtures:
        for location in ['evaluation/questions.json', 'evaluation/agent/questions.json', 'evaluation/retrieval/questions.json']:
            path = app_root / location
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'questions': []}), encoding='utf-8')
    app = AppTest.from_file(str(target), default_timeout=30).run()
    assert not app.exception, [item.message for item in app.exception]
    assert any('暂无可用' in item.value for item in app.info)
    assert not (app_root / 'evaluation/runs').exists()
