"""Provider billing failures are infrastructure incidents, not capability grades."""
import json


class ProviderUnavailable(RuntimeError):
    pass


def is_billing_failure(result):
    error = json.dumps(result.get('error', {}), ensure_ascii=False).lower()
    return 'insufficient balance' in error or 'error code: 402' in error


def require_available(result, run, identifier):
    if is_billing_failure(result):
        directory = run / 'infrastructure-failures'
        directory.mkdir(exist_ok=True)
        (directory / (identifier + '-provider.json')).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        raise ProviderUnavailable('Provider billing unavailable; incident preserved, evaluation stopped')
