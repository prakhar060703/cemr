from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cemr.limits import LimitError, QuotaGuard, count_payload
from cemr.llm_client import LLMClient, LLMUnavailable, LLMQuotaExhausted


def client(tmp_path, **kwargs):
    with patch('openai.OpenAI'):
        result = LLMClient('gemini-3.8-flash',
                         'https://generativelanguage.googleapis.com/v1beta/openai/',
                         'fake-test-key', max_tokens=kwargs.pop('max_tokens', 8192),
                         quota_state_path=str(tmp_path / 'quota.sqlite3'), **kwargs)
        result.client.api_key = 'fake-test-key'
        return result


def test_payload_includes_system_and_history():
    payload = count_payload('gemini-3.8-flash', [
        {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'answer'}])['generateContentRequest']
    assert payload['systemInstruction']['parts'] == [{'text': 'rules'}]
    assert [m['role'] for m in payload['contents']] == ['user', 'model']


@pytest.mark.parametrize('cap', [0, -1, 65537, None])
def test_bad_output_cap(tmp_path, cap):
    with pytest.raises(ValueError, match='max_tokens'):
        client(tmp_path, max_tokens=cap)


def test_oversize_never_sent(tmp_path):
    c = client(tmp_path)
    with patch.object(c, '_count_input', return_value=1048576):
        with pytest.raises(LLMUnavailable, match='exceeds'):
            c.chat([{'role': 'user', 'content': 'huge prompt'}])
    c.client.chat.completions.create.assert_not_called()


def test_count_failure_never_sent(tmp_path):
    c = client(tmp_path)
    with patch('httpx.post', side_effect=RuntimeError('secret payload')):
        with pytest.raises(LLMUnavailable, match='token count failed') as exc:
            c.chat([{'role': 'user', 'content': 'hello'}])
    assert 'secret' not in str(exc.value)
    c.client.chat.completions.create.assert_not_called()


def test_shared_persistent_daily_budget(tmp_path):
    path = tmp_path / 'quota.sqlite3'
    QuotaGuard(path, 'flash', {'rpd': 1}).reserve(10)
    with pytest.raises(LimitError, match='24h'):
        QuotaGuard(path, 'flash', {'rpd': 1}).reserve(10)
    QuotaGuard(path, 'pro', {'rpd': 1}).reserve(10)


@pytest.mark.parametrize('limits,tokens', [({'rpm': 1}, 5), ({'tpm': 10}, 6)])
def test_minute_throttle(tmp_path, limits, tokens):
    now = [1000.0]
    def sleep(seconds):
        now[0] += seconds
    guard = QuotaGuard(tmp_path / 'quota.sqlite3', 'flash', limits)
    with patch('cemr.limits.time.time', side_effect=lambda: now[0]), patch('cemr.limits.time.sleep', side_effect=sleep):
        guard.reserve(tokens)
        guard.reserve(tokens)
    assert now[0] >= 1060


def test_single_prompt_over_tpm(tmp_path):
    with pytest.raises(LimitError, match='TPM'):
        QuotaGuard(tmp_path / 'quota.sqlite3', 'flash', {'tpm': 10}).reserve(11)


def test_truncation_and_usage_logging(tmp_path):
    c = client(tmp_path)
    usage = MagicMock()
    usage.model_dump.return_value = {'prompt_tokens': 12, 'completion_tokens': 8192}
    c.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='partial'), finish_reason='length')], usage=usage)
    with patch.object(c, '_count_input', return_value=12), patch.object(c, '_log') as log:
        with pytest.raises(LLMUnavailable, match='output hit'):
            c.chat([{'role': 'user', 'content': 'hello'}])
    assert log.call_args.args[-2] == usage.model_dump.return_value


def test_retry_reserves_again(tmp_path):
    c = client(tmp_path, busy_retries=1, busy_backoff=0)
    error = RuntimeError('busy')
    error.status_code = 429
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='ok'), finish_reason='stop')], usage=None)
    c.client.chat.completions.create.side_effect = [error, response]
    with patch.object(c, '_count_input', return_value=12), patch.object(c.quota, 'reserve') as reserve:
        assert c.chat([{'role': 'user', 'content': 'hello'}]) == 'ok'
    assert reserve.call_count == 2


@pytest.mark.parametrize('detail,status', [
    ('You exceeded your current quota', 429),
    ('RESOURCE_EXHAUSTED: RequestsPerDay limit reached', 429),
    ('insufficient_quota', 429),
    ('credits depleted', 402),
    ('Quota exceeded RequestsPerMinute limit: 0', 429),
])
def test_quota_never_retries(tmp_path, detail, status, caplog):
    c = client(tmp_path, busy_retries=3, busy_backoff=0)
    error = RuntimeError(detail)
    error.status_code = status
    c.client.chat.completions.create.side_effect = error
    with patch.object(c, '_count_input', return_value=12), patch('cemr.llm_client.time.sleep') as sleep:
        with pytest.raises(LLMQuotaExhausted):
            c.generate_json('system', 'question')
    assert c.client.chat.completions.create.call_count == 1
    sleep.assert_not_called()
    assert 'stopping without retry' in caplog.text


@pytest.mark.parametrize('status,detail', [
    (503, 'temporarily unavailable'), (429, 'rate limit'),
    (429, 'RESOURCE_EXHAUSTED: Quota exceeded RequestsPerMinute'),
    (None, 'connection dropped'),
])
def test_unavailable_has_three_retries_max(tmp_path, status, detail):
    c = client(tmp_path, busy_retries=99, busy_backoff=10)
    error = ConnectionError(detail)
    error.status_code = status
    c.client.chat.completions.create.side_effect = error
    with patch.object(c, '_count_input', return_value=12), patch('cemr.llm_client.time.sleep') as sleep:
        with pytest.raises(LLMUnavailable):
            c.generate_json('system', 'question')
    assert c.client.chat.completions.create.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [10, 20, 40]


def test_daily_local_limit_stops_before_api(tmp_path):
    c = client(tmp_path, model_quotas={'gemini-3.8-flash': {'rpd': 1}})
    c.quota.reserve(1)
    with patch.object(c, '_count_input', return_value=12):
        with pytest.raises(LLMQuotaExhausted, match='24h'):
            c.chat([{'role': 'user', 'content': 'hello'}])
    c.client.chat.completions.create.assert_not_called()


def test_count_tokens_quota_never_retries(tmp_path):
    import httpx
    c = client(tmp_path)
    request = httpx.Request('POST', 'https://example.test/countTokens')
    response = httpx.Response(429, request=request, json={'error': {'message': 'daily quota exhausted'}})
    with patch('httpx.post', return_value=response) as post:
        with pytest.raises(LLMQuotaExhausted):
            c.chat([{'role': 'user', 'content': 'hello'}])
    assert post.call_count == 1
    c.client.chat.completions.create.assert_not_called()


def test_count_tokens_transient_recovers(tmp_path):
    import httpx
    c = client(tmp_path, busy_backoff=0)
    request = httpx.Request('POST', 'https://example.test/countTokens')
    responses = [httpx.Response(503, request=request),
                 httpx.Response(200, request=request, json={'totalTokens': 12})]
    with patch('httpx.post', side_effect=responses) as post:
        assert c._count_input([{'role': 'user', 'content': 'hello'}]) == 12
    assert post.call_count == 2


def test_count_tokens_404_explains_model_restriction(tmp_path):
    import httpx
    c = client(tmp_path)
    request = httpx.Request('POST', 'https://example.test/countTokens')
    response = httpx.Response(404, request=request, json={'error': {
        'message': 'Model no longer available to new users. Use gemini-3.1-pro-preview.'}})
    with patch('httpx.post', return_value=response) as post:
        with pytest.raises(LLMUnavailable, match='no longer available to new users'):
            c.chat([{'role': 'user', 'content': 'hello'}])
    assert post.call_count == 1
    c.client.chat.completions.create.assert_not_called()


def test_preflight_rejects_listed_but_inaccessible_model(tmp_path):
    from cemr.pipeline import Pipeline
    c = client(tmp_path)
    c.client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id=c.model)])
    pipe = Pipeline.__new__(Pipeline)
    pipe.clients = {'verifier': c}
    with patch.object(c, '_count_input', side_effect=LLMUnavailable('model access denied')) as count:
        with pytest.raises(SystemExit, match='preflight failed.*verifier.*model access denied'):
            pipe.preflight()
    count.assert_called_once()
    c.client.chat.completions.create.assert_not_called()


def test_terminal_shows_call_and_output(tmp_path, caplog):
    import logging
    c = client(tmp_path)
    c.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='model answer'), finish_reason='stop')], usage=None)
    with caplog.at_level(logging.INFO), patch.object(c, '_count_input', return_value=12):
        c.chat([{'role': 'user', 'content': 'hello'}])
    assert 'chat.completions attempt 1/4' in caplog.text
    assert 'chat.completions OK' in caplog.text
    assert 'model answer' in caplog.text


def test_verifier_rejection_then_quota_does_not_regenerate_again(tmp_path, caplog):
    from cemr.schemas import VerifierScores
    from cemr.validation.verifier import generate_verified
    c = client(tmp_path, busy_backoff=0)
    error = RuntimeError('daily quota exhausted')
    error.status_code = 429
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{}'), finish_reason='stop')], usage=None)
    c.client.chat.completions.create.side_effect = [response, error]
    verifier = MagicMock(threshold=0.6)
    verifier.advisory.return_value = set()
    verifier.score.return_value = VerifierScores(scores={'grounded': 0.0}, rationale={'grounded': 'unsupported'})
    with patch.object(c, '_count_input', return_value=12):
        with pytest.raises(LLMQuotaExhausted):
            generate_verified(lambda feedback: c.generate_json('system', 'question'),
                              lambda item: [], verifier, 'test', lambda item: item, '', 3)
    assert c.client.chat.completions.create.call_count == 2
    assert verifier.score.call_count == 1
    assert 'VERIFIER REJECT' in caplog.text
