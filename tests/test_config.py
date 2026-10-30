"""
Tests for credential resolution.

The suite must never depend on a real .env, so every test controls the environment
explicitly via monkeypatch.
"""

import pytest

import transformer2.config as config
from transformer2.config import MissingCredentialError, get_secret, huggingface_token, reddit_credentials


class TestGetSecret:
    def test_reads_from_the_environment(self, monkeypatch):
        monkeypatch.setenv('SOME_TOKEN', 'abc123')

        assert get_secret('SOME_TOKEN') == 'abc123'

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv('SOME_TOKEN', raising=False)

        assert get_secret('SOME_TOKEN', default='fallback') == 'fallback'

    def test_strips_surrounding_whitespace(self, monkeypatch):
        """A trailing newline from a .env line would otherwise reach the API call."""
        monkeypatch.setenv('SOME_TOKEN', '  abc123\n')

        assert get_secret('SOME_TOKEN') == 'abc123'

    def test_empty_value_is_treated_as_unset(self, monkeypatch):
        """.env.example ships blank values; those must not count as configured."""
        monkeypatch.setenv('SOME_TOKEN', '')

        assert get_secret('SOME_TOKEN', default='fallback') == 'fallback'

    def test_whitespace_only_value_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv('SOME_TOKEN', '   ')

        assert get_secret('SOME_TOKEN') is None

    def test_required_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv('SOME_TOKEN', raising=False)

        with pytest.raises(MissingCredentialError, match='SOME_TOKEN'):
            get_secret('SOME_TOKEN', required=True)

    def test_error_explains_how_to_fix_it(self, monkeypatch):
        monkeypatch.delenv('SOME_TOKEN', raising=False)

        with pytest.raises(MissingCredentialError, match=r'\.env'):
            get_secret('SOME_TOKEN', required=True)


class TestRedditCredentials:
    def test_resolves_from_the_environment(self, monkeypatch):
        monkeypatch.setenv('REDDIT_CLIENT_ID', 'env-id')
        monkeypatch.setenv('REDDIT_CLIENT_SECRET', 'env-secret')

        assert reddit_credentials()[:2] == ('env-id', 'env-secret')

    def test_explicit_arguments_win(self, monkeypatch):
        monkeypatch.setenv('REDDIT_CLIENT_ID', 'env-id')
        monkeypatch.setenv('REDDIT_CLIENT_SECRET', 'env-secret')

        assert reddit_credentials(client_id='arg-id')[0] == 'arg-id'

    def test_returns_none_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.delenv('REDDIT_CLIENT_ID', raising=False)
        monkeypatch.delenv('REDDIT_CLIENT_SECRET', raising=False)

        assert reddit_credentials()[:2] == (None, None)

    def test_user_agent_has_a_default(self, monkeypatch):
        monkeypatch.delenv('REDDIT_USER_AGENT', raising=False)

        assert reddit_credentials()[2] == 'python:llm-training-data:v1.0'

    def test_user_agent_is_overridable_by_environment(self, monkeypatch):
        monkeypatch.setenv('REDDIT_USER_AGENT', 'custom-agent')

        assert reddit_credentials()[2] == 'custom-agent'

    def test_required_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv('REDDIT_CLIENT_ID', raising=False)
        monkeypatch.delenv('REDDIT_CLIENT_SECRET', raising=False)

        with pytest.raises(MissingCredentialError):
            reddit_credentials(required=True)


class TestHuggingFaceToken:
    def test_reads_huggingface_token(self, monkeypatch):
        monkeypatch.setenv('HUGGINGFACE_TOKEN', 'hf-a')
        monkeypatch.delenv('HF_TOKEN', raising=False)

        assert huggingface_token() == 'hf-a'

    def test_falls_back_to_hf_token(self, monkeypatch):
        """huggingface_hub tooling uses the HF_TOKEN name."""
        monkeypatch.delenv('HUGGINGFACE_TOKEN', raising=False)
        monkeypatch.setenv('HF_TOKEN', 'hf-b')

        assert huggingface_token() == 'hf-b'

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv('HUGGINGFACE_TOKEN', raising=False)
        monkeypatch.delenv('HF_TOKEN', raising=False)

        assert huggingface_token() is None

    def test_required_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv('HUGGINGFACE_TOKEN', raising=False)
        monkeypatch.delenv('HF_TOKEN', raising=False)

        with pytest.raises(MissingCredentialError, match='Hugging Face'):
            huggingface_token(required=True)


class TestNoHardcodedSecrets:
    def test_module_contains_no_literal_credentials(self):
        """config.py resolves secrets; it must never carry one."""
        import inspect

        source = inspect.getsource(config)

        assert 'REDDIT_CLIENT_ID' in source, "sanity: the variable name should appear"
        for marker in ('hf_', 'sk-', 'AKIA'):
            assert marker not in source.replace('hf_token', '').replace('_hf_', '')
