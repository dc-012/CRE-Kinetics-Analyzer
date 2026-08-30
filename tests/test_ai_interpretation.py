"""
Tests for app/ai_interpretation.py - the optional OpenAI explanation
layer.

No real network call is ever made here. The "API available" path is
exercised by injecting a small fake `openai` module into
`sys.modules` (this environment has no network access to install the
real `openai` package, and even where it is installed, tests should
never depend on a live network call or a real key). These tests only
check that ai_interpretation.py wires numbers -> prompt -> client
correctly and degrades gracefully - they never assert anything about
core.calculations, which is tested elsewhere.
"""

import os
import sys

import pytest

from app import ai_interpretation
from core.calculations import analyze_kinetics


@pytest.fixture
def sample_result():
    t = [0, 20, 40, 60, 120, 180, 300]
    c = [10, 8, 6, 5, 3, 2, 1]
    return analyze_kinetics(t, c)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, response_text=None, raise_exc=None):
        self._response_text = response_text
        self._raise_exc = raise_exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeCompletion(self._response_text)


class _FakeChat:
    def __init__(self, chat_completions):
        self.completions = chat_completions


class _FakeOpenAIClient:
    def __init__(self, chat_completions):
        self.chat = _FakeChat(chat_completions)

    def __call__(self, *a, **kw):
        # Supports `OpenAI()` being called with no args, as real client is.
        return self


def _install_fake_openai_module(chat_completions):
    """Inject a fake `openai` module exposing `OpenAI` so
    `from openai import OpenAI` in ai_interpretation.py picks it up,
    without needing the real package (no network in this environment)."""
    import types
    fake_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat(chat_completions)

    fake_module.OpenAI = OpenAI
    sys.modules["openai"] = fake_module


def test_no_api_key_returns_none_without_importing_openai(sample_result):
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        assert ai_interpretation.api_key_available() is False
        assert ai_interpretation.generate_interpretation(sample_result) is None
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_api_key_available_true_when_env_var_set():
    old = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    try:
        assert ai_interpretation.api_key_available() is True
    finally:
        if old is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old


def test_generate_interpretation_success_path(sample_result):
    old_key = os.environ.get("OPENAI_API_KEY")
    old_module = sys.modules.get("openai")
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    fake_completions = _FakeChatCompletions(response_text="This looks like a good fit.")
    _install_fake_openai_module(fake_completions)
    try:
        text = ai_interpretation.generate_interpretation(sample_result, time_unit="min", concentration_unit="mol/L")
        assert text == "This looks like a good fit."
        # The prompt sent to the fake client should reference the
        # already-calculated numbers, not ask it to compute anything.
        sent_prompt = fake_completions.last_kwargs["messages"][-1]["content"]
        assert f"{sample_result.reaction_order:.4g}" in sent_prompt
        assert f"{sample_result.rate_constant:.4g}" in sent_prompt
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key
        if old_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = old_module


def test_generate_interpretation_returns_none_on_api_error(sample_result):
    old_key = os.environ.get("OPENAI_API_KEY")
    old_module = sys.modules.get("openai")
    os.environ["OPENAI_API_KEY"] = "test-key-123"

    import types
    fake_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat(_FakeChatCompletions(raise_exc=RuntimeError("network down")))

    fake_module.OpenAI = OpenAI
    sys.modules["openai"] = fake_module
    try:
        result = ai_interpretation.generate_interpretation(sample_result)
        assert result is None
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key
        if old_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = old_module


def test_build_prompt_contains_key_quantities_only(sample_result):
    prompt = ai_interpretation.build_prompt(sample_result, time_unit="min", concentration_unit="mol/L")
    assert f"{sample_result.reaction_order:.4g}" in prompt
    assert f"{sample_result.rate_constant:.4g}" in prompt
    assert f"{sample_result.r_squared:.5f}" in prompt
    assert "do NOT recalculate" in prompt
