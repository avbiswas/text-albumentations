"""Offline tests for model primitives (no network calls)."""

import pytest

import text_albumentations as ta


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("TEXT_ALBUMENTATIONS_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_openai_model_requires_full_config():
    with pytest.raises(ValueError) as excinfo:
        ta.OpenAIModel()
    message = str(excinfo.value)
    assert "TEXT_ALBUMENTATIONS_MODEL" in message
    assert "OPENAI_BASE_URL" in message
    assert "OPENAI_API_KEY" in message


def test_openai_model_reports_only_missing_pieces():
    with pytest.raises(ValueError) as excinfo:
        ta.OpenAIModel("some-model", base_url="http://localhost:8080/v1")
    missing_section = str(excinfo.value).split("Missing: ")[1].split(". Example")[0]
    assert "api_key" in missing_section
    assert "model" not in missing_section
    assert "base_url" not in missing_section


def test_openai_model_explicit_args():
    model = ta.OpenAIModel("m", base_url="http://localhost:8080/v1", api_key="k")
    assert model.model_name == "m"
    assert model.async_mode is False
    assert model.max_tokens_parameter == "max_completion_tokens"


def test_openai_model_env_fallback(monkeypatch):
    monkeypatch.setenv("TEXT_ALBUMENTATIONS_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model = ta.OpenAIModel()
    assert model.model_name == "env-model"


def test_openai_model_async_mode():
    model = ta.OpenAIModel(
        "m", base_url="http://localhost:8080/v1", api_key="k", async_mode=True
    )
    assert model.async_mode is True


def test_model_primitives_are_runtimes():
    from text_albumentations.runtime import ModelRuntime

    model = ta.OpenAIModel("m", base_url="http://localhost:8080/v1", api_key="k")
    assert isinstance(model, ModelRuntime)
