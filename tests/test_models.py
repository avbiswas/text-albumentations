"""Offline tests for model primitives (no network calls)."""

import pytest
from pydantic import BaseModel

import text_albumentations as ta
from text_albumentations.models import resolve_response_format_mode
from text_albumentations.runtime import validate_json_output


class DummySchema(BaseModel):
    value: str


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
    model = ta.OpenAIModel("openai/gpt-test", base_url="http://localhost:8080/v1", api_key="k")
    assert model.model_name == "openai/gpt-test"
    assert model.async_mode is False
    assert model.max_tokens_parameter == "max_completion_tokens"
    assert model.response_format_mode == "json_schema"
    assert model.response_format is None
    assert model.model.type_adapter.format_output_type(DummySchema)[
        "response_format"
    ]["type"] == "json_schema"
    messages = model._prepare_messages(
        [{"role": "system", "content": "Base prompt."}],
        DummySchema,
    )
    assert messages[0]["content"].startswith("Base prompt.")
    assert "The desired schema you must answer in" in messages[0]["content"]
    assert "value" in messages[0]["content"]


def test_openai_model_unknown_auto_uses_json_object():
    model = ta.OpenAIModel("m", base_url="http://localhost:8080/v1", api_key="k")
    assert model.response_format_mode == "json_object"
    assert model.response_format == {"type": "json_object"}


def test_openai_model_custom_response_format():
    model = ta.OpenAIModel(
        "m",
        base_url="http://localhost:8080/v1",
        api_key="k",
        response_format="json_schema",
    )
    assert model.response_format_mode == "json_schema"
    assert model.response_format is None
    assert model.model.type_adapter.format_output_type(DummySchema)[
        "response_format"
    ]["type"] == "json_schema"


def test_openai_model_json_object_injects_schema_prompt():
    model = ta.OpenAIModel(
        "m",
        base_url="http://localhost:8080/v1",
        api_key="k",
        response_format="json_object",
    )
    assert model.response_format_mode == "json_object"
    assert model.response_format == {"type": "json_object"}
    messages = model._prepare_messages(
        [{"role": "user", "content": "Return a value."}],
        DummySchema,
    )
    assert messages[0]["role"] == "system"
    assert "The desired schema you must answer in" in messages[0]["content"]
    assert "value" in messages[0]["content"]


def test_schema_prompt_appends_to_last_system_message():
    model = ta.OpenAIModel("m", base_url="http://localhost:8080/v1", api_key="k")
    messages = model._prepare_messages(
        [
            {"role": "system", "content": "First system."},
            {"role": "user", "content": "Question"},
            {"role": "system", "content": "Last system."},
        ],
        DummySchema,
    )
    assert "The desired schema you must answer in" not in messages[0]["content"]
    assert messages[2]["content"].startswith("Last system.")
    assert "The desired schema you must answer in" in messages[2]["content"]


def test_openai_model_response_format_auto_catalog():
    assert resolve_response_format_mode("openai/gpt-5.4-nano") == "json_schema"
    assert resolve_response_format_mode("google/gemini-3.5-flash") == "json_schema"
    assert resolve_response_format_mode("anthropic/claude-opus-4.8") == "json_object"
    assert resolve_response_format_mode("deepseek/deepseek-v4-flash") == "json_object"
    assert resolve_response_format_mode("minimax/minimax-m3") == "json_object"
    assert resolve_response_format_mode("z-ai/glm-5") == "json_object"
    assert resolve_response_format_mode("unknown/model") == "json_object"


def test_openai_model_invalid_response_format_raises():
    with pytest.raises(ValueError, match="response_format must be one of"):
        ta.OpenAIModel(
            "m",
            base_url="http://localhost:8080/v1",
            api_key="k",
            response_format="bad",  # type: ignore[arg-type]
        )


def test_validate_json_output_accepts_fenced_json():
    output = '```json\n{"value": "ok"}\n```'
    parsed = validate_json_output(output, DummySchema)
    assert parsed.value == "ok"


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
