from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from typing import List

import text_albumentations as ta


class SampleOutput(BaseModel):
    points: List[str]


@pytest.fixture
def vertex_model():
    model = ta.VertexAIModel.__new__(ta.VertexAIModel)
    model.model_name = "gemini-2.5-flash"
    model.async_mode = False
    model.total_concurrent_calls = 10
    model._semaphore = None
    model._client = MagicMock()
    return model


def test_vertex_model_is_model_runtime():
    assert issubclass(ta.VertexAIModel, ta.ModelRuntime)
    assert "VertexAIModel" in ta.__all__


def test_build_contents_maps_roles(vertex_model):
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    system, contents = vertex_model._build_contents(messages)
    assert system == "Be helpful."
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"


def test_generate_structured_sync(vertex_model):
    expected = SampleOutput(points=["a", "b"])
    mock_response = MagicMock()
    mock_response.text = expected.model_dump_json()
    vertex_model._client.models.generate_content.return_value = mock_response

    result = vertex_model.generate_structured(
        [{"role": "user", "content": "test"}], SampleOutput
    )
    assert result == expected


def test_generate_structured_raises_in_async_mode(vertex_model):
    vertex_model.async_mode = True
    with pytest.raises(RuntimeError, match="async mode"):
        vertex_model.generate_structured([{"role": "user", "content": "x"}], SampleOutput)


def test_generate_variation_with_string(vertex_model):
    expected = SampleOutput(points=["varied"])
    mock_response = MagicMock()
    mock_response.text = expected.model_dump_json()
    vertex_model._client.models.generate_content.return_value = mock_response

    result = vertex_model.generate_variation("raw data", SampleOutput)
    assert result == expected


def test_empty_response_raises(vertex_model):
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = []
    vertex_model._client.models.generate_content.return_value = mock_response

    with pytest.raises(RuntimeError, match="empty response"):
        vertex_model.generate_structured([{"role": "user", "content": "x"}], SampleOutput)


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.delenv("CLOUDSDK_CORE_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    with patch("google.genai.Client") as mock_client:
        ta.VertexAIModel("gemini-2.5-flash")
        kw = mock_client.call_args.kwargs
        assert kw["project"] == "env-project"
        assert kw["location"] == "us-central1"
