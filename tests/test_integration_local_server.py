"""Integration tests against a live OpenAI-compatible server.

Run with: uv run pytest -m integration
Requires a server at localhost:8080 serving mlx-community/Qwen3.5-4B-MLX-4bit.
"""

import pytest

import text_albumentations as ta

pytestmark = pytest.mark.integration

BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.fixture(scope="module")
def model():
    import httpx

    try:
        httpx.get(f"{BASE_URL}/models", timeout=2)
    except Exception:
        pytest.skip(f"no local model server at {BASE_URL}")
    return ta.OpenAIModel(MODEL_NAME, base_url=BASE_URL, api_key="local")


def test_augment_explicit_tasks_live(model, passage):
    rows = ta.augment(passage, tasks=["title"], model=model)
    assert len(rows) == 2
    assert all(row.output for row in rows)


def test_extractive_qa_quotes_verify_live(model, passage):
    from text_albumentations.tasks.extractive_qa import quote_in_passage

    rows = ta.augment(passage, tasks=["extractive_qa"], model=model)
    assert rows, "expected at least one verified quote"
    assert all(quote_in_passage(row.output, passage) for row in rows)
