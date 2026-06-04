"""Offline tests for the generic datapoint quality filter."""

import pytest

import text_albumentations as ta
from text_albumentations.quality import QualityAssessment

from conftest import FakeModel


def test_quality_filter_judges_string_datapoint():
    model = FakeModel(
        {
            "QualityAssessment": QualityAssessment(
                is_quality=True,
                reason="The answer is grounded and complete.",
            ),
        }
    )

    assessment = ta.quality_filter(
        "Input: What is 2+2?\nOutput: 4",
        "A quality datapoint has a correct output for the input.",
        model=model,
    )

    assert assessment.is_quality is True
    assert assessment.reason == "The answer is grounded and complete."
    messages, schema_name = model.calls[0]
    assert schema_name == "QualityAssessment"
    assert "Input: What is 2+2?" in messages[1]["content"]
    assert "A quality datapoint has a correct output" in messages[1]["content"]


def test_quality_filter_serializes_json_datapoint():
    model = FakeModel(
        {
            "QualityAssessment": QualityAssessment(
                is_quality=False,
                reason="The output contradicts the input.",
            ),
        }
    )

    assessment = ta.quality_filter(
        {"instruction": "Answer yes.", "input": "Say no.", "output": "yes"},
        "Reject contradictions between input and output.",
        model=model,
    )

    assert assessment.is_quality is False
    messages, _ = model.calls[0]
    assert '"instruction": "Answer yes."' in messages[1]["content"]
    assert '"output": "yes"' in messages[1]["content"]


def test_quality_filter_requires_prompt():
    with pytest.raises(ValueError, match="non-empty prompt"):
        ta.quality_filter("datapoint", " ", model=FakeModel({}))


@pytest.mark.anyio
async def test_async_quality_filter():
    model = FakeModel(
        {
            "QualityAssessment": QualityAssessment(
                is_quality=True,
                reason="Good.",
            ),
        }
    )

    assessment = await ta.aquality_filter(
        {"input": "x", "output": "x"},
        "Accept exact copies.",
        model=model,
    )

    assert assessment.is_quality is True
