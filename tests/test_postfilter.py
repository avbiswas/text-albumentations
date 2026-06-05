"""Offline tests for the generic datapoint postfilter."""

import pytest

import text_albumentations as ta
from text_albumentations.postfilter import PostfilterAssessment

from conftest import FakeModel


def test_postfilter_judges_string_datapoint():
    model = FakeModel(
        {
            "PostfilterAssessment": PostfilterAssessment(
                is_quality=True,
                reason="The answer is grounded and complete.",
            ),
        }
    )

    assessment = ta.postfilter(
        "Input: What is 2+2?\nOutput: 4",
        "A quality datapoint has a correct output for the input.",
        model=model,
    )

    assert assessment.is_quality is True
    assert assessment.reason == "The answer is grounded and complete."
    messages, schema_name = model.calls[0]
    assert schema_name == "PostfilterAssessment"
    assert "Input: What is 2+2?" in messages[1]["content"]
    assert "A quality datapoint has a correct output" in messages[1]["content"]


def test_postfilter_serializes_json_datapoint():
    model = FakeModel(
        {
            "PostfilterAssessment": PostfilterAssessment(
                is_quality=False,
                reason="The output contradicts the input.",
            ),
        }
    )

    assessment = ta.postfilter(
        {"instruction": "Answer yes.", "input": "Say no.", "output": "yes"},
        "Reject contradictions between input and output.",
        model=model,
    )

    assert assessment.is_quality is False
    messages, _ = model.calls[0]
    assert '"instruction": "Answer yes."' in messages[1]["content"]
    assert '"output": "yes"' in messages[1]["content"]


def test_postfilter_uses_default_prompt():
    model = FakeModel(
        {
            "PostfilterAssessment": PostfilterAssessment(
                is_quality=True,
                reason="Good.",
            ),
        }
    )

    assessment = ta.postfilter({"input": "x", "output": "x"}, model=model)

    assert assessment.is_quality is True
    messages, _ = model.calls[0]
    assert "quality training datapoint" in messages[1]["content"]


def test_postfilter_requires_non_empty_prompt():
    with pytest.raises(ValueError, match="non-empty prompt"):
        ta.postfilter("datapoint", " ", model=FakeModel({}))


@pytest.mark.anyio
async def test_async_postfilter():
    model = FakeModel(
        {
            "PostfilterAssessment": PostfilterAssessment(
                is_quality=True,
                reason="Good.",
            ),
        }
    )

    assessment = await ta.apostfilter(
        {"input": "x", "output": "x"},
        "Accept exact copies.",
        model=model,
    )

    assert assessment.is_quality is True
