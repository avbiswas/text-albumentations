from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from text_albumentations.models import OpenAIModel
from text_albumentations.runtime import ModelRuntime

DEFAULT_POSTFILTER_PROMPT = """\
A quality training datapoint should be useful for supervised fine-tuning.
Keep the datapoint only if the instruction is clear, the input contains enough
context to support the output, and the output directly satisfies the instruction.
Reject contradictions, unsupported claims, truncation, boilerplate, malformed
rows, or examples that are not useful for training.
"""


class PostfilterAssessment(BaseModel):
    is_quality: bool = Field(
        ...,
        description="True if the datapoint satisfies the supplied quality criteria.",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of the quality decision.",
    )


def _serialize_datapoint(datapoint: Any) -> str:
    if isinstance(datapoint, str):
        return datapoint
    return json.dumps(datapoint, ensure_ascii=False, indent=2, sort_keys=True)


def _build_postfilter_messages(
    datapoint: Any,
    prompt: str,
) -> list[dict[str, str]]:
    if not prompt.strip():
        raise ValueError("postfilter requires a non-empty prompt.")

    return [
        {
            "role": "system",
            "content": (
                "You are a data quality judge. Decide whether the datapoint "
                "satisfies the user's quality criteria. Judge only the supplied "
                "datapoint against those criteria. Return a concise decision."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Quality criteria:\n{prompt.strip()}\n\n"
                f"Datapoint:\n{_serialize_datapoint(datapoint)}"
            ),
        },
    ]


def postfilter(
    datapoint: Any,
    prompt: str | None = None,
    *,
    model: ModelRuntime | None = None,
) -> PostfilterAssessment:
    if model is None:
        model = OpenAIModel()
    if prompt is None:
        prompt = DEFAULT_POSTFILTER_PROMPT

    return model.generate_structured(
        _build_postfilter_messages(datapoint, prompt),
        PostfilterAssessment,
        temperature=0.0,
        max_tokens=500,
    )


async def apostfilter(
    datapoint: Any,
    prompt: str | None = None,
    *,
    model: ModelRuntime | None = None,
) -> PostfilterAssessment:
    if model is None:
        model = OpenAIModel(async_mode=True)
    if prompt is None:
        prompt = DEFAULT_POSTFILTER_PROMPT

    return await model.agenerate_structured(
        _build_postfilter_messages(datapoint, prompt),
        PostfilterAssessment,
        temperature=0.0,
        max_tokens=500,
    )
