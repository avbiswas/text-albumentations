from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from text_albumentations.models import OpenAIModel
from text_albumentations.runtime import ModelRuntime


class QualityAssessment(BaseModel):
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


def _build_quality_messages(
    datapoint: Any,
    prompt: str,
) -> list[dict[str, str]]:
    if not prompt.strip():
        raise ValueError("quality_filter requires a non-empty prompt.")

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


def quality_filter(
    datapoint: Any,
    prompt: str,
    *,
    model: ModelRuntime | None = None,
) -> QualityAssessment:
    if model is None:
        model = OpenAIModel()

    return model.generate_structured(
        _build_quality_messages(datapoint, prompt),
        QualityAssessment,
        temperature=0.0,
        max_tokens=500,
    )


async def aquality_filter(
    datapoint: Any,
    prompt: str,
    *,
    model: ModelRuntime | None = None,
) -> QualityAssessment:
    if model is None:
        model = OpenAIModel(async_mode=True)

    return await model.agenerate_structured(
        _build_quality_messages(datapoint, prompt),
        QualityAssessment,
        temperature=0.0,
        max_tokens=500,
    )
