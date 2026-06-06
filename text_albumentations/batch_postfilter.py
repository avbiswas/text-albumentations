"""Batch postfilter — score multiple rows in a single LLM call.

Reduces postfilter call volume by 10x (batch_size rows → 1 call instead of N).
Drop-in replacement for the per-row postfilter loop in easy.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from text_albumentations.models import OpenAIModel
from text_albumentations.postfilter import (
    DEFAULT_POSTFILTER_PROMPT,
    PostfilterAssessment,
    _serialize_datapoint,
)
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset


class BatchPostfilterAssessment(BaseModel):
    assessments: list[PostfilterAssessment] = Field(
        ...,
        description="Quality assessments for each datapoint, in the same order as input.",
    )


def _build_batch_postfilter_messages(
    datapoints: list[Any],
    prompt: str,
) -> list[dict[str, str]]:
    if not prompt.strip():
        raise ValueError("postfilter requires a non-empty prompt.")

    serialized = json.dumps(
        [_serialize_datapoint(dp) for dp in datapoints],
        ensure_ascii=False,
        indent=2,
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a data quality judge. You will receive a JSON array of "
                f"{len(datapoints)} datapoints. For EACH datapoint, decide whether "
                "it satisfies the user's quality criteria. Return exactly "
                f"{len(datapoints)} assessments in the same order as the input array. "
                "Be strict but fair — judge each datapoint independently."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Quality criteria:\n{prompt.strip()}\n\n"
                f"Datapoints ({len(datapoints)} items):\n{serialized}"
            ),
        },
    ]


def batch_postfilter(
    datapoints: list[Any],
    prompt: str | None = None,
    *,
    model: ModelRuntime | None = None,
    batch_size: int = 10,
) -> list[PostfilterAssessment]:
    """Score multiple datapoints in batched LLM calls.

    Args:
        datapoints: List of datapoints to assess (dicts, strings, or AlpacaDataset).
        prompt: Quality criteria prompt.
        model: ModelRuntime to use.
        batch_size: Max datapoints per single LLM call.

    Returns:
        List of PostfilterAssessment, one per input datapoint, in order.
    """
    if model is None:
        model = OpenAIModel()
    if prompt is None:
        prompt = DEFAULT_POSTFILTER_PROMPT

    results: list[PostfilterAssessment] = []
    for i in range(0, len(datapoints), batch_size):
        batch = datapoints[i : i + batch_size]
        messages = _build_batch_postfilter_messages(batch, prompt)
        assessment = model.generate_structured(
            messages,
            BatchPostfilterAssessment,
            temperature=0.0,
            max_tokens=500 * len(batch),
        )
        # Ensure we got the right number of assessments.
        if len(assessment.assessments) != len(batch):
            # Fallback: pad or truncate
            if len(assessment.assessments) < len(batch):
                assessment.assessments.extend(
                    PostfilterAssessment(is_quality=False, reason="Missing assessment")
                    for _ in range(len(batch) - len(assessment.assessments))
                )
            else:
                assessment.assessments = assessment.assessments[: len(batch)]
        results.extend(assessment.assessments)

    return results


async def abatch_postfilter(
    datapoints: list[Any],
    prompt: str | None = None,
    *,
    model: ModelRuntime | None = None,
    batch_size: int = 10,
) -> list[PostfilterAssessment]:
    """Async variant of batch_postfilter."""
    if model is None:
        model = OpenAIModel(async_mode=True)
    if prompt is None:
        prompt = DEFAULT_POSTFILTER_PROMPT

    async def _process_batch(batch: list[Any]) -> list[PostfilterAssessment]:
        messages = _build_batch_postfilter_messages(batch, prompt)
        assessment = await model.agenerate_structured(
            messages,
            BatchPostfilterAssessment,
            temperature=0.0,
            max_tokens=500 * len(batch),
        )
        if len(assessment.assessments) != len(batch):
            if len(assessment.assessments) < len(batch):
                assessment.assessments.extend(
                    PostfilterAssessment(is_quality=False, reason="Missing assessment")
                    for _ in range(len(batch) - len(assessment.assessments))
                )
            else:
                assessment.assessments = assessment.assessments[: len(batch)]
        return assessment.assessments

    batches = [
        datapoints[i : i + batch_size]
        for i in range(0, len(datapoints), batch_size)
    ]
    batch_results = await asyncio.gather(*[_process_batch(b) for b in batches])
    return [a for batch in batch_results for a in batch]


def batch_filter_rows(
    rows: list[AlpacaDataset],
    prompt: str | None,
    model: ModelRuntime,
    *,
    batch_size: int = 10,
) -> list[AlpacaDataset]:
    """Drop-in replacement for _filter_rows() using batch postfilter."""
    if prompt is None:
        return rows

    assessments = batch_postfilter(
        [row.model_dump() for row in rows],
        prompt,
        model=model,
        batch_size=batch_size,
    )
    return [
        row
        for row, assessment in zip(rows, assessments, strict=True)
        if assessment.is_quality
    ]


async def abatch_filter_rows(
    rows: list[AlpacaDataset],
    prompt: str | None,
    model: ModelRuntime,
    *,
    batch_size: int = 10,
) -> list[AlpacaDataset]:
    """Async drop-in replacement for _afilter_rows() using batch postfilter."""
    if prompt is None:
        return rows

    assessments = await abatch_postfilter(
        [row.model_dump() for row in rows],
        prompt,
        model=model,
        batch_size=batch_size,
    )
    return [
        row
        for row, assessment in zip(rows, assessments, strict=True)
        if assessment.is_quality
    ]
