from __future__ import annotations

import os
import time
import argparse
from enum import Enum
from typing import Literal

import openai
import outlines
from outlines.inputs import Chat
from pydantic import BaseModel, ConfigDict, Field, conint

import text_albumentations as ta


BASE_URL = "https://openrouter.ai/api/v1"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=8, max_length=220)
    supports_claim: bool
    confidence: float = Field(ge=0, le=1)


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["grounding", "format", "completeness", "contradiction", "noise"]
    severity: Severity
    explanation: str = Field(min_length=20, max_length=240)
    evidence: list[EvidenceSpan] = Field(min_length=1, max_length=2)


class QualityRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_quality: bool
    score: conint(ge=0, le=100)
    decision: Literal["keep", "reject", "borderline"]
    primary_issue: Literal["none", "unsupported", "truncated", "ambiguous", "malformed"]
    issues: list[Issue] = Field(min_length=1, max_length=3)
    corrected_instruction: str | None = Field(max_length=180)
    tags: list[Literal["paper", "qa", "summary", "math", "table", "noisy"]] = Field(
        min_length=1,
        max_length=4,
    )


MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a strict dataset quality judge. Return only JSON. "
            "Use the requested schema exactly."
        ),
    },
    {
        "role": "user",
        "content": """\
Judge this datapoint.

instruction: Generate one question and its corresponding answer from this passage in markdown format.

input: produces an output in response. Such prompts may be textual--"Write a poem about trees."--or take other forms: images, audio, videos, or a combination thereof. The ability to prompt models, particularly prompting with natural language, makes them easy to interact with and use flexibly across a wide range of use cases. Knowing how to effectively structure, evaluate, and perform other tasks with prompts is essential to using these models. Empirically, better prompts lead to improved results across a wide range of tasks. However, as prompting is an emerging field, the use of prompts continues to be poorly understood, with only a fraction of existing terminologies and techniques being well-known among practitioners.

output: **Question:** What is the current state of understanding regarding prompting?

**Answer:** Prompting is an emerging field where the use of prompts continues to be poorly understood, with only a fraction of existing terminologies and techniques being well-known among practitioners.
""",
    },
]


def run_case(name: str, fn):
    started = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - started
        print(f"\n{name}: OK in {elapsed:.2f}s")
        print(result.model_dump_json(indent=2))
    except Exception as error:
        elapsed = time.perf_counter() - started
        print(f"\n{name}: FAILED in {elapsed:.2f}s")
        print(type(error).__name__)
        print(error)


def old_outlines_json_schema(model_name: str) -> QualityRubric:
    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = outlines.from_openai(client, model_name)
    raw = model(
        Chat(MESSAGES),
        QualityRubric,
        temperature=0,
        max_completion_tokens=800,
    )
    return QualityRubric.model_validate_json(raw)


def new_text_albumentations_json_object(model_name: str) -> QualityRubric:
    model = ta.OpenAIModel(
        model_name,
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        response_format="json_object",
    )
    return model.generate_structured(
        MESSAGES,
        QualityRubric,
        temperature=0,
        max_tokens=800,
    )


def text_albumentations_default_json_schema(model_name: str) -> QualityRubric:
    model = ta.OpenAIModel(
        model_name,
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        response_format="json_schema",
    )
    return model.generate_structured(
        MESSAGES,
        QualityRubric,
        temperature=0,
        max_tokens=800,
    )


def text_albumentations_auto(model_name: str) -> QualityRubric:
    model = ta.OpenAIModel(
        model_name,
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    return model.generate_structured(
        MESSAGES,
        QualityRubric,
        temperature=0,
        max_tokens=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    args = parser.parse_args()

    print(f"model={args.model}")
    run_case("old_outlines_json_schema", lambda: old_outlines_json_schema(args.model))
    run_case(
        "text_albumentations_auto",
        lambda: text_albumentations_auto(args.model),
    )
    run_case(
        "text_albumentations_default_json_schema",
        lambda: text_albumentations_default_json_schema(args.model),
    )
    run_case(
        "new_text_albumentations_json_object",
        lambda: new_text_albumentations_json_object(args.model),
    )
