import asyncio

import pytest
from pydantic import BaseModel

from text_albumentations.runtime import ModelRuntime
from text_albumentations.tasks.retrieval import (
    NoAnswerReason,
    RetrievalAugmentation,
    RetrievalReason,
    UniqueQuestions,
)


class RetrievalFakeModel(ModelRuntime):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_structured(
        self,
        messages,
        output_type,
        *,
        temperature=0.2,
        max_tokens=5000,
    ) -> BaseModel:
        self.calls.append(output_type.__name__)
        if output_type is UniqueQuestions:
            return UniqueQuestions(questions=["q1", "q2", "q3"])
        if output_type is RetrievalReason:
            return RetrievalReason(why="positive reason")
        if output_type is NoAnswerReason:
            return NoAnswerReason(why="negative reason")
        raise AssertionError(f"Unexpected schema: {output_type.__name__}")

    def generate_variation(
        self,
        output,
        output_type,
        *,
        context=None,
        temperature=0.5,
        max_tokens=5000,
    ) -> BaseModel:
        raise NotImplementedError


class AsyncRetrievalFakeModel(RetrievalFakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.active_reason_calls = 0
        self.max_active_reason_calls = 0

    async def agenerate_structured(
        self,
        messages,
        output_type,
        *,
        temperature=0.2,
        max_tokens=5000,
    ) -> BaseModel:
        self.calls.append(output_type.__name__)
        if output_type is UniqueQuestions:
            return UniqueQuestions(questions=["q1", "q2"])

        self.active_reason_calls += 1
        self.max_active_reason_calls = max(
            self.max_active_reason_calls,
            self.active_reason_calls,
        )
        await asyncio.sleep(0.01)
        self.active_reason_calls -= 1

        if output_type is RetrievalReason:
            return RetrievalReason(why="positive reason")
        if output_type is NoAnswerReason:
            return NoAnswerReason(why="negative reason")
        raise AssertionError(f"Unexpected schema: {output_type.__name__}")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_retrieval_controls_passage_question_and_negative_generation(monkeypatch):
    monkeypatch.setattr("text_albumentations.tasks.retrieval.random.shuffle", lambda _: None)
    model = RetrievalFakeModel()
    augmentation = RetrievalAugmentation(
        max_passages=2,
        max_questions_per_passage=1,
        include_negative_examples=False,
    )

    rows = augmentation.build_dataset(["p1", "p2", "p3"], model)

    assert len(rows) == 4
    assert model.calls == [
        "UniqueQuestions",
        "UniqueQuestions",
        "RetrievalReason",
        "RetrievalReason",
    ]
    assert all("Passage 3:" not in row.input for row in rows)
    assert all("None" not in row.output for row in rows)


@pytest.mark.anyio
async def test_retrieval_async_reason_generation_runs_concurrently(monkeypatch):
    monkeypatch.setattr("text_albumentations.tasks.retrieval.random.shuffle", lambda _: None)
    model = AsyncRetrievalFakeModel()
    augmentation = RetrievalAugmentation()

    rows = await augmentation.abuild_dataset(["p1", "p2"], model)

    assert len(rows) == 16
    assert model.calls.count("RetrievalReason") == 4
    assert model.calls.count("NoAnswerReason") == 4
    assert model.max_active_reason_calls > 1
