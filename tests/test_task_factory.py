"""Offline tests for ta.task() — terse custom task definition."""

import pytest
from pydantic import BaseModel, Field

import text_albumentations as ta
from text_albumentations.utils import AlpacaDataset

from conftest import FakeModel


class KeyStat(BaseModel):
    statistic: str = Field(max_length=200)
    context: str = Field(max_length=300)


class Tldr(BaseModel):
    tldr: str = Field(max_length=200)


FAKE = FakeModel(
    {
        "KeyStat": KeyStat(statistic="28.4 BLEU", context="WMT14 En-De score"),
        "Tldr": Tldr(tldr="Attention replaces recurrence."),
    }
)


def test_template_output(passage):
    key_stat = ta.task(
        prompt="Extract the most important statistic.",
        schema=KeyStat,
        output="{statistic} — {context}",
    )
    rows = ta.augment(passage, tasks=[key_stat], model=FAKE)
    assert len(rows) == 1
    assert rows[0].output == "28.4 BLEU — WMT14 En-De score"
    assert rows[0].instruction == "Extract the most important statistic."
    assert rows[0].input == passage


def test_callable_output_and_custom_instruction(passage):
    key_stat = ta.task(
        prompt="Extract the most important statistic.",
        schema=KeyStat,
        output=lambda out: out.statistic.upper(),
        instruction="Find the key number.",
    )
    rows = ta.augment(passage, tasks=[key_stat], model=FAKE)
    assert rows[0].output == "28.4 BLEU"
    assert rows[0].instruction == "Find the key number."


def test_custom_task_instruction_variants(monkeypatch, passage):
    monkeypatch.setattr("text_albumentations.runner.random.choice", lambda seq: seq[-1])
    key_stat = ta.task(
        prompt="Extract the most important statistic.",
        schema=KeyStat,
        output="{statistic}",
        instruction="Find the key number.",
        instruction_variants=[
            "Identify the most important number in this passage.",
            "Extract the key statistic from this passage.",
        ],
    )

    rows = ta.augment(
        passage,
        tasks=[key_stat],
        model=FAKE,
        sample_instruction_template=True,
    )

    assert rows[0].instruction == "Extract the key statistic from this passage."
    assert rows[0].output == "28.4 BLEU"


def test_custom_task_batch_samples_instruction_variants_by_default(monkeypatch, passage):
    monkeypatch.setattr("text_albumentations.runner.random.choice", lambda seq: seq[-1])
    key_stat = ta.task(
        prompt="Extract the most important statistic.",
        schema=KeyStat,
        output="{statistic}",
        instruction="Find the key number.",
        instruction_variants=[
            "Identify the most important number in this passage.",
            "Extract the key statistic from this passage.",
        ],
    )

    rows = ta.run_batch_augmentation(
        [passage, passage],
        key_stat,
        FAKE,
    )

    assert [row.instruction for row in rows] == [
        "Extract the key statistic from this passage.",
        "Extract the key statistic from this passage.",
    ]


def test_single_field_schema_needs_no_output(passage):
    tldr = ta.task(prompt="Write a one-sentence TLDR.", schema=Tldr)
    rows = ta.augment(passage, tasks=[tldr], model=FAKE)
    assert rows[0].output == "Attention replaces recurrence."


def test_multi_field_schema_without_output_raises():
    with pytest.raises(ValueError, match="pass output="):
        ta.task(prompt="p", schema=KeyStat)


def test_rows_callable_full_control(passage):
    def to_rows(passage_text, out):
        return [
            AlpacaDataset(instruction="i1", input=passage_text, output=out.statistic),
            AlpacaDataset(instruction="i2", input=passage_text, output=out.context),
        ]

    key_stat = ta.task(prompt="Extract.", schema=KeyStat, rows=to_rows)
    rows = ta.augment(passage, tasks=[key_stat], model=FAKE)
    assert [row.instruction for row in rows] == ["i1", "i2"]


def test_instruction_variants_require_standard_task_rows():
    with pytest.raises(ValueError, match="instruction_variants"):
        ta.task(
            prompt="p",
            schema=KeyStat,
            rows=lambda p, o: [],
            instruction_variants=["variant"],
        )


def test_output_and_rows_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        ta.task(prompt="p", schema=KeyStat, output="{statistic}", rows=lambda p, o: [])


def test_generation_kwargs_forwarded():
    aug = ta.task(prompt="p", schema=Tldr, temperature=0.9, num_generations=2)
    assert aug.temperature == 0.9
    assert aug.num_generations == 2
