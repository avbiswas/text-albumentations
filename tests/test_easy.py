"""Offline tests for the high-level ta.augment API, using FakeModel."""

import json

import pytest

import text_albumentations as ta
from text_albumentations.meta import MetaSelection
from text_albumentations.reasoning import ReasoningTrace
from text_albumentations.tasks.style_transfer import StyleRewrite, StyleTransferAugmentation
from text_albumentations.tasks.summarize import Summary
from text_albumentations.tasks.title import TitleHeadline

from conftest import FakeModel

EXPECTED_TASKS = {
    "bullets",
    "qa_pairs",
    "rephrase",
    "continuation",
    "triplets",
    "summarize",
    "title",
    "cloze",
    "extractive_qa",
    "classification",
    "style_transfer",
    "backtranslation",
    "counterfactual",
}


def test_list_tasks_contains_all_builtins():
    assert set(ta.list_tasks()) == EXPECTED_TASKS


def test_augment_explicit_tasks(passage):
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})
    rows = ta.augment(passage, tasks=["title"], model=model)
    assert [row.output for row in rows] == ["T", "H"]


def test_augment_unknown_task_raises(passage):
    with pytest.raises(ValueError, match="Unknown task 'nope'"):
        ta.augment(passage, tasks=["nope"], model=FakeModel({}))


def test_augment_mixes_names_and_instances(passage):
    model = FakeModel(
        {
            "TitleHeadline": TitleHeadline(title="T", headline="H"),
            "StyleRewrite": StyleRewrite(rewritten="rewritten text"),
        }
    )
    rows = ta.augment(
        passage,
        tasks=["title", StyleTransferAugmentation(style="casual")],
        model=model,
    )
    assert len(rows) == 3
    assert rows[-1].output == "rewritten text"


def test_augment_auto_mode_runs_selected_tasks(passage):
    model = FakeModel(
        {
            "MetaSelection": MetaSelection(
                is_low_quality=False,
                low_quality_reason="",
                selected=["title"],
                reasoning="title fits",
            ),
            "TitleHeadline": TitleHeadline(title="T", headline="H"),
        }
    )
    rows = ta.augment(passage, model=model)
    assert [row.output for row in rows] == ["T", "H"]


def test_augment_auto_mode_quality_gate_rejects(passage):
    model = FakeModel(
        {
            "MetaSelection": MetaSelection(
                is_low_quality=True,
                low_quality_reason="junk",
                selected=[],
                reasoning="",
            ),
        }
    )
    assert ta.augment("asdf qwerty", model=model) == []


def test_augment_add_reasoning(passage):
    model = FakeModel(
        {
            "Summary": Summary(tldr="t", summary="s"),
            "ReasoningTrace": ReasoningTrace(reasoning="because the passage says so"),
        }
    )
    rows = ta.augment(passage, tasks=["summarize"], model=model, add_reasoning=True)
    assert all(row.reasoning == "because the passage says so" for row in rows)


def test_augment_save_to_writes_jsonl(passage, tmp_path):
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})
    out_file = tmp_path / "rows.jsonl"
    ta.augment(passage, tasks=["title"], model=model, save_to=str(out_file))

    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["output"] == "T"
