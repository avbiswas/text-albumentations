"""Offline tests for the high-level ta.augment API, using FakeModel."""

import json

import pytest

import text_albumentations as ta
from text_albumentations.meta import MetaSelection
from text_albumentations.postfilter import PostfilterAssessment
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


def test_public_task_resolvers():
    assert ta.get_task("title") is not None
    assert len(ta.resolve_tasks(["title", "summarize"])) == 2
    assert set(ta.list_multi_tasks()) == {"retrieval", "comparison"}
    assert ta.get_multi_task("retrieval") is not None
    assert len(ta.resolve_multi_tasks(["retrieval", "comparison"])) == 2


def test_augment_explicit_tasks(passage):
    # Deterministic legacy instruction text remains available by opting out.
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})
    rows = ta.augment(
        passage,
        tasks=["title"],
        model=model,
        sample_instruction_template=False,
    )
    assert [row.output for row in rows] == ["T", "H"]


def test_augment_samples_instruction_templates_by_default(monkeypatch, passage):
    monkeypatch.setattr("text_albumentations.runner.random.choice", lambda seq: seq[-1])
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})

    rows = ta.augment(
        passage,
        tasks=["title"],
        model=model,
    )

    assert rows[0].instruction == "Title this passage in a few words."
    assert rows[1].instruction == "Generate a short headline based on the passage."
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


def test_augment_auto_mode_uses_task_whitelist(passage):
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
    rows = ta.augment(
        passage,
        tasks=["title", "summarize"],
        selection_mode="auto",
        model=model,
    )
    assert [row.output for row in rows] == ["T", "H"]
    selection_prompt = model.calls[0][0][0]["content"]
    assert "- title:" in selection_prompt
    assert "- summarize:" in selection_prompt
    assert "- bullets:" not in selection_prompt


def test_select_tasks_returns_inspectable_selection(passage):
    model = FakeModel(
        {
            "MetaSelection": MetaSelection(
                is_low_quality=False,
                low_quality_reason="",
                selected=["title"],
                reasoning="title fits",
            ),
        }
    )
    selection = ta.select_tasks(
        passage,
        tasks=["title", "summarize"],
        model=model,
    )
    assert selection.is_quality is True
    assert selection.is_low_quality is False
    assert selection.selected_tasks == ["title"]
    assert selection.reasoning == "title fits"


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


def test_augment_prefilter_false_keeps_selected_tasks_for_low_quality_text(passage):
    model = FakeModel(
        {
            "MetaSelection": MetaSelection(
                is_low_quality=True,
                low_quality_reason="junk",
                selected=["title"],
                reasoning="title still requested",
            ),
            "TitleHeadline": TitleHeadline(title="T", headline="H"),
        }
    )
    rows = ta.augment("asdf qwerty", model=model, prefilter=False)
    assert [row.output for row in rows] == ["T", "H"]


def test_augment_sample_mode(monkeypatch, passage):
    monkeypatch.setattr("text_albumentations.easy.random.random", lambda: 0.2)
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})

    rows = ta.augment(
        passage,
        tasks={"title": 0.25, "summarize": 0.0},
        selection_mode="sample",
        model=model,
    )

    assert [row.output for row in rows] == ["T", "H"]


def test_augment_sample_mode_can_return_empty(monkeypatch, passage):
    monkeypatch.setattr("text_albumentations.easy.random.random", lambda: 0.9)
    rows = ta.augment(
        passage,
        tasks={"title": 0.25},
        selection_mode="sample",
        model=FakeModel({}),
    )
    assert rows == []


def test_augment_sample_mode_validates_probabilities(passage):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ta.augment(
            passage,
            tasks={"title": 1.5},
            selection_mode="sample",
            model=FakeModel({}),
        )


def test_augment_postfilter_keeps_only_quality_rows(passage):
    model = FakeModel(
        {
            "TitleHeadline": TitleHeadline(title="T", headline="H"),
            "PostfilterAssessment": PostfilterAssessment(
                is_quality=False,
                reason="Bad row.",
            ),
        }
    )

    rows = ta.augment(passage, tasks=["title"], model=model, postfilter=True)

    assert rows == []


def test_augment_add_reasoning(passage):
    model = FakeModel(
        {
            "Summary": Summary(tldr="t", summary="s"),
            "ReasoningTrace": ReasoningTrace(reasoning="because the passage says so"),
        }
    )
    rows = ta.augment(passage, tasks=["summarize"], model=model, add_reasoning=True)
    assert all(row.reasoning == "because the passage says so" for row in rows)


@pytest.mark.anyio
async def test_aaugment_explicit_tasks(passage):
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})
    rows = await ta.aaugment(passage, tasks=["title"], model=model)
    assert [row.output for row in rows] == ["T", "H"]


@pytest.mark.anyio
async def test_aselect_tasks(passage):
    model = FakeModel(
        {
            "MetaSelection": MetaSelection(
                is_low_quality=False,
                low_quality_reason="",
                selected=["title"],
                reasoning="title fits",
            ),
        }
    )
    selection = await ta.aselect_tasks(passage, tasks=["title"], model=model)
    assert selection.selected_tasks == ["title"]


def test_augment_save_to_writes_jsonl(passage, tmp_path):
    model = FakeModel({"TitleHeadline": TitleHeadline(title="T", headline="H")})
    out_file = tmp_path / "rows.jsonl"
    ta.augment(
        passage,
        tasks=["title"],
        model=model,
        save_to=str(out_file),
        sample_instruction_template=False,
    )

    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["output"] == "T"
