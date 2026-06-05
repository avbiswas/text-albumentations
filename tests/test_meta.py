"""Offline tests for the smart-switch (MetaAugmentation)."""

import pytest
from pydantic import ValidationError

from text_albumentations.meta import MetaAugmentation, _build_selection_prompt
from text_albumentations.tasks.summarize import summarize_augmentation
from text_albumentations.tasks.title import title_augmentation


@pytest.fixture
def meta() -> MetaAugmentation:
    return MetaAugmentation(
        [
            ("title", title_augmentation),
            ("summarize", summarize_augmentation),
        ]
    )


def test_selected_is_constrained_to_real_names(meta):
    schema = meta.get_schema()
    selection = schema(selected=["title"])
    assert selection.selected == ["title"]

    with pytest.raises(ValidationError):
        schema(selected=["hallucinated_task"])


def test_selection_prompt_built_from_hints(meta):
    prompt = _build_selection_prompt(meta._aug_options)
    assert "- title: pick when the passage has one coherent" in prompt
    assert "- summarize: pick when the passage is longer than a couple" in prompt
    # the augmenter-facing system prompts must never leak into the selector
    assert summarize_augmentation.system_prompt.strip() not in prompt


def test_augmentation_without_hint_raises():
    from text_albumentations.easy import task as ta_task
    from pydantic import BaseModel

    class Tldr(BaseModel):
        tldr: str

    hintless = ta_task(prompt="Write a TLDR.", schema=Tldr)
    with pytest.raises(ValueError, match="no selection_hint"):
        MetaAugmentation([("tldr", hintless)])


def test_hint_override_via_third_tuple_element():
    from text_albumentations.easy import task as ta_task
    from pydantic import BaseModel

    class Tldr(BaseModel):
        tldr: str

    hintless = ta_task(prompt="Write a TLDR.", schema=Tldr)
    meta = MetaAugmentation([("tldr", hintless, "the passage is compressible.")])
    prompt = _build_selection_prompt(meta._aug_options)
    assert "- tldr: pick when the passage is compressible." in prompt


def test_all_builtin_tasks_carry_selection_hints():
    from text_albumentations.easy import _build_task_registry

    for name, aug in _build_task_registry().items():
        assert aug.selection_hint, f"{name} has no selection_hint"


def test_ta_task_selection_hint_reaches_the_switch():
    import text_albumentations as ta
    from pydantic import BaseModel

    class Tldr(BaseModel):
        tldr: str

    custom = ta.task(
        prompt="Write a TLDR.",
        schema=Tldr,
        selection_hint="the passage is dense enough to compress.",
    )
    meta = MetaAugmentation([("tldr", custom)])
    prompt = _build_selection_prompt(meta._aug_options)
    assert "pick when the passage is dense enough to compress." in prompt
