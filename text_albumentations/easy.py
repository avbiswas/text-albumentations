"""High-level entry points: ta.augment(...) / ta.save(...).

Designed so the common path is one import and one call, while every knob
(custom augmentation instances, custom runtimes) stays available for
power users via the existing lower-level APIs.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel

from text_albumentations.base import (
    BaseMultiChunkAugmentation,
    BaseSingleChunkAugmentation,
)
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.meta import MetaAugmentation
from text_albumentations.models import OpenAIModel
from text_albumentations.postfilter import apostfilter, postfilter
from text_albumentations.reasoning import (
    aadd_reasoning_to_dataset,
    add_reasoning_to_dataset,
)
from text_albumentations.runner import arun_augmentation, run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset, save_dataset

SelectionMode = Literal["auto", "explicit", "sample"]
TaskSpec = (
    list[str | BaseSingleChunkAugmentation]
    | tuple[str | BaseSingleChunkAugmentation, ...]
    | dict[str, float]
    | None
)

DEFAULT_POSTFILTER_PROMPT = """\
A quality generated training datapoint should be useful for supervised fine-tuning.
Keep the row only if the instruction is clear, the input contains enough context,
and the output directly satisfies the instruction without contradiction,
unsupported claims, truncation, or malformed boilerplate.
"""


@dataclass(frozen=True)
class TaskSelection:
    is_low_quality: bool
    low_quality_reason: str
    selected_tasks: list[str]
    reasoning: str

    @property
    def is_quality(self) -> bool:
        return not self.is_low_quality


def _build_task_registry() -> dict[str, BaseSingleChunkAugmentation]:
    from text_albumentations import tasks

    return {
        "bullets": tasks.bullet_augmentation,
        "qa_pairs": tasks.qa_pair_augmentation,
        "rephrase": tasks.rephrase_augmentation,
        "continuation": tasks.continuation_augmentation,
        "triplets": tasks.triplet_augmentation,
        "summarize": tasks.summarize_augmentation,
        "title": tasks.title_augmentation,
        "cloze": tasks.cloze_augmentation,
        "extractive_qa": tasks.extractive_qa_augmentation,
        "classification": tasks.classification_augmentation,
        "style_transfer": tasks.style_transfer_augmentation,
        "backtranslation": tasks.backtranslation_augmentation,
        "counterfactual": tasks.counterfactual_augmentation,
    }


def _build_multi_task_registry() -> dict[str, BaseMultiChunkAugmentation]:
    from text_albumentations import tasks

    return {
        "retrieval": tasks.retrieval_augmentation,
        "comparison": tasks.comparison_augmentation,
    }


def list_tasks() -> dict[str, str]:
    """Names and selection hints of all built-in single-passage tasks."""
    return {
        name: aug.selection_hint or ""
        for name, aug in _build_task_registry().items()
    }


def list_multi_tasks() -> dict[str, str]:
    """Names and hints of all built-in multi-passage tasks."""
    return {
        name: aug.selection_hint or ""
        for name, aug in _build_multi_task_registry().items()
    }


def get_task(name: str) -> BaseSingleChunkAugmentation:
    registry = _build_task_registry()
    if name not in registry:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(registry)}")
    return registry[name]


def get_multi_task(name: str) -> BaseMultiChunkAugmentation:
    registry = _build_multi_task_registry()
    if name not in registry:
        raise ValueError(
            f"Unknown multi-task '{name}'. Available multi-tasks: {sorted(registry)}"
        )
    return registry[name]


def _build_output_renderer(
    schema: type[BaseModel],
    output: str | Callable | None,
) -> Callable[[BaseModel], str]:
    if output is None:
        fields = list(schema.model_fields)
        if len(fields) != 1:
            raise ValueError(
                f"Schema '{schema.__name__}' has fields {fields}; pass output= "
                "to say how they become the training row's output. Either a "
                "template string like '{" + fields[0] + "}' or a callable."
            )
        only_field = fields[0]
        return lambda out: str(getattr(out, only_field))
    if isinstance(output, str):
        return lambda out: output.format(**out.model_dump())
    return output


def task(
    *,
    prompt: str,
    schema: type[BaseModel],
    output: str | Callable | None = None,
    instruction: str | None = None,
    rows: Callable[[str, BaseModel], list[AlpacaDataset]] | None = None,
    selection_hint: str | None = None,
    **augmentation_kwargs,
) -> BaseSingleChunkAugmentation:
    """Define a custom augmentation without writing adapter or class boilerplate.

    - ``prompt``: system prompt sent to the model.
    - ``schema``: pydantic model the LLM must produce (enforced by Outlines).
    - ``output``: how the schema becomes the training row's output — a template
      string over schema fields ("{statistic} — {context}"), a callable
      ``(output) -> str``, or omitted when the schema has exactly one field.
    - ``instruction``: the training row's instruction; defaults to ``prompt``.
    - ``rows``: full-control alternative to ``output``/``instruction`` — a
      callable ``(passage, output) -> list[AlpacaDataset]`` producing any
      number of rows.
    - ``selection_hint``: tells the smart-switch when this task fits a
      passage, if the task is used with auto-pick.

    Extra keyword arguments (``temperature``, ``num_generations``,
    ``variations``, ...) are forwarded to the augmentation.
    """
    if rows is not None and output is not None:
        raise ValueError("Pass either output= or rows=, not both.")

    if rows is not None:
        convert = rows
    else:
        render = _build_output_renderer(schema, output)
        row_instruction = instruction if instruction is not None else prompt.strip()

        def convert(passages: str, out: BaseModel) -> list[AlpacaDataset]:
            return [
                AlpacaDataset(
                    instruction=row_instruction,
                    input=passages,
                    output=render(out),
                )
            ]

    class _FunctionAdapter(BaseAlpacaAdapter):
        def convert(self, passages, out):
            return convert(passages, out)

    class _FunctionTask(BaseSingleChunkAugmentation):
        pass

    _FunctionTask.schema = schema
    _FunctionTask.system_prompt = prompt
    _FunctionTask.adapters = (_FunctionAdapter(),)
    _FunctionTask.__name__ = f"{schema.__name__}Task"
    if selection_hint is not None:
        _FunctionTask.selection_hint = selection_hint
    return _FunctionTask(**augmentation_kwargs)


def resolve_tasks(
    tasks: list[str | BaseSingleChunkAugmentation],
) -> list[BaseSingleChunkAugmentation]:
    registry = _build_task_registry()
    resolved = []
    for task in tasks:
        if isinstance(task, str):
            resolved.append(get_task(task))
        else:
            resolved.append(task)
    return resolved


def resolve_multi_tasks(
    tasks: list[str | BaseMultiChunkAugmentation],
) -> list[BaseMultiChunkAugmentation]:
    resolved = []
    for task in tasks:
        if isinstance(task, str):
            resolved.append(get_multi_task(task))
        else:
            resolved.append(task)
    return resolved


def _infer_selection_mode(tasks: TaskSpec, selection_mode: SelectionMode | None) -> SelectionMode:
    if selection_mode is not None:
        return selection_mode
    if tasks is None:
        return "auto"
    if isinstance(tasks, dict):
        return "sample"
    return "explicit"


def _task_entries(
    tasks: list[str] | tuple[str, ...] | None,
) -> list[tuple[str, BaseSingleChunkAugmentation]]:
    registry = _build_task_registry()
    if tasks is None:
        return list(registry.items())
    entries = []
    for name in tasks:
        if not isinstance(name, str):
            raise TypeError("selection_mode='auto' requires named built-in tasks.")
        entries.append((name, get_task(name)))
    return entries


def _sample_task_names(tasks: dict[str, float]) -> list[str]:
    selected = []
    for name, probability in tasks.items():
        if probability < 0 or probability > 1:
            raise ValueError("Task probabilities must be between 0 and 1.")
        get_task(name)
        if random.random() < probability:
            selected.append(name)
    return selected


def _normalize_postfilter_prompt(postfilter_setting: bool | str) -> str | None:
    if postfilter_setting is False:
        return None
    if postfilter_setting is True:
        return DEFAULT_POSTFILTER_PROMPT
    if not postfilter_setting.strip():
        raise ValueError("postfilter criteria must be non-empty.")
    return postfilter_setting


def _filter_rows(
    rows: list[AlpacaDataset],
    prompt: str | None,
    model: ModelRuntime,
) -> list[AlpacaDataset]:
    if prompt is None:
        return rows
    kept = []
    for row in rows:
        assessment = postfilter(row.model_dump(), prompt, model=model)
        if assessment.is_quality:
            kept.append(row)
    return kept


async def _afilter_rows(
    rows: list[AlpacaDataset],
    prompt: str | None,
    model: ModelRuntime,
) -> list[AlpacaDataset]:
    if prompt is None:
        return rows
    assessments = await asyncio.gather(
        *[apostfilter(row.model_dump(), prompt, model=model) for row in rows]
    )
    return [
        row
        for row, assessment in zip(rows, assessments, strict=True)
        if assessment.is_quality
    ]


def select_tasks(
    text: str,
    tasks: list[str] | tuple[str, ...] | None = None,
    *,
    model: ModelRuntime | None = None,
    prefilter: bool = True,
) -> TaskSelection:
    if model is None:
        model = OpenAIModel()
    meta = MetaAugmentation(_task_entries(tasks), prefilter=prefilter)
    selection = meta.generate_one(text, model)
    selected = [] if prefilter and selection.is_low_quality else selection.selected
    return TaskSelection(
        is_low_quality=selection.is_low_quality,
        low_quality_reason=selection.low_quality_reason,
        selected_tasks=list(selected),
        reasoning=selection.reasoning,
    )


async def aselect_tasks(
    text: str,
    tasks: list[str] | tuple[str, ...] | None = None,
    *,
    model: ModelRuntime | None = None,
    prefilter: bool = True,
) -> TaskSelection:
    if model is None:
        model = OpenAIModel(async_mode=True)
    meta = MetaAugmentation(_task_entries(tasks), prefilter=prefilter)
    selection = await meta.agenerate_one(text, model)
    selected = [] if prefilter and selection.is_low_quality else selection.selected
    return TaskSelection(
        is_low_quality=selection.is_low_quality,
        low_quality_reason=selection.low_quality_reason,
        selected_tasks=list(selected),
        reasoning=selection.reasoning,
    )


def augment(
    text: str,
    tasks: TaskSpec = None,
    *,
    selection_mode: SelectionMode | None = None,
    model: ModelRuntime | None = None,
    add_reasoning: bool = False,
    prefilter: bool = True,
    postfilter: bool | str = False,
    save_to: str | None = None,
) -> list[AlpacaDataset]:
    """Generate training rows from a passage with one call.

    By default an LLM smart-switch picks the augmentations that fit the
    passage and rejects low-quality input. Pass ``tasks`` (names from
    ``list_tasks()`` or augmentation instances) to choose explicitly.

    ``model`` is a model primitive (``OpenAIModel``, ``LocalMLXModel``,
    ``LocalHFModel``, or any custom ``ModelRuntime``). When omitted, an
    ``OpenAIModel`` is built from the ``TEXT_ALBUMENTATIONS_MODEL``,
    ``OPENAI_BASE_URL``, and ``OPENAI_API_KEY`` environment variables.
    """
    if model is None:
        model = OpenAIModel()

    mode = _infer_selection_mode(tasks, selection_mode)
    postfilter_prompt = _normalize_postfilter_prompt(postfilter)

    if mode == "auto":
        if isinstance(tasks, dict):
            raise TypeError("selection_mode='auto' does not accept probability maps.")
        selection = select_tasks(
            text,
            tasks,
            model=model,
            prefilter=prefilter,
        )
        dataset = []
        for augmentation in resolve_tasks(selection.selected_tasks):
            dataset.extend(run_augmentation(text, augmentation, model))
    elif mode == "explicit":
        if tasks is None or isinstance(tasks, dict):
            raise TypeError("selection_mode='explicit' requires a task list.")
        dataset = []
        for augmentation in resolve_tasks(list(tasks)):
            dataset.extend(run_augmentation(text, augmentation, model))
    elif mode == "sample":
        if not isinstance(tasks, dict):
            raise TypeError("selection_mode='sample' requires a probability map.")
        dataset = []
        for augmentation in resolve_tasks(_sample_task_names(tasks)):
            dataset.extend(run_augmentation(text, augmentation, model))
    else:
        raise ValueError("selection_mode must be one of: auto, explicit, sample.")

    dataset = _filter_rows(dataset, postfilter_prompt, model)
    if add_reasoning:
        dataset = add_reasoning_to_dataset(text, dataset, model)

    if save_to is not None:
        save_dataset(dataset, save_to)
    return dataset


async def aaugment(
    text: str,
    tasks: TaskSpec = None,
    *,
    selection_mode: SelectionMode | None = None,
    model: ModelRuntime | None = None,
    add_reasoning: bool = False,
    prefilter: bool = True,
    postfilter: bool | str = False,
    save_to: str | None = None,
) -> list[AlpacaDataset]:
    if model is None:
        model = OpenAIModel(async_mode=True)

    mode = _infer_selection_mode(tasks, selection_mode)
    postfilter_prompt = _normalize_postfilter_prompt(postfilter)

    if mode == "auto":
        if isinstance(tasks, dict):
            raise TypeError("selection_mode='auto' does not accept probability maps.")
        selection = await aselect_tasks(
            text,
            tasks,
            model=model,
            prefilter=prefilter,
        )
        augmentations = resolve_tasks(selection.selected_tasks)
    elif mode == "explicit":
        if tasks is None or isinstance(tasks, dict):
            raise TypeError("selection_mode='explicit' requires a task list.")
        augmentations = resolve_tasks(list(tasks))
    elif mode == "sample":
        if not isinstance(tasks, dict):
            raise TypeError("selection_mode='sample' requires a probability map.")
        augmentations = resolve_tasks(_sample_task_names(tasks))
    else:
        raise ValueError("selection_mode must be one of: auto, explicit, sample.")

    datasets = await asyncio.gather(
        *[arun_augmentation(text, augmentation, model) for augmentation in augmentations]
    )
    dataset = [row for rows in datasets for row in rows]
    dataset = await _afilter_rows(dataset, postfilter_prompt, model)
    if add_reasoning:
        dataset = await aadd_reasoning_to_dataset(text, dataset, model)
    if save_to is not None:
        save_dataset(dataset, save_to)
    return dataset


save = save_dataset
