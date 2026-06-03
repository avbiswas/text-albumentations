"""High-level entry points: ta.augment(...) / ta.save(...).

Designed so the common path is one import and one call, while every knob
(custom augmentation instances, custom runtimes) stays available for
power users via the existing lower-level APIs.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.meta import apply_best_augmentations
from text_albumentations.models import OpenAIModel
from text_albumentations.reasoning import add_reasoning_to_dataset
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset, save_dataset


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


def list_tasks() -> dict[str, str]:
    """Names and selection hints of all built-in single-passage tasks."""
    return {
        name: aug.selection_hint or ""
        for name, aug in _build_task_registry().items()
    }


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


def _resolve_tasks(
    tasks: list[str | BaseSingleChunkAugmentation],
    registry: dict[str, BaseSingleChunkAugmentation],
) -> list[BaseSingleChunkAugmentation]:
    resolved = []
    for task in tasks:
        if isinstance(task, str):
            if task not in registry:
                raise ValueError(
                    f"Unknown task '{task}'. Available tasks: {sorted(registry)}"
                )
            resolved.append(registry[task])
        else:
            resolved.append(task)
    return resolved


def augment(
    text: str,
    tasks: list[str | BaseSingleChunkAugmentation] | None = None,
    *,
    model: ModelRuntime | None = None,
    add_reasoning: bool = False,
    quality_filter: bool = True,
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

    registry = _build_task_registry()

    if tasks is None:
        dataset = apply_best_augmentations(
            text,
            list(registry.items()),
            model,
            enable_quality_filter=quality_filter,
            add_reasoning=add_reasoning,
        )
    else:
        dataset = []
        for augmentation in _resolve_tasks(tasks, registry):
            dataset.extend(run_augmentation(text, augmentation, model))
        if add_reasoning:
            dataset = add_reasoning_to_dataset(text, dataset, model)

    if save_to is not None:
        save_dataset(dataset, save_to)
    return dataset


save = save_dataset
