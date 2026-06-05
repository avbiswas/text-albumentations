from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.runner import arun_augmentation, run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset


class PassageQuality(BaseModel):
    is_quality: bool


_PREFILTER_SYSTEM_PROMPT = (
    "You are a text quality filter. "
    "Return is_quality=true only if the passage contains meaningful natural-language content "
    "with at least roughly ten words; return false for markup-only, code, random characters, "
    "headers, or boilerplate."
)


def prefilter_passage(text: str, runtime: ModelRuntime) -> bool:
    result = runtime.generate_structured(
        [
            {"role": "system", "content": _PREFILTER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        PassageQuality,
        temperature=0.1,
        max_tokens=20,
    )
    return result.is_quality


async def aprefilter_passage(text: str, runtime: ModelRuntime) -> bool:
    result = await runtime.agenerate_structured(
        [
            {"role": "system", "content": _PREFILTER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        PassageQuality,
        temperature=0.1,
        max_tokens=20,
    )
    return result.is_quality


# (name, augmentation) — the selection hint comes from the augmentation's
# `selection_hint` attribute. A third element overrides it.
AugmentationEntry = (
    tuple[str, BaseSingleChunkAugmentation]
    | tuple[str, BaseSingleChunkAugmentation, str]
)


class AugmentationOption(BaseModel):
    name: str
    selection_hint: str


class MetaSelection(BaseModel):
    selected: list[str] = Field(
        ...,
        description="Names of augmentations that would work well for this passage",
    )


def _build_selection_schema(names: tuple[str, ...]) -> type[MetaSelection]:
    if not names:
        return MetaSelection
    return create_model(
        "ConfiguredMetaSelection",
        __base__=MetaSelection,
        selected=(
            list[Literal[names]],  # type: ignore[valid-type]
            Field(
                ...,
                description="Names of augmentations that would work well for this passage",
            ),
        ),
    )


def _describe_option(option: AugmentationOption) -> str:
    return f"- {option.name}: pick when {option.selection_hint}"


def _build_selection_prompt(options: list[AugmentationOption]) -> str:
    aug_descriptions = "\n".join(_describe_option(opt) for opt in options)
    return (
        "You are an augmentation selector. Select which augmentations from the list below "
        "would be most appropriate for the passage. Choose only augmentations that are "
        "well-suited to the passage's content, structure, and length. "
        "You may select zero, one, or multiple augmentations.\n\n"
        "Available augmentations:\n"
        f"{aug_descriptions}"
    )


class MetaAugmentation(BaseSingleChunkAugmentation[MetaSelection]):
    schema = MetaSelection
    temperature = 0.1
    max_tokens = 1024

    def __init__(
        self,
        augmentations: list[AugmentationEntry],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._aug_map: dict[str, BaseSingleChunkAugmentation] = {}
        self._aug_options: list[AugmentationOption] = []
        for entry in augmentations:
            name, aug = entry[0], entry[1]
            hint = entry[2] if len(entry) > 2 else aug.selection_hint
            if not hint:
                raise ValueError(
                    f"Augmentation '{name}' has no selection_hint. Set the "
                    "selection_hint attribute on the augmentation (or pass it "
                    "as a third tuple element) so the smart switch knows when "
                    "to pick it."
                )
            self._aug_map[name] = aug
            self._aug_options.append(
                AugmentationOption(name=name, selection_hint=hint)
            )
        # Constrain `selected` to the actual augmentation names so the
        # selector cannot hallucinate a task that does not exist.
        self.schema = _build_selection_schema(tuple(self._aug_map))

    def build_messages(
        self,
        passages: str,
        response_format=None,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _build_selection_prompt(self._aug_options)},
            {"role": "user", "content": passages},
        ]

    def build_dataset(
        self,
        passages: str,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        validated = self.validate_passages(passages)
        selection = self.generate_one(validated, runtime)
        dataset: list[AlpacaDataset] = []
        for name in selection.selected:
            aug = self._aug_map.get(name)
            if aug is not None:
                dataset.extend(run_augmentation(validated, aug, runtime))
        return dataset

    async def abuild_dataset(
        self,
        passages: str,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        validated = self.validate_passages(passages)
        selection = await self.agenerate_one(validated, runtime)
        dataset: list[AlpacaDataset] = []
        for name in selection.selected:
            aug = self._aug_map.get(name)
            if aug is not None:
                dataset.extend(await arun_augmentation(validated, aug, runtime))
        return dataset


def apply_best_augmentations(
    passage: str,
    augmentations: list[AugmentationEntry],
    runtime: ModelRuntime,
    *,
    prefilter: bool = True,
    add_reasoning: bool = False,
) -> list[AlpacaDataset]:
    if prefilter and not prefilter_passage(passage, runtime):
        return []
    meta = MetaAugmentation(augmentations)
    dataset = run_augmentation(passage, meta, runtime)
    if add_reasoning:
        from text_albumentations.reasoning import add_reasoning_to_dataset
        dataset = add_reasoning_to_dataset(passage, dataset, runtime)
    return dataset


async def aapply_best_augmentations(
    passage: str,
    augmentations: list[AugmentationEntry],
    runtime: ModelRuntime,
    *,
    prefilter: bool = True,
    add_reasoning: bool = False,
) -> list[AlpacaDataset]:
    if prefilter and not await aprefilter_passage(passage, runtime):
        return []
    meta = MetaAugmentation(augmentations)
    dataset = await arun_augmentation(passage, meta, runtime)
    if add_reasoning:
        from text_albumentations.reasoning import aadd_reasoning_to_dataset
        dataset = await aadd_reasoning_to_dataset(passage, dataset, runtime)
    return dataset
