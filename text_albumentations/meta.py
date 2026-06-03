from __future__ import annotations

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.runner import arun_augmentation, run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset


class AugmentationOption(BaseModel):
    name: str
    description: str


class MetaSelection(BaseModel):
    is_low_quality: bool = Field(
        ...,
        description="True if the passage is too short, nonsensical, or contains no meaningful content",
    )
    low_quality_reason: str = Field(
        ...,
        description="Explanation of why the passage is low quality, if applicable",
    )
    selected: list[str] = Field(
        ...,
        description="Names of augmentations that would work well for this passage",
    )
    reasoning: str = Field(
        ...,
        description="Brief reasoning for why these augmentations were selected",
    )


def _build_selection_prompt(options: list[AugmentationOption]) -> str:
    aug_descriptions = "\n".join(
        f"- {opt.name}: {opt.description}" for opt in options
    )
    return (
        "You are a data quality evaluator and augmentation selector.\n\n"
        "Step 1: Assess whether the passage is low quality. A passage is low quality if it is:\n"
        "- Too short (fewer than roughly 10 meaningful words)\n"
        "- Nonsensical or random character sequences\n"
        "- Only contains formatting, markup, or code with no natural language content\n"
        "- Contains only metadata, headers, or boilerplate with no substantive content\n\n"
        "Step 2: If the passage passes the quality check, select which augmentations "
        "from the list below would be most appropriate. Choose only augmentations "
        "that are well-suited to the passage's content, structure, and length. "
        "You may select zero, one, or multiple augmentations.\n\n"
        "Available augmentations:\n"
        f"{aug_descriptions}\n\n"
        "Return your assessment."
    )


class MetaAugmentation(BaseSingleChunkAugmentation[MetaSelection]):
    schema = MetaSelection
    temperature = 0.1
    max_tokens = 1024

    def __init__(
        self,
        augmentations: list[tuple[str, BaseSingleChunkAugmentation, str]],
        *,
        enable_quality_filter: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._aug_map: dict[str, BaseSingleChunkAugmentation] = {
            name: aug for name, aug, _ in augmentations
        }
        self._aug_options = [
            AugmentationOption(name=name, description=desc)
            for name, _, desc in augmentations
        ]
        self.enable_quality_filter = enable_quality_filter

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

        if self.enable_quality_filter and selection.is_low_quality:
            return []

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

        if self.enable_quality_filter and selection.is_low_quality:
            return []

        dataset: list[AlpacaDataset] = []
        for name in selection.selected:
            aug = self._aug_map.get(name)
            if aug is not None:
                dataset.extend(await arun_augmentation(validated, aug, runtime))
        return dataset


def apply_best_augmentations(
    passage: str,
    augmentations: list[tuple[str, BaseSingleChunkAugmentation, str]],
    runtime: ModelRuntime,
    *,
    enable_quality_filter: bool = True,
    add_reasoning: bool = False,
) -> list[AlpacaDataset]:
    meta = MetaAugmentation(
        augmentations,
        enable_quality_filter=enable_quality_filter,
    )
    dataset = run_augmentation(passage, meta, runtime)
    if add_reasoning:
        from text_albumentations.reasoning import add_reasoning_to_dataset
        dataset = add_reasoning_to_dataset(passage, dataset, runtime)
    return dataset


async def aapply_best_augmentations(
    passage: str,
    augmentations: list[tuple[str, BaseSingleChunkAugmentation, str]],
    runtime: ModelRuntime,
    *,
    enable_quality_filter: bool = True,
    add_reasoning: bool = False,
) -> list[AlpacaDataset]:
    meta = MetaAugmentation(
        augmentations,
        enable_quality_filter=enable_quality_filter,
    )
    dataset = await arun_augmentation(passage, meta, runtime)
    if add_reasoning:
        from text_albumentations.reasoning import aadd_reasoning_to_dataset
        dataset = await aadd_reasoning_to_dataset(passage, dataset, runtime)
    return dataset
