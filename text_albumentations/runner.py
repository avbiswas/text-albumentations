from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Generic, TypeVar

from text_albumentations.base import BaseAugmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset

PassageT = TypeVar("PassageT")


@dataclass
class AugmentationRunner(Generic[PassageT]):
    data: PassageT
    runtime: ModelRuntime
    augmentation: BaseAugmentation
    add_reasoning: bool = False
    sample_instruction_template: bool = True
    instruction_rng: random.Random | None = None

    def run(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        dataset = self.augmentation.build_dataset(validated_data, self.runtime)
        dataset = sample_instruction_templates(
            dataset,
            self.augmentation.instruction_templates,
            enabled=self.sample_instruction_template,
            rng=self.instruction_rng,
        )
        if self.add_reasoning:
            from text_albumentations.reasoning import add_reasoning_to_dataset
            dataset = add_reasoning_to_dataset(validated_data, dataset, self.runtime)
        return dataset

    async def arun(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        dataset = await self.augmentation.abuild_dataset(validated_data, self.runtime)
        dataset = sample_instruction_templates(
            dataset,
            self.augmentation.instruction_templates,
            enabled=self.sample_instruction_template,
            rng=self.instruction_rng,
        )
        if self.add_reasoning:
            from text_albumentations.reasoning import aadd_reasoning_to_dataset
            dataset = await aadd_reasoning_to_dataset(validated_data, dataset, self.runtime)
        return dataset


def run_augmentation(
    data: PassageT,
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
    *,
    add_reasoning: bool = False,
    sample_instruction_template: bool = True,
    instruction_rng: random.Random | None = None,
) -> list[AlpacaDataset]:
    return AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
        add_reasoning=add_reasoning,
        sample_instruction_template=sample_instruction_template,
        instruction_rng=instruction_rng,
    ).run()


async def arun_augmentation(
    data: PassageT,
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
    *,
    add_reasoning: bool = False,
    sample_instruction_template: bool = True,
    instruction_rng: random.Random | None = None,
) -> list[AlpacaDataset]:
    return await AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
        add_reasoning=add_reasoning,
        sample_instruction_template=sample_instruction_template,
        instruction_rng=instruction_rng,
    ).arun()


def sample_instruction_templates(
    dataset: list[AlpacaDataset],
    instruction_templates: dict[str, Sequence[str]],
    *,
    enabled: bool,
    rng: random.Random | None = None,
) -> list[AlpacaDataset]:
    if not enabled or not instruction_templates:
        return dataset

    rng = rng or random
    sampled = []
    for row in dataset:
        templates = instruction_templates.get(row.instruction)
        if not templates:
            sampled.append(row)
            continue
        sampled.append(
            row.model_copy(
                update={"instruction": rng.choice(list(templates))}
            )
        )
    return sampled


def run_batch_augmentation(
    data_batch: list[PassageT],
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
    *,
    sample_instruction_template: bool = True,
    instruction_rng: random.Random | None = None,
) -> list[AlpacaDataset]:
    validated_batch = [
        augmentation.validate_passages(data)
        for data in data_batch
    ]
    dataset = augmentation.build_batch_dataset(validated_batch, runtime)
    return sample_instruction_templates(
        dataset,
        augmentation.instruction_templates,
        enabled=sample_instruction_template,
        rng=instruction_rng,
    )


async def arun_batch_augmentation(
    data_batch: list[PassageT],
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
    *,
    sample_instruction_template: bool = True,
    instruction_rng: random.Random | None = None,
) -> list[AlpacaDataset]:
    validated_batch = [
        augmentation.validate_passages(data)
        for data in data_batch
    ]
    dataset = await augmentation.abuild_batch_dataset(validated_batch, runtime)
    return sample_instruction_templates(
        dataset,
        augmentation.instruction_templates,
        enabled=sample_instruction_template,
        rng=instruction_rng,
    )
