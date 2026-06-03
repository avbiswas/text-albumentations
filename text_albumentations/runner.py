from __future__ import annotations

from dataclasses import dataclass
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

    def run(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        dataset = self.augmentation.build_dataset(validated_data, self.runtime)
        if self.add_reasoning:
            from text_albumentations.reasoning import add_reasoning_to_dataset
            dataset = add_reasoning_to_dataset(validated_data, dataset, self.runtime)
        return dataset

    async def arun(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        dataset = await self.augmentation.abuild_dataset(validated_data, self.runtime)
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
) -> list[AlpacaDataset]:
    return AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
        add_reasoning=add_reasoning,
    ).run()


async def arun_augmentation(
    data: PassageT,
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
    *,
    add_reasoning: bool = False,
) -> list[AlpacaDataset]:
    return await AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
        add_reasoning=add_reasoning,
    ).arun()


def run_batch_augmentation(
    data_batch: list[PassageT],
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    validated_batch = [
        augmentation.validate_passages(data)
        for data in data_batch
    ]
    return augmentation.build_batch_dataset(validated_batch, runtime)


async def arun_batch_augmentation(
    data_batch: list[PassageT],
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    validated_batch = [
        augmentation.validate_passages(data)
        for data in data_batch
    ]
    return await augmentation.abuild_batch_dataset(validated_batch, runtime)
