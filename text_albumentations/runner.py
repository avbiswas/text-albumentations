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

    def run(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        return self.augmentation.build_dataset(validated_data, self.runtime)

    async def arun(self) -> list[AlpacaDataset]:
        validated_data = self.augmentation.validate_passages(self.data)
        return await self.augmentation.abuild_dataset(validated_data, self.runtime)


def run_augmentation(
    data: PassageT,
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    return AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
    ).run()


async def arun_augmentation(
    data: PassageT,
    augmentation: BaseAugmentation,
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    return await AugmentationRunner(
        data=data,
        runtime=runtime,
        augmentation=augmentation,
    ).arun()
