from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from text_albumentations.utils import AlpacaDataset

PassageT = TypeVar("PassageT")
OutputT = TypeVar("OutputT")


class BaseAlpacaAdapter(ABC, Generic[PassageT, OutputT]):
    @abstractmethod
    def convert(self, passages: PassageT, output: OutputT) -> list[AlpacaDataset]:
        raise NotImplementedError
