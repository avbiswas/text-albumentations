from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from text_albumentations.output_format_adapters.alpaca import BaseAlpacaAdapter
from text_albumentations.utils import AlpacaDataset

PassageT = TypeVar("PassageT")
OutputT = TypeVar("OutputT")


class BaseResponseFormat(ABC, Generic[PassageT, OutputT]):
    name: str
    format_instruction: str = ""

    def build_system_prompt(self, base_system_prompt: str) -> str:
        if not self.format_instruction.strip():
            return base_system_prompt

        return (
            f"{base_system_prompt.strip()}\n\n"
            "Expected response format for the Alpaca output field:\n"
            f"{self.format_instruction.strip()}"
        )

    @abstractmethod
    def convert(self, passages: PassageT, output: OutputT) -> list[AlpacaDataset]:
        raise NotImplementedError


@dataclass(frozen=True)
class AlpacaResponseFormat(BaseResponseFormat[PassageT, OutputT]):
    name: str
    adapter: BaseAlpacaAdapter[PassageT, OutputT]
    format_instruction: str = ""

    def convert(self, passages: PassageT, output: OutputT) -> list[AlpacaDataset]:
        return self.adapter.convert(passages, output)
