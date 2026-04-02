from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from pydantic import BaseModel

from text_albumentations.output_format_adapters.alpaca import BaseAlpacaAdapter
from text_albumentations.response_formats import BaseResponseFormat
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset

PassageT = TypeVar("PassageT")
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAugmentation(ABC, Generic[PassageT, OutputT]):
    schema: type[OutputT]
    system_prompt: str
    adapters: Sequence[BaseAlpacaAdapter[PassageT, OutputT]] = ()
    response_formats: Sequence[BaseResponseFormat[PassageT, OutputT]] = ()
    temperature: float = 0.2
    variation_temperature: float = 0.5
    max_tokens: int = 5000
    variation_max_tokens: int = 5000
    num_generations: int = 1
    variations: int = 0
    variation_context: str | None = None

    def __init__(
        self,
        *,
        adapters: Sequence[BaseAlpacaAdapter[PassageT, OutputT]] | None = None,
        response_formats: Sequence[BaseResponseFormat[PassageT, OutputT]] | None = None,
        temperature: float | None = None,
        variation_temperature: float | None = None,
        max_tokens: int | None = None,
        variation_max_tokens: int | None = None,
        num_generations: int | None = None,
        variations: int | None = None,
        variation_context: str | None = None,
    ) -> None:
        if adapters is not None:
            self.adapters = adapters
        if response_formats is not None:
            self.response_formats = response_formats
        if temperature is not None:
            self.temperature = temperature
        if variation_temperature is not None:
            self.variation_temperature = variation_temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if variation_max_tokens is not None:
            self.variation_max_tokens = variation_max_tokens
        if num_generations is not None:
            self.num_generations = num_generations
        if variations is not None:
            self.variations = variations
        if variation_context is not None:
            self.variation_context = variation_context

    @abstractmethod
    def build_user_message(self, passages: PassageT) -> str:
        raise NotImplementedError

    def get_schema(self, passages: PassageT | list[PassageT] | None = None) -> type[OutputT]:
        return self.schema

    def build_messages(
        self,
        passages: PassageT,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = self.system_prompt
        if response_format is not None:
            system_prompt = response_format.build_system_prompt(system_prompt)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.build_user_message(passages)},
        ]

    def build_messages_batch(
        self,
        passages_batch: list[PassageT],
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> list[list[dict[str, str]]]:
        return [
            self.build_messages(passages, response_format)
            for passages in passages_batch
        ]

    def validate_passages(self, passages: PassageT) -> PassageT:
        return passages

    def generate_one(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> OutputT:
        schema = self.get_schema(passages)
        return runtime.generate_structured(
            self.build_messages(passages, response_format),
            schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def agenerate_one(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> OutputT:
        schema = self.get_schema(passages)
        return await runtime.agenerate_structured(
            self.build_messages(passages, response_format),
            schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate_outputs(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> list[OutputT]:
        outputs = []
        schema = self.get_schema(passages)

        for _ in range(self.num_generations):
            base_output = self.generate_one(passages, runtime, response_format)
            outputs.append(base_output)

            for _ in range(self.variations):
                outputs.append(
                    runtime.generate_variation(
                        base_output,
                        schema,
                        context=self.variation_context,
                        temperature=self.variation_temperature,
                        max_tokens=self.variation_max_tokens,
                    )
                )

        return outputs

    async def agenerate_outputs(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> list[OutputT]:
        schema = self.get_schema(passages)

        async def generate_output_chain() -> list[OutputT]:
            chain_outputs = []
            base_output = await self.agenerate_one(passages, runtime, response_format)
            chain_outputs.append(base_output)

            for _ in range(self.variations):
                chain_outputs.append(
                    await runtime.agenerate_variation(
                        base_output,
                        schema,
                        context=self.variation_context,
                        temperature=self.variation_temperature,
                        max_tokens=self.variation_max_tokens,
                    )
                )

            return chain_outputs

        generated_chains = await asyncio.gather(
            *[generate_output_chain() for _ in range(self.num_generations)]
        )
        return [
            output
            for generated_chain in generated_chains
            for output in generated_chain
        ]

    def build_dataset_from_output(
        self,
        passages: PassageT,
        output: OutputT,
        response_format: BaseResponseFormat[PassageT, OutputT] | None = None,
    ) -> list[AlpacaDataset]:
        if response_format is not None:
            return response_format.convert(passages, output)

        dataset = []
        for adapter in self.adapters:
            dataset.extend(adapter.convert(passages, output))
        return dataset

    def build_dataset(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        response_formats: list[BaseResponseFormat[PassageT, OutputT] | None]
        if self.response_formats:
            response_formats = list(self.response_formats)
        else:
            response_formats = [None]

        for response_format in response_formats:
            for output in self.generate_outputs(passages, runtime, response_format):
                dataset.extend(
                    self.build_dataset_from_output(
                        passages,
                        output,
                        response_format,
                    )
                )

        return dataset

    def build_batch_dataset(
        self,
        passages_batch: list[PassageT],
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        schema = self.get_schema(passages_batch)
        response_formats: list[BaseResponseFormat[PassageT, OutputT] | None]
        if self.response_formats:
            response_formats = list(self.response_formats)
        else:
            response_formats = [None]

        for response_format in response_formats:
            outputs = runtime.generate_structured_batch(
                self.build_messages_batch(passages_batch, response_format),
                schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            dataset.extend(
                self._build_batch_rows_from_outputs(
                    passages_batch,
                    outputs,
                    response_format,
                    runtime,
                )
            )

        return dataset

    async def abuild_dataset(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        response_formats: list[BaseResponseFormat[PassageT, OutputT] | None]
        if self.response_formats:
            response_formats = list(self.response_formats)
        else:
            response_formats = [None]

        format_datasets = await asyncio.gather(
            *[
                self._abuild_dataset_for_response_format(
                    passages,
                    runtime,
                    response_format,
                )
                for response_format in response_formats
            ]
        )

        for format_dataset in format_datasets:
            dataset.extend(format_dataset)

        return dataset

    async def abuild_batch_dataset(
        self,
        passages_batch: list[PassageT],
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        response_formats: list[BaseResponseFormat[PassageT, OutputT] | None]
        if self.response_formats:
            response_formats = list(self.response_formats)
        else:
            response_formats = [None]

        for response_format in response_formats:
            outputs = await runtime.agenerate_structured_batch(
                self.build_messages_batch(passages_batch, response_format),
                schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            dataset.extend(
                await self._abuild_batch_rows_from_outputs(
                    passages_batch,
                    outputs,
                    response_format,
                    runtime,
                )
            )

        return dataset

    async def _abuild_dataset_for_response_format(
        self,
        passages: PassageT,
        runtime: ModelRuntime,
        response_format: BaseResponseFormat[PassageT, OutputT] | None,
    ) -> list[AlpacaDataset]:
        dataset = []
        outputs = await self.agenerate_outputs(passages, runtime, response_format)

        for output in outputs:
            dataset.extend(
                self.build_dataset_from_output(
                    passages,
                    output,
                    response_format,
                )
            )

        return dataset

    def _build_batch_rows_from_outputs(
        self,
        passages_batch: list[PassageT],
        outputs: list[OutputT],
        response_format: BaseResponseFormat[PassageT, OutputT] | None,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        schema = self.get_schema(passages_batch)

        for passages, output in zip(passages_batch, outputs):
            output_chain = [output]
            for _ in range(self.variations):
                output_chain.append(
                    runtime.generate_variation(
                        output,
                        schema,
                        context=self.variation_context,
                        temperature=self.variation_temperature,
                        max_tokens=self.variation_max_tokens,
                    )
                )

            for chained_output in output_chain:
                dataset.extend(
                    self.build_dataset_from_output(
                        passages,
                        chained_output,
                        response_format,
                    )
                )

        return dataset

    async def _abuild_batch_rows_from_outputs(
        self,
        passages_batch: list[PassageT],
        outputs: list[OutputT],
        response_format: BaseResponseFormat[PassageT, OutputT] | None,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        dataset = []
        schema = self.get_schema(passages_batch)

        for passages, output in zip(passages_batch, outputs):
            output_chain = [output]
            for _ in range(self.variations):
                output_chain.append(
                    await runtime.agenerate_variation(
                        output,
                        schema,
                        context=self.variation_context,
                        temperature=self.variation_temperature,
                        max_tokens=self.variation_max_tokens,
                    )
                )

            for chained_output in output_chain:
                dataset.extend(
                    self.build_dataset_from_output(
                        passages,
                        chained_output,
                        response_format,
                    )
                )

        return dataset


class BaseSingleChunkAugmentation(BaseAugmentation[str, OutputT], ABC):
    def validate_passages(self, passages: str) -> str:
        if not isinstance(passages, str):
            raise TypeError("Single chunk augmentations expect one string passage.")

        passage = passages.strip()
        if not passage:
            raise ValueError("Single chunk augmentation received an empty passage.")
        return passage

    def build_user_message(self, passages: str) -> str:
        return passages


class BaseMultiChunkAugmentation(BaseAugmentation[list[str], OutputT], ABC):
    min_passages: int = 2

    def validate_passages(self, passages: list[str]) -> list[str]:
        if not isinstance(passages, list):
            raise TypeError("Multi chunk augmentations expect a list of passages.")

        cleaned_passages = [passage.strip() for passage in passages if passage.strip()]
        if len(cleaned_passages) < self.min_passages:
            raise ValueError(
                f"Multi chunk augmentation requires at least {self.min_passages} passages."
            )
        return cleaned_passages
