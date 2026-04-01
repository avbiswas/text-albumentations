from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TypeVar

import outlines
from outlines.inputs import Chat
from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)

DEFAULT_MODEL_NAME = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
DEFAULT_OPENAI_CONCURRENCY = 10
_OPENAI_ASYNC_SEMAPHORE: asyncio.Semaphore | None = None
_OPENAI_ASYNC_SEMAPHORE_LIMIT: int | None = None


class ModelRuntime(ABC):
    @abstractmethod
    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def generate_variation(
        self,
        output: BaseModel | str,
        output_type: type[OutputT],
        *,
        context: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ) -> OutputT:
        raise NotImplementedError

    async def agenerate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        return self.generate_structured(
            messages,
            output_type,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def agenerate_variation(
        self,
        output: BaseModel | str,
        output_type: type[OutputT],
        *,
        context: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ) -> OutputT:
        return self.generate_variation(
            output,
            output_type,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class OutlinesModel(ModelRuntime):
    def __init__(
        self,
        model,
        *,
        async_mode: bool = False,
        total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
        max_tokens_parameter: str = "max_tokens",
    ) -> None:
        self.model = model
        self.async_mode = async_mode
        self.total_concurrent_calls = total_concurrent_calls
        self.max_tokens_parameter = max_tokens_parameter

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        if self.async_mode:
            raise RuntimeError(
                "This runtime is configured for async mode. "
                "Use 'await runtime.agenerate_structured(...)' instead."
            )

        output = self.model(Chat(messages), output_type, **self._build_generation_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
        ))
        return output_type.model_validate_json(output)

    def generate_variation(
        self,
        output: BaseModel | str,
        output_type: type[OutputT],
        *,
        context: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ) -> OutputT:
        if self.async_mode:
            raise RuntimeError(
                "This runtime is configured for async mode. "
                "Use 'await runtime.agenerate_variation(...)' instead."
            )

        serialized_output = (
            output.model_dump_json()
            if hasattr(output, "model_dump_json")
            else output
        )
        new_output = self.model(
            Chat(build_variation_messages(serialized_output, context)),
            output_type,
            **self._build_generation_kwargs(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return output_type.model_validate_json(new_output)

    async def agenerate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        if not self.async_mode:
            return self.generate_structured(messages, output_type, temperature=temperature, max_tokens=max_tokens)

        semaphore = get_openai_async_semaphore(self.total_concurrent_calls)
        async with semaphore:
            output = await self.model(
                Chat(messages),
                output_type,
                **self._build_generation_kwargs(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        return output_type.model_validate_json(output)

    async def agenerate_variation(
        self,
        output: BaseModel | str,
        output_type: type[OutputT],
        *,
        context: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ) -> OutputT:
        if not self.async_mode:
            return self.generate_variation(output, output_type, context=context, temperature=temperature, max_tokens=max_tokens)

        serialized_output = (
            output.model_dump_json()
            if hasattr(output, "model_dump_json")
            else output
        )
        semaphore = get_openai_async_semaphore(self.total_concurrent_calls)
        async with semaphore:
            response = await self.model(
                Chat(build_variation_messages(serialized_output, context)),
                output_type,
                **self._build_generation_kwargs(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        return output_type.model_validate_json(response)

    def _build_generation_kwargs(
        self,
        *,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "temperature": temperature,
            self.max_tokens_parameter: max_tokens,
        }
        if self.max_tokens_parameter == "max_tokens":
            kwargs["sampler"] = self._build_sampler(temperature)
            kwargs.pop("temperature")
        return kwargs

    def _build_sampler(self, temperature: float):
        import mlx_lm

        return mlx_lm.sample_utils.make_sampler(temp=temperature)


OutlinesModelRuntime = OutlinesModel


def get_openai_async_semaphore(
    total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
) -> asyncio.Semaphore:
    global _OPENAI_ASYNC_SEMAPHORE
    global _OPENAI_ASYNC_SEMAPHORE_LIMIT

    if total_concurrent_calls <= 0:
        raise ValueError("total_concurrent_calls must be greater than 0.")

    if (
        _OPENAI_ASYNC_SEMAPHORE is None
        or _OPENAI_ASYNC_SEMAPHORE_LIMIT != total_concurrent_calls
    ):
        _OPENAI_ASYNC_SEMAPHORE = asyncio.Semaphore(total_concurrent_calls)
        _OPENAI_ASYNC_SEMAPHORE_LIMIT = total_concurrent_calls

    return _OPENAI_ASYNC_SEMAPHORE


def build_variation_messages(
    output: str,
    context: str | None = None,
) -> list[dict[str, str]]:
    context_text = f"Additional context: {context}" if context else ""
    return [
        {
            "role": "system",
            "content": (
                "Generate augmentation of the same data structure, without "
                "changing the data itself. Do not lose any information about "
                f"the data. {context_text}. \n Generate text that is different "
                "that the original request"
            ),
        },
        {
            "role": "user",
            "content": output,
        },
    ]


def build_mlx_outlines_model(model_name: str = DEFAULT_MODEL_NAME):
    import mlx_lm

    return outlines.from_mlxlm(*mlx_lm.load(model_name))


@lru_cache(maxsize=4)
def get_default_outlines_runtime(
    model_name: str = DEFAULT_MODEL_NAME,
) -> OutlinesModel:
    return OutlinesModel(build_mlx_outlines_model(model_name))
