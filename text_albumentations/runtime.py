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
    def __init__(self, model) -> None:
        self.model = model

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        import mlx_lm

        chat_messages = Chat(messages)
        sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)

        output = self.model(
            chat_messages,
            output_type,
            max_tokens=max_tokens,
            sampler=sampler,
        )
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
        import mlx_lm

        context_text = f"Additional context: {context}" if context else ""

        if hasattr(output, "model_dump_json"):
            output = output.model_dump_json()

        messages = Chat(
            [
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
        )

        sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
        new_output = self.model(
            messages,
            output_type,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return output_type.model_validate_json(new_output)


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


class OpenAIOutlinesModel(ModelRuntime):
    def __init__(
        self,
        client,
        model_name: str,
        *,
        async_mode: bool = False,
        total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.async_mode = async_mode
        self.total_concurrent_calls = total_concurrent_calls
        self.model = outlines.from_openai(client, model_name)

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
                "This OpenAI runtime is configured for async mode. "
                "Use 'await runtime.agenerate_structured(...)' instead."
            )

        output = self.model(
            Chat(messages),
            output_type,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
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
                "This OpenAI runtime is configured for async mode. "
                "Use 'await runtime.agenerate_variation(...)' instead."
            )

        serialized_output = (
            output.model_dump_json()
            if hasattr(output, "model_dump_json")
            else output
        )
        response = self.model(
            Chat(build_variation_messages(serialized_output, context)),
            output_type,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return output_type.model_validate_json(response)

    async def agenerate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        if not self.async_mode:
            return self.generate_structured(
                messages,
                output_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        semaphore = get_openai_async_semaphore(self.total_concurrent_calls)
        async with semaphore:
            output = await self.model(
                Chat(messages),
                output_type,
                temperature=temperature,
                max_completion_tokens=max_tokens,
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
            return self.generate_variation(
                output,
                output_type,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )

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
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        return output_type.model_validate_json(response)


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


def create_outlines_runtime(model) -> OutlinesModel:
    return OutlinesModel(model=model)


def create_openai_runtime(
    client,
    model_name: str,
    *,
    async_mode: bool = False,
    total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
) -> OpenAIOutlinesModel:
    return OpenAIOutlinesModel(
        client=client,
        model_name=model_name,
        async_mode=async_mode,
        total_concurrent_calls=total_concurrent_calls,
    )


@lru_cache(maxsize=4)
def get_default_outlines_runtime(
    model_name: str = DEFAULT_MODEL_NAME,
) -> OutlinesModel:
    return create_outlines_runtime(build_mlx_outlines_model(model_name))
