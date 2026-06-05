from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TypeVar

import outlines
from outlines.inputs import Chat
from pydantic import BaseModel
from pydantic_core import ValidationError

OutputT = TypeVar("OutputT", bound=BaseModel)

DEFAULT_MODEL_NAME = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
DEFAULT_OPENAI_CONCURRENCY = 100
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

    def generate_structured_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> list[OutputT]:
        return [
            self.generate_structured(
                messages,
                output_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for messages in messages_batch
        ]

    async def agenerate_structured_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> list[OutputT]:
        return [
            await self.agenerate_structured(
                messages,
                output_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for messages in messages_batch
        ]


class OutlinesModel(ModelRuntime):
    def __init__(
        self,
        model,
        *,
        async_mode: bool = False,
        total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
        max_tokens_parameter: str = "max_tokens",
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.async_mode = async_mode
        self.total_concurrent_calls = total_concurrent_calls
        self.max_tokens_parameter = max_tokens_parameter
        self.generation_kwargs = generation_kwargs or {}

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

        output = self.model(Chat(self._prepare_messages(messages, output_type)), output_type, **self._build_generation_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
        ))
        return validate_json_output(output, output_type)

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
            Chat(self._prepare_messages(
                build_variation_messages(serialized_output, context),
                output_type,
            )),
            output_type,
            **self._build_generation_kwargs(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return validate_json_output(new_output, output_type)

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
                Chat(self._prepare_messages(messages, output_type)),
                output_type,
                **self._build_generation_kwargs(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        return validate_json_output(output, output_type)

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
                Chat(self._prepare_messages(
                    build_variation_messages(serialized_output, context),
                    output_type,
                )),
                output_type,
                **self._build_generation_kwargs(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        return validate_json_output(response, output_type)

    def generate_structured_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> list[OutputT]:
        if self.async_mode:
            raise RuntimeError(
                "This runtime is configured for async mode. "
                "Use 'await runtime.agenerate_structured_batch(...)' instead."
            )

        outputs = self.model.batch(
            [
                Chat(self._prepare_messages(messages, output_type))
                for messages in messages_batch
            ],
            output_type=output_type,
            **self._build_generation_kwargs(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return [validate_json_output(output, output_type) for output in outputs]

    async def agenerate_structured_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> list[OutputT]:
        if not self.async_mode:
            return self.generate_structured_batch(
                messages_batch,
                output_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        semaphore = get_openai_async_semaphore(self.total_concurrent_calls)
        async with semaphore:
            outputs = await self.model.batch(
                [
                    Chat(self._prepare_messages(messages, output_type))
                    for messages in messages_batch
                ],
                output_type=output_type,
                **self._build_generation_kwargs(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        return [validate_json_output(output, output_type) for output in outputs]

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
        kwargs.update(self.generation_kwargs)
        if self.max_tokens_parameter == "max_tokens":
            kwargs["sampler"] = self._build_sampler(temperature)
            kwargs.pop("temperature")
        return kwargs

    def _prepare_messages(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
    ) -> list[dict[str, str]]:
        schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
        schema_instruction = (
            "The desired schema you must answer in is the following JSON Schema. "
            "Return only JSON that conforms to it. Do not include markdown, "
            f"prose, or extra keys.\nJSON Schema:\n{schema}"
        )
        prepared = [dict(message) for message in messages]
        system_indexes = [
            index
            for index, message in enumerate(prepared)
            if message.get("role") == "system"
        ]
        if not system_indexes:
            prepared.insert(0, {"role": "system", "content": schema_instruction})
            return prepared

        system_index = system_indexes[-1]
        prepared[system_index]["content"] = (
            f"{prepared[system_index].get('content', '').rstrip()}\n\n"
            f"{schema_instruction}"
        )
        return prepared

    def _build_sampler(self, temperature: float):
        import mlx_lm

        return mlx_lm.sample_utils.make_sampler(temp=temperature)


OutlinesModelRuntime = OutlinesModel


def validate_json_output(output: str, output_type: type[OutputT]) -> OutputT:
    try:
        return output_type.model_validate_json(output)
    except ValidationError:
        if not isinstance(output, str):
            raise
        return output_type.model_validate_json(extract_json_object(output))


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text


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
