from __future__ import annotations

import asyncio
import os
from typing import TypeVar

from pydantic import BaseModel

from text_albumentations.runtime import (
    DEFAULT_OPENAI_CONCURRENCY,
    ModelRuntime,
    build_variation_messages,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class VertexAIModel(ModelRuntime):

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        project: str | None = None,
        location: str | None = None,
        async_mode: bool = False,
        total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
    ) -> None:
        from google import genai

        self.model_name = model
        self.async_mode = async_mode
        self.total_concurrent_calls = total_concurrent_calls
        self._semaphore: asyncio.Semaphore | None = None

        project = project or os.environ.get(
            "GOOGLE_CLOUD_PROJECT",
            os.environ.get("CLOUDSDK_CORE_PROJECT"),
        )
        location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.total_concurrent_calls)
        return self._semaphore

    def _build_contents(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict]]:
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return system_instruction, contents

    def _generate(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        from google.genai import types

        system_instruction, contents = self._build_contents(messages)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=output_type,
            ),
        )

        text = response.text
        if not text:
            raise RuntimeError(
                f"Gemini returned an empty response (model={self.model_name}). "
                "This may be due to safety filters or an unsupported schema. "
                f"Finish reason: {getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no candidates'}."
            )
        return output_type.model_validate_json(text)

    async def _agenerate(
        self,
        messages: list[dict[str, str]],
        output_type: type[OutputT],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> OutputT:
        from google.genai import types

        system_instruction, contents = self._build_contents(messages)

        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=output_type,
            ),
        )

        text = response.text
        if not text:
            raise RuntimeError(
                f"Gemini returned an empty response (model={self.model_name}). "
                "This may be due to safety filters or an unsupported schema. "
                f"Finish reason: {getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no candidates'}."
            )
        return output_type.model_validate_json(text)

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
        return self._generate(messages, output_type, temperature=temperature, max_tokens=max_tokens)

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
        serialized = (
            output.model_dump_json() if hasattr(output, "model_dump_json") else output
        )
        messages = build_variation_messages(serialized, context)
        return self._generate(messages, output_type, temperature=temperature, max_tokens=max_tokens)

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
                messages, output_type, temperature=temperature, max_tokens=max_tokens
            )
        sem = self._get_semaphore()
        async with sem:
            return await self._agenerate(
                messages, output_type, temperature=temperature, max_tokens=max_tokens
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
        if not self.async_mode:
            return self.generate_variation(
                output, output_type, context=context, temperature=temperature, max_tokens=max_tokens
            )
        serialized = (
            output.model_dump_json() if hasattr(output, "model_dump_json") else output
        )
        messages = build_variation_messages(serialized, context)
        sem = self._get_semaphore()
        async with sem:
            return await self._agenerate(
                messages, output_type, temperature=temperature, max_tokens=max_tokens
            )

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
        return [
            self._generate(msgs, output_type, temperature=temperature, max_tokens=max_tokens)
            for msgs in messages_batch
        ]

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
                messages_batch, output_type, temperature=temperature, max_tokens=max_tokens
            )
        sem = self._get_semaphore()
        tasks = []
        for msgs in messages_batch:
            async def _call(m=msgs):
                async with sem:
                    return await self._agenerate(
                        m, output_type, temperature=temperature, max_tokens=max_tokens
                    )
            tasks.append(_call())
        return await asyncio.gather(*tasks)
