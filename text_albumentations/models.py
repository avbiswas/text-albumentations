"""Model primitives: the "where does generation run" abstraction.

Each class is ready to pass to ``ta.augment(text, model=...)`` or to any
lower-level API that accepts a model.
"""

from __future__ import annotations

import os
from typing import Literal

from text_albumentations.runtime import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OPENAI_CONCURRENCY,
    OutlinesModel,
    build_mlx_outlines_model,
)
from text_albumentations.vertex import VertexAIModel


class _JsonModeOpenAITypeAdapter:
    def __init__(self, base_adapter, response_format: dict[str, object]) -> None:
        self._base_adapter = base_adapter
        self._response_format = response_format

    def format_input(self, model_input):
        return self._base_adapter.format_input(model_input)

    def format_output_type(self, output_type=None) -> dict[str, object]:
        if output_type is None:
            return {}
        return {"response_format": self._response_format}


ResponseFormatMode = Literal["auto", "json_schema", "json_object"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")


MODEL_RESPONSE_FORMAT_CATALOG: tuple[tuple[str, ResponseFormatMode], ...] = (
    ("deepseek/", "json_object"),
    ("deepseek", "json_object"),
    ("openai/", "json_schema"),
    ("gpt", "json_schema"),
    ("google/", "json_schema"),
    ("gemini", "json_schema"),
    ("anthropic/", "json_object"),
    ("claude", "json_object"),
    ("minimax/", "json_object"),
    ("glm", "json_object"),
    ("z-ai/", "json_object"),
)


def resolve_response_format_mode(
    model: str,
    response_format: ResponseFormatMode = "auto",
) -> ResponseFormatMode:
    if response_format not in ("auto", "json_schema", "json_object"):
        raise ValueError(
            "response_format must be one of: 'auto', 'json_schema', 'json_object'."
        )

    if response_format != "auto":
        return response_format

    normalized_model = model.lower()
    for pattern, mode in MODEL_RESPONSE_FORMAT_CATALOG:
        if pattern in normalized_model:
            return mode

    return "json_object"


def build_response_format(mode: ResponseFormatMode) -> dict[str, object] | None:
    if mode == "json_schema":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    raise ValueError(f"Cannot build response format for unresolved mode '{mode}'.")


class OpenAIModel(OutlinesModel):
    """Any OpenAI-compatible endpoint: OpenAI, local MLX server, vLLM, Ollama, ...

    ``model``, ``base_url``, and ``api_key`` are required. Each falls back to
    an environment variable when not passed: ``TEXT_ALBUMENTATIONS_MODEL``,
    ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        async_mode: bool = False,
        total_concurrent_calls: int = DEFAULT_OPENAI_CONCURRENCY,
        response_format: ResponseFormatMode = "auto",
        reasoning_effort: ReasoningEffort | None = "low",
        completion_kwargs: dict[str, object] | None = None,
    ) -> None:
        import openai
        import outlines

        model = model or os.environ.get("TEXT_ALBUMENTATIONS_MODEL")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        missing = [
            f"{name} (or set {env_var})"
            for name, value, env_var in (
                ("model", model, "TEXT_ALBUMENTATIONS_MODEL"),
                ("base_url", base_url, "OPENAI_BASE_URL"),
                ("api_key", api_key, "OPENAI_API_KEY"),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "OpenAIModel needs an OpenAI-compatible endpoint. Missing: "
                + ", ".join(missing)
                + ". Example: OpenAIModel('gpt-5-mini', "
                "base_url='https://api.openai.com/v1', api_key='sk-...')."
            )
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_VALUES:
            raise ValueError(
                "reasoning_effort must be one of: "
                "'none', 'minimal', 'low', 'medium', 'high', 'xhigh', or None."
            )

        client = (
            openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            if async_mode
            else openai.OpenAI(base_url=base_url, api_key=api_key)
        )
        outlines_model = outlines.from_openai(client, model)
        self.response_format_mode = resolve_response_format_mode(model, response_format)
        self.response_format = build_response_format(self.response_format_mode)
        if self.response_format is not None:
            outlines_model.type_adapter = _JsonModeOpenAITypeAdapter(
                outlines_model.type_adapter,
                self.response_format,
            )

        generation_kwargs = {}
        if reasoning_effort is not None:
            generation_kwargs["reasoning_effort"] = reasoning_effort
        if completion_kwargs:
            generation_kwargs.update(completion_kwargs)

        super().__init__(
            outlines_model,
            async_mode=async_mode,
            total_concurrent_calls=total_concurrent_calls,
            max_tokens_parameter="max_completion_tokens",
            generation_kwargs=generation_kwargs,
        )
        self.model_name = model


class LocalMLXModel(OutlinesModel):
    """An MLX model loaded in-process via mlx-lm (Apple Silicon)."""

    def __init__(self, model: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__(build_mlx_outlines_model(model))
        self.model_name = model


class LocalHFModel(OutlinesModel):
    """A Hugging Face Transformers model loaded in-process."""

    def __init__(
        self,
        model: str,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(model)
        super().__init__(
            outlines.from_transformers(hf_model, hf_tokenizer),
            max_tokens_parameter="max_new_tokens",
        )
        self.model_name = model
