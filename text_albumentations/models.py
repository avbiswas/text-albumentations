"""Model primitives: the "where does generation run" abstraction.

Each class is ready to pass to ``ta.augment(text, model=...)`` or to any
lower-level API that accepts a model.
"""

from __future__ import annotations

import os

from text_albumentations.runtime import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OPENAI_CONCURRENCY,
    OutlinesModel,
    build_mlx_outlines_model,
)


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

        client = (
            openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            if async_mode
            else openai.OpenAI(base_url=base_url, api_key=api_key)
        )
        super().__init__(
            outlines.from_openai(client, model),
            async_mode=async_mode,
            total_concurrent_calls=total_concurrent_calls,
            max_tokens_parameter="max_completion_tokens",
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
