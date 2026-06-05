from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI


DEFAULT_PROMPT = (
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. "
    "What does the ball cost? Briefly answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reasoning_effort settings and reported token usage."
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-5-nano", "deepseek/deepseek-v4-flash"],
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=["low", "high"],
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument(
        "--openrouter-reasoning-object",
        action="store_true",
        help=(
            "Send OpenRouter's unified extra_body reasoning object instead of "
            "top-level OpenAI-style reasoning_effort."
        ),
    )
    return parser.parse_args()


def usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": (
            getattr(details, "reasoning_tokens", None) if details else None
        ),
    }


def main() -> None:
    args = parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
    )
    messages = [
        {"role": "system", "content": "Return a concise final answer only."},
        {"role": "user", "content": args.prompt},
    ]

    for model in args.models:
        for effort in args.efforts:
            request_kwargs: dict[str, Any]
            if args.openrouter_reasoning_object:
                request_kwargs = {"extra_body": {"reasoning": {"effort": effort}}}
            else:
                request_kwargs = {"reasoning_effort": effort}

            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=args.max_completion_tokens,
                    **request_kwargs,
                )
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "model": model,
                            "effort": effort,
                            "ok": False,
                            "error": repr(error),
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            message = response.choices[0].message
            print(
                json.dumps(
                    {
                        "model": model,
                        "effort": effort,
                        "ok": True,
                        "elapsed_s": round(time.perf_counter() - started, 2),
                        **usage_dict(response.usage),
                        "has_reasoning_text": bool(getattr(message, "reasoning", None)),
                        "content": message.content,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
