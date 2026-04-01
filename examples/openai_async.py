# Demonstrates async OpenAI usage with bounded concurrency when the user creates the Outlines OpenAI model outside the library.

import asyncio

import openai
import outlines

from text_albumentations import (
    OutlinesModel,
    arun_augmentation,
)
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."
TOTAL_CONCURRENT_CALLS = 4


async def main():
    client = openai.AsyncOpenAI()
    model = outlines.from_openai(client, MODEL_NAME)
    runtime = OutlinesModel(
        model,
        async_mode=True,
        total_concurrent_calls=TOTAL_CONCURRENT_CALLS,
        max_tokens_parameter="max_completion_tokens",
    )

    print("client_type=AsyncOpenAI")
    print("runtime_mode=async")
    print(f"total_concurrent_calls={TOTAL_CONCURRENT_CALLS}")

    rows = await arun_augmentation(PASSAGE, bullet_augmentation, runtime)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
