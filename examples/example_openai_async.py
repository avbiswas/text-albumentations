# Demonstrates async OpenAI usage with bounded concurrency through the runtime wrapper.

import asyncio

import openai

from text_albumentations import arun_augmentation, create_openai_runtime
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."
TOTAL_CONCURRENT_CALLS = 4


async def main():
    client = openai.AsyncOpenAI()
    runtime = create_openai_runtime(
        client,
        MODEL_NAME,
        async_mode=True,
        total_concurrent_calls=TOTAL_CONCURRENT_CALLS,
    )

    print("client_type=AsyncOpenAI")
    print("runtime_mode=async")
    print(f"total_concurrent_calls={TOTAL_CONCURRENT_CALLS}")

    rows = await arun_augmentation(PASSAGE, bullet_augmentation, runtime)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
