# Demonstrates async OpenAI usage with bounded concurrency through the OpenAIModel primitive.

import asyncio

import text_albumentations as ta
from text_albumentations import arun_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."
TOTAL_CONCURRENT_CALLS = 4


async def main():
    model = ta.OpenAIModel(
        MODEL_NAME,
        base_url="https://api.openai.com/v1",
        async_mode=True,
        total_concurrent_calls=TOTAL_CONCURRENT_CALLS,
    )

    print("model_type=OpenAIModel")
    print("mode=async")
    print(f"total_concurrent_calls={TOTAL_CONCURRENT_CALLS}")

    rows = await arun_augmentation(PASSAGE, bullet_augmentation, model)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
