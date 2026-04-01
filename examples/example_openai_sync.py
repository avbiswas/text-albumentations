# Demonstrates sync OpenAI usage through the runtime wrapper for a single augmentation.

import openai

from text_albumentations import create_openai_runtime, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."


def main():
    client = openai.OpenAI()
    runtime = create_openai_runtime(
        client,
        MODEL_NAME,
        async_mode=False,
    )

    print("client_type=OpenAI")
    print("runtime_mode=sync")

    rows = run_augmentation(PASSAGE, bullet_augmentation, runtime)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
