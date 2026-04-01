# Demonstrates sync OpenAI usage when the user creates the Outlines OpenAI model outside the library.

import openai
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."


def main():
    client = openai.OpenAI()
    model = outlines.from_openai(client, MODEL_NAME)
    runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

    print("client_type=OpenAI")
    print("runtime_mode=sync")

    rows = run_augmentation(PASSAGE, bullet_augmentation, runtime)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
