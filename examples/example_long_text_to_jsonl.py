# Demonstrates chunking a long string, running an augmentation per chunk, and saving Alpaca rows to JSONL.

import openai
import outlines

from text_albumentations import OutlinesModel, save_long_text_dataset
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "gpt-5.4-nano"
OUTPUT_JSONL = "long_text_bullets.jsonl"
LONG_TEXT = """
The Transformer replaces recurrence with attention mechanisms.
This improves parallelization and leads to strong translation results.
It also generalizes to other sequence modeling tasks.

Convolutional networks apply local filters and are effective for vision tasks.
Recurrent networks process tokens sequentially, which can limit parallelism.
Attention-based models can compare tokens directly across the sequence.
""" * 10


def main():
    model = outlines.from_openai(openai.OpenAI(), MODEL_NAME)
    runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")
    rows = save_long_text_dataset(
        LONG_TEXT,
        OUTPUT_JSONL,
        bullet_augmentation,
        runtime,
        chunk_size_chars=300,
        overlap_chars=0,
    )
    print(f"rows_saved={len(rows)}")
    print(f"output_jsonl={OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
