# Demonstrates chunking a long string, running an augmentation per chunk, and saving Alpaca rows to JSONL.

import text_albumentations as ta
from text_albumentations import save_long_text_dataset
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
    model = ta.OpenAIModel(MODEL_NAME, base_url="https://api.openai.com/v1")
    rows = save_long_text_dataset(
        LONG_TEXT,
        OUTPUT_JSONL,
        bullet_augmentation,
        model,
        chunk_size_chars=300,
        overlap_chars=0,
    )
    print(f"rows_saved={len(rows)}")
    print(f"output_jsonl={OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
