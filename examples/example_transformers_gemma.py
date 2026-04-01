# Demonstrates using a user-created Outlines Transformers model with google/gemma-3-1b-it.

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "google/gemma-3-1b-it"
PASSAGE = """
The Transformer replaces recurrence with attention mechanisms.
This improves parallelization and leads to strong translation results.
It also generalizes to other sequence modeling tasks.
""".strip()


def main():
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = outlines.from_transformers(hf_model, hf_tokenizer)
    runtime = OutlinesModel(model, max_tokens_parameter="max_new_tokens")

    rows = run_augmentation(PASSAGE, bullet_augmentation, runtime)

    print(f"model_backend=transformers")
    print(f"model_name={MODEL_NAME}")
    print(f"rows={len(rows)}")

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
