# Demonstrates batch decoding through the text_albumentations augmentation layer with one shared schema.

import text_albumentations as ta
from text_albumentations import run_batch_augmentation
from text_albumentations.tasks.bullets import BulletAugmentation


MODEL_NAME = "google/gemma-3-1b-it"
PASSAGES = [
    "The Transformer replaces recurrence with attention and improves parallelization.",
    "Outlines constrains generation so outputs match the expected structure.",
    "Synthetic supervision can be derived from raw documents with task-shaped prompts.",
    "Batch decoding is useful when many passages share the same schema and augmentation.",
]


def main():
    print("loading_transformers_model")
    model = ta.LocalHFModel(MODEL_NAME)
    augmentation = BulletAugmentation(max_tokens=128, variations=0)

    print("running_batch_augmentation")
    rows = run_batch_augmentation(PASSAGES, augmentation, model)

    print("mode=batch_augmentation")
    print("model=LocalHFModel")
    print(f"passages={len(PASSAGES)}")
    print(f"rows={len(rows)}")

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
