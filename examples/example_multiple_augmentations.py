# Demonstrates running multiple augmentations over the same passage and combining all generated rows.

import openai

from text_albumentations import create_openai_runtime, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.continuation import continuation_augmentation
from text_albumentations.tasks.qa_pairs import qa_pair_augmentation
from text_albumentations.tasks.rephrase import rephrase_augmentation
from text_albumentations.tasks.triplets import triplet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = """
The Transformer replaces recurrence with attention mechanisms.
This improves parallelization and leads to strong translation results.
It also generalizes to other sequence modeling tasks.
""".strip()

AUGMENTATIONS = [
    ("bullets", bullet_augmentation),
    ("qa_pairs", qa_pair_augmentation),
    ("rephrase", rephrase_augmentation),
    ("continuation", continuation_augmentation),
    ("triplets", triplet_augmentation),
]


def main():
    runtime = create_openai_runtime(openai.OpenAI(), MODEL_NAME, async_mode=False)

    all_rows = []
    for name, augmentation in AUGMENTATIONS:
        rows = run_augmentation(PASSAGE, augmentation, runtime)
        all_rows.extend(rows)
        print(f"augmentation={name} rows={len(rows)}")

    print(f"total_rows={len(all_rows)}")

    for row in all_rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
