# Demonstrates running multiple augmentations over the same passage and combining all generated rows.

import openai
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.continuation import continuation_augmentation
from text_albumentations.tasks.qa_pairs import qa_pair_augmentation
from text_albumentations.tasks.rephrase import rephrase_augmentation
from text_albumentations.tasks.triplets import triplet_augmentation


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = """
LLMs are powerful but their outputs are unpredictable. Most solutions attempt to fix bad outputs after generation using parsing, regex, or fragile code that breaks easily.

Outlines guarantees structured outputs during generation — directly from any LLM.

Works with any model - Same code runs across OpenAI, Ollama, vLLM, and more
Simple integration - Just pass your desired output type: model(prompt, output_type)
Guaranteed valid structure - No more parsing headaches or broken JSON
Provider independence - Switch models without changing code
""".strip()

AUGMENTATIONS = [
    ("bullets", bullet_augmentation),
    ("qa_pairs", qa_pair_augmentation),
    ("rephrase", rephrase_augmentation),
    ("continuation", continuation_augmentation),
    ("triplets", triplet_augmentation),
]


def main():
    model = outlines.from_openai(openai.OpenAI(), MODEL_NAME)
    runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

    all_rows = []
    for name, augmentation in AUGMENTATIONS:
        rows = run_augmentation(PASSAGE, augmentation, runtime)
        all_rows.extend(rows)
        print(f"augmentation={name} rows={len(rows)}")

    print(f"total_rows={len(all_rows)}")

    print()
    for row in all_rows:
        print(f"Instructions: \n{row.instruction}\n")
        print(f"Input: \n{row.input}\n")
        print(f"Output: \n{row.output}\n")
        print("\n\n--------\n\n")

if __name__ == "__main__":
    main()
