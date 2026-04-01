# Demonstrates batch decoding through the text_albumentations augmentation layer with one shared schema.

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_albumentations import OutlinesModel, run_batch_augmentation
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
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    print("loading_tokenizer")
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("building_outlines_model")
    model = outlines.from_transformers(hf_model, hf_tokenizer)
    runtime = OutlinesModel(model, max_tokens_parameter="max_new_tokens")
    augmentation = BulletAugmentation(max_tokens=128, variations=0)

    print("running_batch_augmentation")
    rows = run_batch_augmentation(PASSAGES, augmentation, runtime)

    print("mode=batch_augmentation")
    print("runtime=outlines_transformers")
    print(f"passages={len(PASSAGES)}")
    print(f"rows={len(rows)}")

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
