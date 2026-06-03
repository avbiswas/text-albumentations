# Demonstrates the smallest local workflow for running one augmentation on one passage.

import text_albumentations as ta


MODEL_NAME = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."


def main():
    model = ta.LocalMLXModel(MODEL_NAME)

    rows = ta.augment(PASSAGE, tasks=["bullets"], model=model)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
