# Demonstrates the smallest local Outlines workflow for running one augmentation on one passage.

import mlx_lm
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


MODEL_NAME = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."


def main():
    model = outlines.from_mlxlm(*mlx_lm.load(MODEL_NAME))
    runtime = OutlinesModel(model=model)

    rows = run_augmentation(PASSAGE, bullet_augmentation, runtime)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
