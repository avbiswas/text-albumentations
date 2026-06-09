"""Run retrieval augmentation over externally managed passage groups.

Set OPENROUTER_API_KEY and run:

    python examples/retrieval_external_groups.py

The important point is that corpus grouping stays outside text-albumentations:
you decide which passages belong together, then call run_augmentation on each
group.
"""

from __future__ import annotations

import os

import text_albumentations as ta
from text_albumentations.tasks.retrieval import RetrievalAugmentation


PASSAGE_GROUPS = [
    [
        (
            "The Transformer is a sequence transduction architecture based "
            "solely on attention mechanisms. It avoids recurrence and "
            "convolutions while improving parallelization."
        ),
        (
            "Convolutional neural networks apply learned filters over local "
            "neighborhoods and are commonly used for computer vision tasks."
        ),
    ],
    [
        (
            "Retrieval-augmented generation combines a retriever with a "
            "generator so answers can be grounded in external documents."
        ),
        (
            "Instruction tuning trains models on instruction-response pairs "
            "so they learn to follow user requests more reliably."
        ),
    ],
]


def build_model() -> ta.OpenAIModel:
    return ta.OpenAIModel(
        "deepseek/deepseek-v4-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        reasoning_effort=None,
    )


def main() -> None:
    model = build_model()
    retrieval = RetrievalAugmentation(
        max_questions_per_passage=1,
        max_passages=2,
        include_negative_examples=False,
    )

    rows = []
    for group in PASSAGE_GROUPS:
        rows.extend(ta.run_augmentation(group, retrieval, model))

    print(f"generated_rows={len(rows)}")
    for row in rows[:2]:
        print("---")
        print(row.instruction)
        print(row.input[:240].replace("\n", " "))
        print(row.output)


if __name__ == "__main__":
    main()
