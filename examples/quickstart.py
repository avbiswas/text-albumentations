# The one-import entry point: ta.augment picks tasks, runs them, and saves rows.

import text_albumentations as ta

PASSAGE = (
    "The Transformer replaces recurrence with attention and improves "
    "parallelization, achieving state-of-the-art BLEU scores on WMT 2014 "
    "translation tasks at a fraction of the training cost."
)


def main():
    print("Available tasks:", list(ta.list_tasks()))

    # Explicit tasks against a local OpenAI-compatible server.
    model = ta.OpenAIModel(
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        base_url="http://localhost:8080/v1",
        api_key="local",
    )
    rows = ta.augment(PASSAGE, tasks=["summarize", "title"], model=model)

    for row in rows:
        print(row.model_dump_json())

    ta.save(rows, "quickstart_rows.jsonl")


if __name__ == "__main__":
    main()
