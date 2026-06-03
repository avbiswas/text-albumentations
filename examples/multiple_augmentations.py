# Demonstrates running multiple augmentations over the same passage and combining all generated rows.

import text_albumentations as ta


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = """
LLMs are powerful but their outputs are unpredictable. Most solutions attempt to fix bad outputs after generation using parsing, regex, or fragile code that breaks easily.

Outlines guarantees structured outputs during generation — directly from any LLM.

Works with any model - Same code runs across OpenAI, Ollama, vLLM, and more
Simple integration - Just pass your desired output type: model(prompt, output_type)
Guaranteed valid structure - No more parsing headaches or broken JSON
Provider independence - Switch models without changing code
""".strip()

TASKS = ["bullets", "qa_pairs", "rephrase", "continuation", "triplets"]


def main():
    model = ta.OpenAIModel(MODEL_NAME, base_url="https://api.openai.com/v1")

    all_rows = ta.augment(PASSAGE, tasks=TASKS, model=model)
    print(f"total_rows={len(all_rows)}")

    print()
    for row in all_rows:
        print(f"Instructions: \n{row.instruction}\n")
        print(f"Input: \n{row.input}\n")
        print(f"Output: \n{row.output}\n")
        print("\n\n--------\n\n")


if __name__ == "__main__":
    main()
