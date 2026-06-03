# Demonstrates sync OpenAI usage through the OpenAIModel primitive.

import text_albumentations as ta


MODEL_NAME = "gpt-5.4-nano"
PASSAGE = "The Transformer replaces recurrence with attention and improves parallelization."


def main():
    # api_key/base_url fall back to OPENAI_API_KEY / OPENAI_BASE_URL env vars
    model = ta.OpenAIModel(MODEL_NAME, base_url="https://api.openai.com/v1")

    print("model_type=OpenAIModel")
    print("mode=sync")

    rows = ta.augment(PASSAGE, tasks=["bullets"], model=model)

    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
