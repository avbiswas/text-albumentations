# Demonstrates direct Outlines Transformers batch decoding with Qwen/Qwen3.5-2B, without the text_albumentations wrapper.

from typing import Literal

import outlines
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-2B"

SENTIMENT_PROMPTS = [
    "Classify the sentiment of: I loved this movie.",
    "Classify the sentiment of: This was awful.",
    "Classify the sentiment of: Amazing product.",
    "Classify the sentiment of: I want a refund.",
]

COUNTRY_PROMPTS = [
    "Answer as JSON. Paris is in which country?",
    "Answer as JSON. Berlin is in which country?",
    "Answer as JSON. Rome is in which country?",
    "Answer as JSON. Madrid is in which country?",
]


class CountryAnswer(BaseModel):
    country: str


def main():
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = outlines.from_transformers(hf_model, hf_tokenizer)

    literal_outputs = model.batch(
        SENTIMENT_PROMPTS,
        output_type=Literal["positive", "negative"],
        max_new_tokens=8,
    )

    basemodel_outputs = model.batch(
        COUNTRY_PROMPTS,
        output_type=CountryAnswer,
        max_new_tokens=32,
    )

    print(f"model_name={MODEL_NAME}")
    print(f"literal_outputs={literal_outputs}")
    print(f"basemodel_outputs={basemodel_outputs}")


if __name__ == "__main__":
    main()
