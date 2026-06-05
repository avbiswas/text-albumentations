import re

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class ExtractiveQaItem(BaseModel):
    question: str = Field(max_length=300)
    supporting_quote: str = Field(max_length=500)


class ExtractiveQa(BaseModel):
    items: list[ExtractiveQaItem] = Field(min_length=1, max_length=5)


SYSTEM_PROMPT = """
Given this passage, generate questions that are each answered by one exact sentence or phrase from the passage. For every question, copy the supporting sentence or phrase verbatim from the passage, character for character, without paraphrasing. Only generate answers.
    """

INSTRUCTION = (
    "Answer this question by quoting the exact sentence or phrase from the passage "
    "that supports the answer."
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_in_passage(quote: str, passage: str) -> bool:
    return normalize_whitespace(quote).lower() in normalize_whitespace(passage).lower()


class ExtractiveQaAdapter(BaseAlpacaAdapter[str, ExtractiveQa]):
    def convert(self, passages: str, output: ExtractiveQa) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=INSTRUCTION,
                input=f"Passage:\n{passages}\n\nQuestion: {item.question}",
                output=item.supporting_quote,
            )
            for item in output.items
            if quote_in_passage(item.supporting_quote, passages)
        ]


class ExtractiveQaAugmentation(BaseSingleChunkAugmentation[ExtractiveQa]):
    schema = ExtractiveQa
    system_prompt = SYSTEM_PROMPT
    selection_hint = "specific facts in the passage can be quoted verbatim as answers."
    adapters = (ExtractiveQaAdapter(),)
    temperature = 0.3
    instruction_templates = {
        INSTRUCTION: (
            INSTRUCTION,
            "Use an exact quote from the passage to answer this question.",
            "Answer by copying the supporting sentence or phrase from the passage.",
            "Quote the exact passage text that answers the question.",
            "Answer with the exact sentence or phrase from the passage.",
            "Find the supporting quote in the passage and use it as the answer.",
            "Respond by quoting only the passage text that supports the answer.",
            "Copy the exact evidence from the passage that answers the question.",
        ),
    }


extractive_qa_augmentation = ExtractiveQaAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        extractive_qa_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
