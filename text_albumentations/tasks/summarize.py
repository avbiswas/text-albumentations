from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset, estimate_max_length_from_words


class Summary(BaseModel):
    tldr: str = Field(max_length=200)
    summary: str = Field(max_length=1000)


SYSTEM_PROMPT = """
Given this passage, summarize it. Produce a one-sentence TLDR of fewer than 25 words and a short prose summary covering the main points. Only present content that is strictly present in the passage, do not introduce new concepts outside the scope of this input. Do not re-quote the original. Only generate answers.
    """

TLDR_INSTRUCTION = "Summarize this passage in a single sentence."
SUMMARY_INSTRUCTION = "Write a concise summary of this passage covering its main points."


class SummarizeAdapter(BaseAlpacaAdapter[str, Summary]):
    def convert(self, passages: str, output: Summary) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=TLDR_INSTRUCTION,
                input=passages,
                output=output.tldr,
            ),
            AlpacaDataset(
                instruction=SUMMARY_INSTRUCTION,
                input=passages,
                output=output.summary,
            ),
        ]


class SummarizeAugmentation(BaseSingleChunkAugmentation[Summary]):
    schema = Summary
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage is longer than a couple of sentences and has compressible content."
    adapters = (SummarizeAdapter(),)
    temperature = 0.3
    instruction_templates = {
        TLDR_INSTRUCTION: (
            TLDR_INSTRUCTION,
            "Write a one-sentence TLDR for this passage.",
            "Condense this passage into one sentence.",
            "Summarize the passage in one sentence.",
            "Give a single-sentence summary of this passage.",
            "Write a brief one-line summary of the passage.",
            "Capture the main point of this passage in one sentence.",
        ),
        SUMMARY_INSTRUCTION: (
            SUMMARY_INSTRUCTION,
            "Summarize the main points of this passage concisely.",
            "Write a brief summary covering the key ideas in this passage.",
            "Give a concise overview of the passage's main points.",
            "Summarize the important ideas from this passage.",
            "Write a short summary of the passage.",
            "Describe the passage's main content in a concise summary.",
        ),
    }

    def __init__(
        self,
        *,
        max_summary_length: int | None = None,
        summary_length_multiplier: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_summary_length = max_summary_length
        self.summary_length_multiplier = summary_length_multiplier
        self._configured_schema: type[Summary] | None = None
        self._configured_schema_key: int | None = None

    def get_schema(self, passages: str | list[str] | None = None) -> type[Summary]:
        max_summary_length = self.max_summary_length
        if max_summary_length is None:
            max_summary_length = estimate_max_length_from_words(
                passages,
                self.summary_length_multiplier,
                minimum=1000,
            )
        if max_summary_length == 1000:
            return self.schema
        if self._configured_schema is None or self._configured_schema_key != max_summary_length:
            self._configured_schema = create_model(
                "ConfiguredSummary",
                tldr=(str, Field(max_length=200)),
                summary=(str, Field(max_length=max_summary_length)),
                __base__=BaseModel,
            )
            self._configured_schema_key = max_summary_length
        return self._configured_schema


summarize_augmentation = SummarizeAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        summarize_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
