from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class TitleHeadline(BaseModel):
    title: str = Field(max_length=120)
    headline: str = Field(max_length=200)


SYSTEM_PROMPT = """
Given this passage, generate a short descriptive title of fewer than 12 words and a one-sentence headline of fewer than 25 words that captures its main point. Only present content that is strictly present in the passage, do not introduce new concepts outside the scope of this input. Only generate answers.
    """

TITLE_INSTRUCTION = "Write a short descriptive title for this passage."
HEADLINE_INSTRUCTION = "Write a one-sentence headline capturing the main point of this passage."


class TitleAdapter(BaseAlpacaAdapter[str, TitleHeadline]):
    def convert(self, passages: str, output: TitleHeadline) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=TITLE_INSTRUCTION,
                input=passages,
                output=output.title,
            ),
            AlpacaDataset(
                instruction=HEADLINE_INSTRUCTION,
                input=passages,
                output=output.headline,
            ),
        ]


class TitleAugmentation(BaseSingleChunkAugmentation[TitleHeadline]):
    schema = TitleHeadline
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage has one coherent main subject to name."
    adapters = (TitleAdapter(),)
    temperature = 0.4


title_augmentation = TitleAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        title_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
