from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class BacktranslatedInstruction(BaseModel):
    instruction: str = Field(max_length=400)


SYSTEM_PROMPT = """
Given this passage, write the instruction or prompt that this passage would be the ideal answer to. The instruction should be self-contained, specific to the passage's content, and phrased as a request a user would plausibly write. Do not quote the passage. Only generate the instruction.
    """


class BacktranslationAdapter(BaseAlpacaAdapter[str, BacktranslatedInstruction]):
    def convert(
        self,
        passages: str,
        output: BacktranslatedInstruction,
    ) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=output.instruction,
                input="",
                output=passages,
            )
        ]


class BacktranslationAugmentation(BaseSingleChunkAugmentation[BacktranslatedInstruction]):
    schema = BacktranslatedInstruction
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage reads like a complete answer to some plausible request."
    adapters = (BacktranslationAdapter(),)
    temperature = 0.5
    max_tokens = 1024


backtranslation_augmentation = BacktranslationAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        backtranslation_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
