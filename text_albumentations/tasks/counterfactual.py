from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class Counterfactual(BaseModel):
    original_claim: str = Field(max_length=300)
    altered_premise: str = Field(max_length=300)
    question: str = Field(max_length=300)
    consequence: str = Field(max_length=1000)


SYSTEM_PROMPT = """
Given this passage, pick one central factual claim. Construct a counterfactual premise by altering that claim, then pose a self-contained question that restates the altered premise and asks what would follow if it were true, and answer it reasoning only from information in the passage. Do not introduce outside knowledge. Only generate answers.
    """


class CounterfactualAdapter(BaseAlpacaAdapter[str, Counterfactual]):
    def convert(self, passages: str, output: Counterfactual) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=(
                    f"Counterfactual premise: {output.altered_premise}\n\n{output.question}"
                ),
                input=passages,
                output=output.consequence,
            )
        ]


class CounterfactualAugmentation(BaseSingleChunkAugmentation[Counterfactual]):
    schema = Counterfactual
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage makes factual claims whose alteration has reasoned consequences."
    adapters = (CounterfactualAdapter(),)
    temperature = 0.7


counterfactual_augmentation = CounterfactualAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        counterfactual_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
