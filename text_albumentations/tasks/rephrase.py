from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class Rewritten(BaseModel):
    rephrased: str = Field(max_length=1000)


SYSTEM_PROMPT = """
Given this passage, rephrase it. Elaborate on the sentences by explaining the meaning. Only present content that is strictly present in the passage, do not introduce new concepts outside the scope of this input. Do not re-quote the original. Only generate answers.
    """


class RephraseAdapter(BaseAlpacaAdapter[str, Rewritten]):
    def convert(self, passages: str, output: Rewritten) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=SYSTEM_PROMPT,
                input=passages,
                output=output.rephrased,
            )
        ]


class RephraseAugmentation(BaseSingleChunkAugmentation[Rewritten]):
    schema = Rewritten
    system_prompt = SYSTEM_PROMPT
    adapters = (RephraseAdapter(),)
    temperature = 0.5

    def __init__(
        self,
        *,
        max_rephrased_length: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_rephrased_length = max_rephrased_length
        self._configured_schema: type[Rewritten] | None = None

    def get_schema(self) -> type[Rewritten]:
        if self.max_rephrased_length == 1000:
            return self.schema
        if self._configured_schema is None:
            self._configured_schema = create_model(
                "ConfiguredRewritten",
                rephrased=(str, Field(max_length=self.max_rephrased_length)),
                __base__=BaseModel,
            )
        return self._configured_schema


rephrase_augmentation = RephraseAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        rephrase_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. 

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
             """
    )

    print(len(dataset))
