from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseMultiChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class Comparisons(BaseModel):
    answer: str = Field(max_length=500)


SYSTEM_PROMPT = """
Given 2 passages of text, generate a detailed comparison of the two
    """


class ComparisonAdapter(BaseAlpacaAdapter[list[str], Comparisons]):
    def convert(self, passages: list[str], output: Comparisons) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=SYSTEM_PROMPT,
                input=build_comparison_input(passages),
                output=output.answer,
            )
        ]


def build_comparison_input(passages: list[str]) -> str:
    if len(passages) != 2:
        raise ValueError("Comparison augmentation expects exactly two passages.")

    return (
        f"Passage 1:\n{passages[0]}\n\n"
        f"Passage 2:\n{passages[1]}"
    )


class ComparisonAugmentation(BaseMultiChunkAugmentation[Comparisons]):
    schema = Comparisons
    system_prompt = SYSTEM_PROMPT
    adapters = (ComparisonAdapter(),)
    temperature = 0.5
    num_generations = 3
    min_passages = 2

    def __init__(
        self,
        *,
        max_answer_length: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_answer_length = max_answer_length
        self._configured_schema: type[Comparisons] | None = None

    def get_schema(self) -> type[Comparisons]:
        if self.max_answer_length == 500:
            return self.schema
        if self._configured_schema is None:
            self._configured_schema = create_model(
                "ConfiguredComparisons",
                answer=(str, Field(max_length=self.max_answer_length)),
                __base__=BaseModel,
            )
        return self._configured_schema

    def validate_passages(self, passages: list[str]) -> list[str]:
        cleaned_passages = super().validate_passages(passages)
        if len(cleaned_passages) != 2:
            raise ValueError("Comparison augmentation requires exactly two passages.")
        return cleaned_passages

    def build_user_message(self, passages: list[str]) -> str:
        return build_comparison_input(passages)


comparison_augmentation = ComparisonAugmentation()


def main(passages: str | list[str]) -> list[AlpacaDataset]:
    if isinstance(passages, str):
        split_passages = [
            segment.strip()
            for segment in passages.split("\n\nPassage 2:\n", maxsplit=1)
        ]
        if len(split_passages) == 2 and split_passages[0].startswith("Passage 1:\n"):
            split_passages[0] = split_passages[0].removeprefix("Passage 1:\n").strip()
            return run_augmentation(
                split_passages,
                comparison_augmentation,
                get_default_outlines_runtime(),
            )

        raise ValueError("Comparison string input must contain both Passage 1 and Passage 2.")

    return run_augmentation(
        passages,
        comparison_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        [
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        ]
    )

    print(len(dataset))
