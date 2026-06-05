from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset, estimate_max_length_from_words

DEFAULT_STYLES: dict[str, str] = {
    "eli5": "plain, simple language that a curious twelve-year-old could understand",
    "formal": "formal, polished prose suitable for a professional report",
    "casual": "casual, conversational language as if explaining to a friend",
}


class StyleRewrite(BaseModel):
    rewritten: str = Field(max_length=2000)


def build_system_prompt(style_description: str) -> str:
    return (
        f"Given this passage, rewrite it in {style_description}. "
        "Preserve all factual content from the passage, do not introduce new concepts "
        "outside the scope of this input. Do not re-quote the original. Only generate answers."
    )


class StyleTransferAdapter(BaseAlpacaAdapter[str, StyleRewrite]):
    def __init__(self, style_description: str) -> None:
        self.style_description = style_description

    def convert(self, passages: str, output: StyleRewrite) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=f"Rewrite this passage in {self.style_description}.",
                input=passages,
                output=output.rewritten,
            )
        ]


class StyleTransferAugmentation(BaseSingleChunkAugmentation[StyleRewrite]):
    schema = StyleRewrite
    selection_hint = "the passage's register could plausibly be rewritten in a different style."
    temperature = 0.5

    def __init__(
        self,
        *,
        style: str = "eli5",
        style_description: str | None = None,
        max_rewritten_length: int | None = None,
        rewritten_length_multiplier: float = 3.0,
        **kwargs,
    ) -> None:
        if style_description is None:
            if style not in DEFAULT_STYLES:
                raise ValueError(
                    f"Unknown style '{style}'. Use one of {sorted(DEFAULT_STYLES)} "
                    "or pass style_description explicitly."
                )
            style_description = DEFAULT_STYLES[style]

        self.style = style
        self.style_description = style_description
        self.system_prompt = build_system_prompt(style_description)
        instruction = f"Rewrite this passage in {style_description}."
        self.instruction_templates = {
            instruction: (
                instruction,
                f"Recast this passage in {style_description}.",
                f"Rewrite the passage using {style_description}.",
                f"Express this passage in {style_description}.",
                f"Transform the passage into {style_description}.",
                f"Restyle this passage using {style_description}.",
                f"Write this passage again in {style_description}.",
                f"Convert the passage to {style_description}.",
            ),
        }
        kwargs.setdefault("adapters", (StyleTransferAdapter(style_description),))
        super().__init__(**kwargs)
        self.max_rewritten_length = max_rewritten_length
        self.rewritten_length_multiplier = rewritten_length_multiplier
        self._configured_schema: type[StyleRewrite] | None = None
        self._configured_schema_key: int | None = None

    def get_schema(self, passages: str | list[str] | None = None) -> type[StyleRewrite]:
        max_rewritten_length = self.max_rewritten_length
        if max_rewritten_length is None:
            max_rewritten_length = estimate_max_length_from_words(
                passages,
                self.rewritten_length_multiplier,
                minimum=2000,
            )
        if max_rewritten_length == 2000:
            return self.schema
        if self._configured_schema is None or self._configured_schema_key != max_rewritten_length:
            self._configured_schema = create_model(
                "ConfiguredStyleRewrite",
                rewritten=(str, Field(max_length=max_rewritten_length)),
                __base__=BaseModel,
            )
            self._configured_schema_key = max_rewritten_length
        return self._configured_schema


style_transfer_augmentation = StyleTransferAugmentation()
eli5_style_augmentation = StyleTransferAugmentation(style="eli5")
formal_style_augmentation = StyleTransferAugmentation(style="formal")
casual_style_augmentation = StyleTransferAugmentation(style="casual")


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        style_transfer_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
