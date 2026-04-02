from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.response_formats import AlpacaResponseFormat
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class BulletList(BaseModel):
    bullets: list[str] = Field(..., min_length=1, max_length=6)


SYSTEM_PROMPT = """
Given this passage, extract a short list of important bullet points.
Keep each point concise and grounded in the passage.
Do not introduce information that is not present in the passage.
"""


def format_markdown_bullets(bullets: list[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in bullets)


def format_python_list(bullets: list[str]) -> str:
    return str(bullets)


class MarkdownBulletAdapter(BaseAlpacaAdapter[str, BulletList]):
    def convert(self, passages: str, output: BulletList) -> list[AlpacaDataset]:
        markdown_bullets = format_markdown_bullets(output.bullets)
        return [
            AlpacaDataset(
                instruction="Extract the important points from this passage as markdown bullet points.",
                input=passages,
                output=markdown_bullets,
            ),
            AlpacaDataset(
                instruction="Summarize this passage as markdown bullet points.",
                input=passages,
                output=markdown_bullets,
            ),
        ]


class PythonListBulletAdapter(BaseAlpacaAdapter[str, BulletList]):
    def convert(self, passages: str, output: BulletList) -> list[AlpacaDataset]:
        python_list_bullets = format_python_list(output.bullets)
        return [
            AlpacaDataset(
                instruction="Extract the important points from this passage as a Python list of strings.",
                input=passages,
                output=python_list_bullets,
            ),
            AlpacaDataset(
                instruction="Return a Python list of the key points from this passage.",
                input=passages,
                output=python_list_bullets,
            ),
        ]


class BulletAugmentation(BaseSingleChunkAugmentation[BulletList]):
    schema = BulletList
    system_prompt = SYSTEM_PROMPT
    response_formats = (
        AlpacaResponseFormat(
            name="markdown",
            adapter=MarkdownBulletAdapter(),
            format_instruction=(
                "Represent the extracted points as markdown bullet points."
            ),
        ),
        AlpacaResponseFormat(
            name="python_list",
            adapter=PythonListBulletAdapter(),
            format_instruction=(
                "Represent the extracted points as a Python list of strings."
            ),
        ),
    )
    temperature = 0.2
    variations = 1

    def __init__(
        self,
        *,
        max_bullets: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_bullets = max_bullets
        self._configured_schema: type[BulletList] | None = None

    def get_schema(self, passages: str | list[str] | None = None) -> type[BulletList]:
        if self.max_bullets == 6:
            return self.schema
        if self._configured_schema is None:
            self._configured_schema = create_model(
                "ConfiguredBulletList",
                bullets=(list[str], Field(..., min_length=1, max_length=self.max_bullets)),
                __base__=BaseModel,
            )
        return self._configured_schema


bullet_augmentation = BulletAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        bullet_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer replaces recurrence and convolutions with attention mechanisms.
It improves parallelization and achieves strong machine translation performance.
It also generalizes well to other tasks such as parsing.
        """
    )

    print(len(dataset))
