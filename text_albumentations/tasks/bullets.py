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
    selection_hint = "the passage contains several distinct facts or points worth itemizing."
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
    instruction_templates = {
        "Extract the important points from this passage as markdown bullet points.": (
            "Extract the important points from this passage as markdown bullet points.",
            "List the key points from this passage as markdown bullets.",
            "Pull out the main points from this passage as markdown bullet points.",
            "Return the passage's important points as markdown bullets.",
            "Write markdown bullet points for the key ideas in this passage.",
            "Use markdown bullets to extract the important points from this passage.",
            "Create a markdown bullet list of this passage's main points.",
        ),
        "Summarize this passage as markdown bullet points.": (
            "Summarize this passage as markdown bullet points.",
            "Write a markdown bullet-point summary of this passage.",
            "Condense this passage into markdown bullet points.",
            "Summarize the passage using markdown bullets.",
            "Create a markdown bullet summary of the passage.",
            "Return a concise markdown bullet list summarizing this passage.",
            "Write the key summary points as markdown bullets.",
        ),
        "Extract the important points from this passage as a Python list of strings.": (
            "Extract the important points from this passage as a Python list of strings.",
            "Return the key points from this passage as a Python list of strings.",
            "List the main points from this passage as a Python list of strings.",
            "Output the passage's important points as a Python list of strings.",
            "Create a Python list of strings containing the key points from this passage.",
            "Represent the important points as a Python list of strings.",
            "Use a Python list of strings for the passage's main points.",
        ),
        "Return a Python list of the key points from this passage.": (
            "Return a Python list of the key points from this passage.",
            "Provide the passage's key points as a Python list.",
            "Output a Python list containing the main points of this passage.",
            "Write the passage's key points as a Python list of strings.",
            "Create a Python list with the important points from this passage.",
            "Return the main ideas from this passage in Python list format.",
            "Use Python list syntax to list the passage's key points.",
        ),
    }

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
