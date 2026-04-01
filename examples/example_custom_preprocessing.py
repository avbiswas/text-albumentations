# Demonstrates extending the pipeline with a custom preprocessed Pydantic input model.

from openai import OpenAI
from pydantic import BaseModel, Field

from text_albumentations import (
    AlpacaResponseFormat,
    BaseAugmentation,
    create_openai_runtime,
    run_augmentation,
)
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.tasks.bullets import BulletList
from text_albumentations.utils import AlpacaDataset


MODEL_NAME = "gpt-5.4-nano"


class PreprocessedPassage(BaseModel):
    source: str
    title: str
    cleaned_text: str
    keywords: list[str] = Field(default_factory=list)


def preprocess(raw_text: str, source: str = "demo") -> PreprocessedPassage:
    cleaned_text = " ".join(raw_text.split())
    title = cleaned_text.split(".")[0][:80].strip() or "Untitled"
    keywords = [word.strip(".,") for word in cleaned_text.split()[:3]]
    return PreprocessedPassage(
        source=source,
        title=title,
        cleaned_text=cleaned_text,
        keywords=keywords,
    )


class BulletMarkdownAdapter(BaseAlpacaAdapter[PreprocessedPassage, BulletList]):
    def convert(
        self,
        passages: PreprocessedPassage,
        output: BulletList,
    ) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction="Summarize this document as markdown bullet points.",
                input=(
                    f"Source: {passages.source}\n"
                    f"Title: {passages.title}\n\n"
                    f"{passages.cleaned_text}"
                ),
                output="\n".join(f"- {bullet}" for bullet in output.bullets),
            )
        ]


class CustomPreprocessedBulletAugmentation(
    BaseAugmentation[PreprocessedPassage, BulletList]
):
    schema = BulletList
    system_prompt = (
        "Given a preprocessed document, extract a short list of important bullet points. "
        "Keep each point concise and grounded in the document."
    )
    response_formats = (
        AlpacaResponseFormat(
            name="markdown",
            adapter=BulletMarkdownAdapter(),
            format_instruction="Represent the answer as markdown bullet points.",
        ),
    )

    def validate_passages(self, passages: PreprocessedPassage) -> PreprocessedPassage:
        if not isinstance(passages, PreprocessedPassage):
            raise TypeError("Expected a PreprocessedPassage instance.")
        return passages

    def build_user_message(self, passages: PreprocessedPassage) -> str:
        return (
            f"Source: {passages.source}\n"
            f"Title: {passages.title}\n"
            f"Keywords: {', '.join(passages.keywords)}\n\n"
            f"Document:\n{passages.cleaned_text}"
        )


def main():
    raw_text = """
    The Transformer replaces recurrence with attention mechanisms.
    This improves parallelization and leads to strong translation results.
    """

    preprocessed = preprocess(raw_text, source="custom-preprocessor")
    runtime = create_openai_runtime(OpenAI(), MODEL_NAME, async_mode=False)
    augmentation = CustomPreprocessedBulletAugmentation()

    rows = run_augmentation(preprocessed, augmentation, runtime)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
