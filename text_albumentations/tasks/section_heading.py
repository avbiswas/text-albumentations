from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class SectionHeading(BaseModel):
    heading: str = Field(max_length=100)
    rationale: str = Field(max_length=300)


SYSTEM_PROMPT = """
Given this passage, write a concise section or subsection heading that would fit above it in a document. Also provide a short rationale grounded in the passage. Only generate answers.
    """

HEADING_INSTRUCTION = "Write a concise section heading for this passage."
RATIONALE_INSTRUCTION = "Explain briefly why this section heading fits the passage."


class SectionHeadingAdapter(BaseAlpacaAdapter[str, SectionHeading]):
    def convert(self, passages: str, output: SectionHeading) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=HEADING_INSTRUCTION,
                input=passages,
                output=output.heading,
            ),
            AlpacaDataset(
                instruction=RATIONALE_INSTRUCTION,
                input=f"Passage:\n{passages}\n\nHeading: {output.heading}",
                output=output.rationale,
            ),
        ]


class SectionHeadingAugmentation(BaseSingleChunkAugmentation[SectionHeading]):
    schema = SectionHeading
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage is a coherent paragraph or subsection that can be locally labeled."
    adapters = (SectionHeadingAdapter(),)
    temperature = 0.3
    instruction_templates = {
        HEADING_INSTRUCTION: (
            HEADING_INSTRUCTION,
            "Create a concise section heading for this passage.",
            "Write a short subsection heading that fits this passage.",
            "Provide a compact heading for this passage.",
            "Name this passage section with a concise heading.",
        ),
        RATIONALE_INSTRUCTION: (
            RATIONALE_INSTRUCTION,
            "Briefly justify why this heading fits the passage.",
            "Explain why the heading matches the passage content.",
            "Give a short passage-grounded rationale for the heading.",
            "State why this section heading is appropriate for the passage.",
        ),
    }


section_heading_augmentation = SectionHeadingAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        section_heading_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer replaces recurrent and convolutional sequence transduction models with attention mechanisms.
        """
    )
    print(len(dataset))
