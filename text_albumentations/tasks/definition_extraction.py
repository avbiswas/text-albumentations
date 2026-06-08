from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.tasks.extractive_qa import quote_in_passage
from text_albumentations.utils import AlpacaDataset


class DefinitionItem(BaseModel):
    term: str = Field(max_length=120)
    definition: str = Field(max_length=500)
    supporting_quote: str = Field(max_length=500)


class Definitions(BaseModel):
    definitions: list[DefinitionItem] = Field(min_length=1, max_length=6)


SYSTEM_PROMPT = """
Given this passage, extract terms that the passage defines or clearly explains. For each term, write a concise definition based only on the passage and copy a supporting quote verbatim from the passage. Do not include terms whose meaning is not explained. Only generate answers.
    """

DEFINE_INSTRUCTION = "Define the term using only the information in the passage."
QUOTE_INSTRUCTION = "Quote the passage text that supports this definition."


class DefinitionExtractionAdapter(BaseAlpacaAdapter[str, Definitions]):
    def convert(self, passages: str, output: Definitions) -> list[AlpacaDataset]:
        rows = []
        for item in output.definitions:
            if not quote_in_passage(item.supporting_quote, passages):
                continue
            term_input = f"Passage:\n{passages}\n\nTerm: {item.term}"
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=DEFINE_INSTRUCTION,
                        input=term_input,
                        output=item.definition,
                    ),
                    AlpacaDataset(
                        instruction=QUOTE_INSTRUCTION,
                        input=term_input,
                        output=item.supporting_quote,
                    ),
                ]
            )
        return rows


class DefinitionExtractionAugmentation(BaseSingleChunkAugmentation[Definitions]):
    schema = Definitions
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage defines or explains technical terms, concepts, acronyms, or named methods."
    adapters = (DefinitionExtractionAdapter(),)
    temperature = 0.2
    instruction_templates = {
        DEFINE_INSTRUCTION: (
            DEFINE_INSTRUCTION,
            "Define this term based only on the passage.",
            "Use only the passage to explain what this term means.",
            "Write a passage-grounded definition for the term.",
            "Provide the term's definition using only information in the passage.",
        ),
        QUOTE_INSTRUCTION: (
            QUOTE_INSTRUCTION,
            "Copy the exact passage quote that supports this definition.",
            "Quote the exact evidence from the passage for this definition.",
            "Return the verbatim supporting quote for the definition.",
            "Provide only the passage quote that supports the term definition.",
        ),
    }


definition_extraction_augmentation = DefinitionExtractionAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        definition_extraction_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is a network architecture based solely on attention mechanisms.
        """
    )
    print(len(dataset))
