from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.tasks.extractive_qa import quote_in_passage
from text_albumentations.utils import AlpacaDataset


class EvidenceSelectionItem(BaseModel):
    claim: str = Field(max_length=300)
    candidate_quotes: list[str] = Field(min_length=2, max_length=5)
    supporting_quote: str = Field(max_length=500)
    rationale: str = Field(max_length=400)


class EvidenceSelections(BaseModel):
    items: list[EvidenceSelectionItem] = Field(min_length=1, max_length=4)


SYSTEM_PROMPT = """
Given this passage, create evidence-selection examples. For each example, write a short claim supported by the passage, provide 2-5 candidate quotes copied verbatim from the passage, identify the one candidate quote that directly supports the claim, and explain why that quote supports it. Do not invent quotes. Only generate answers.
    """

INSTRUCTION = (
    "Choose the candidate quote that best supports the claim. "
    "Answer with the exact quote."
)
RATIONALE_INSTRUCTION = "Explain why the selected quote supports the claim."
QUOTE_SUPPORT_INSTRUCTION = (
    "Determine whether the quote supports the claim. "
    "Answer yes or no."
)


def _format_candidates(candidates: list[str]) -> str:
    return "\n".join(
        f"{idx}. {candidate}"
        for idx, candidate in enumerate(candidates, start=1)
    )


class EvidenceSelectionAdapter(BaseAlpacaAdapter[str, EvidenceSelections]):
    def convert(self, passages: str, output: EvidenceSelections) -> list[AlpacaDataset]:
        rows = []
        for item in output.items:
            if item.supporting_quote not in item.candidate_quotes:
                continue
            if not quote_in_passage(item.supporting_quote, passages):
                continue
            selection_input = (
                f"Passage:\n{passages}\n\n"
                f"Claim: {item.claim}\n\n"
                f"Candidate quotes:\n{_format_candidates(item.candidate_quotes)}"
            )
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=INSTRUCTION,
                        input=selection_input,
                        output=item.supporting_quote,
                    ),
                    AlpacaDataset(
                        instruction=RATIONALE_INSTRUCTION,
                        input=selection_input,
                        output=item.rationale,
                    ),
                ]
            )
            rows.extend(
                AlpacaDataset(
                    instruction=QUOTE_SUPPORT_INSTRUCTION,
                    input=(
                        f"Passage:\n{passages}\n\n"
                        f"Claim: {item.claim}\n\n"
                        f"Quote: {candidate}"
                    ),
                    output="yes" if candidate == item.supporting_quote else "no",
                )
                for candidate in item.candidate_quotes
                if quote_in_passage(candidate, passages)
            )
        return rows


class EvidenceSelectionAugmentation(BaseSingleChunkAugmentation[EvidenceSelections]):
    schema = EvidenceSelections
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains claims with multiple candidate supporting quotes."
    adapters = (EvidenceSelectionAdapter(),)
    temperature = 0.3
    instruction_templates = {
        INSTRUCTION: (
            INSTRUCTION,
            "Select the candidate quote that directly supports the claim. Answer with the exact quote.",
            "Choose the exact candidate quote that best supports the claim.",
            "Find the supporting quote among the candidates and copy it exactly.",
            "Answer with only the candidate quote that supports the claim.",
        ),
        RATIONALE_INSTRUCTION: (
            RATIONALE_INSTRUCTION,
            "Explain why the selected quote supports the claim, using only the passage.",
            "Briefly justify why this quote is evidence for the claim.",
            "State why the selected quote supports the claim.",
            "Give a passage-grounded rationale for the selected supporting quote.",
        ),
        QUOTE_SUPPORT_INSTRUCTION: (
            QUOTE_SUPPORT_INSTRUCTION,
            "Decide whether the quote supports the claim. Answer yes or no.",
            "Based on the passage, does this quote support the claim? Answer yes or no.",
            "Label whether the quote supports the claim with yes or no.",
            "Answer yes if the quote supports the claim, otherwise answer no.",
        ),
    }


evidence_selection_augmentation = EvidenceSelectionAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        evidence_selection_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We propose the Transformer, based solely on attention mechanisms. Experiments show these models are more parallelizable and require significantly less time to train.
        """
    )
    print(len(dataset))
