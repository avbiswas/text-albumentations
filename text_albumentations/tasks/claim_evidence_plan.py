from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.tasks.extractive_qa import quote_in_passage
from text_albumentations.utils import AlpacaDataset


class ClaimEvidencePlan(BaseModel):
    claim: str = Field(max_length=300)
    supporting_quote: str = Field(max_length=500)
    unrelated_quote: str = Field(max_length=500)


SYSTEM_PROMPT = """
Given this passage, write one short claim that is directly supported by an exact quote from the passage. Also copy one unrelated quote from the passage that does not establish the claim. Both quotes must be copied verbatim from the passage. Only generate answers.
    """

VERIFY_INSTRUCTION = (
    "Classify the claim as supported, refuted, or not enough information "
    "based only on the evidence."
)
EVIDENCE_SELECTION_INSTRUCTION = (
    "Choose which candidate quote supports the claim. Answer with the exact quote."
)


class ClaimEvidenceAdapter(BaseAlpacaAdapter[str, ClaimEvidencePlan]):
    def convert(self, passages: str, output: ClaimEvidencePlan) -> list[AlpacaDataset]:
        if not quote_in_passage(output.supporting_quote, passages):
            return []
        if not quote_in_passage(output.unrelated_quote, passages):
            return []
        if output.supporting_quote == output.unrelated_quote:
            return []

        supported_input = (
            f"Claim: {output.claim}\n\n"
            f"Evidence: {output.supporting_quote}"
        )
        unrelated_input = (
            f"Claim: {output.claim}\n\n"
            f"Evidence: {output.unrelated_quote}"
        )
        candidates_input = (
            f"Claim: {output.claim}\n\n"
            f"Candidate quotes:\n"
            f"1. {output.supporting_quote}\n"
            f"2. {output.unrelated_quote}"
        )
        return [
            AlpacaDataset(
                instruction=VERIFY_INSTRUCTION,
                input=supported_input,
                output="supported",
            ),
            AlpacaDataset(
                instruction=VERIFY_INSTRUCTION,
                input=unrelated_input,
                output="not enough information",
            ),
            AlpacaDataset(
                instruction=EVIDENCE_SELECTION_INSTRUCTION,
                input=candidates_input,
                output=output.supporting_quote,
            ),
        ]


class ClaimEvidenceAugmentation(BaseSingleChunkAugmentation[ClaimEvidencePlan]):
    schema = ClaimEvidencePlan
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains explicit evidence that can support a short factual claim, plus other quoteable text."
    adapters = (ClaimEvidenceAdapter(),)
    temperature = 0.3
    instruction_templates = {
        VERIFY_INSTRUCTION: (
            VERIFY_INSTRUCTION,
            "Using only the evidence, label the claim as supported, refuted, or not enough information.",
            "Decide whether the evidence supports, refutes, or does not establish the claim.",
            "Return the factual verification label for the claim and evidence.",
        ),
        EVIDENCE_SELECTION_INSTRUCTION: (
            EVIDENCE_SELECTION_INSTRUCTION,
            "Select the candidate quote that supports the claim. Answer with the exact quote.",
            "Choose the exact evidence quote that supports the claim.",
            "Find the supporting quote among the candidates and copy it exactly.",
        ),
    }


claim_evidence_augmentation = ClaimEvidenceAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        claim_evidence_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is based solely on attention mechanisms. Convolutional networks use learned filters over local neighborhoods.
        """
    )
    print(len(dataset))
