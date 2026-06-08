from typing import Literal
import json

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


ClaimLabel = Literal["supported", "contradicted", "not enough information"]


class ClaimVerificationItem(BaseModel):
    claim: str = Field(max_length=300)
    label: ClaimLabel
    evidence: str = Field(max_length=600)


class ClaimVerifications(BaseModel):
    items: list[ClaimVerificationItem] = Field(min_length=1, max_length=5)


SYSTEM_PROMPT = """
Given this passage, create claim-verification examples. Include claims that are supported, contradicted, or not fully answerable from the passage when possible. For each claim, assign exactly one label: supported, contradicted, or not enough information. Write concise evidence grounded in the passage. Only generate answers.
    """

INSTRUCTION = (
    "Classify the claim as supported, contradicted, or not enough information "
    "based only on the passage. Include a short evidence explanation."
)
LABEL_INSTRUCTION = (
    "Classify the claim as supported, contradicted, or not enough information "
    "based only on the passage."
)
EVIDENCE_INSTRUCTION = "Provide the passage-grounded evidence for this claim label."
JSON_INSTRUCTION = (
    "Classify the claim based only on the passage. "
    "Return a JSON object with keys 'label' and 'evidence'."
)


class ClaimVerificationAdapter(BaseAlpacaAdapter[str, ClaimVerifications]):
    def convert(self, passages: str, output: ClaimVerifications) -> list[AlpacaDataset]:
        rows = []
        for item in output.items:
            row_input = f"Passage:\n{passages}\n\nClaim: {item.claim}"
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=INSTRUCTION,
                        input=row_input,
                        output=f"Label: {item.label}\nEvidence: {item.evidence}",
                    ),
                    AlpacaDataset(
                        instruction=LABEL_INSTRUCTION,
                        input=row_input,
                        output=item.label,
                    ),
                    AlpacaDataset(
                        instruction=EVIDENCE_INSTRUCTION,
                        input=(
                            f"{row_input}\n\n"
                            f"Label: {item.label}"
                        ),
                        output=item.evidence,
                    ),
                    AlpacaDataset(
                        instruction=JSON_INSTRUCTION,
                        input=row_input,
                        output=json.dumps(
                            {"label": item.label, "evidence": item.evidence},
                            ensure_ascii=False,
                        ),
                    ),
                ]
            )
        return rows


class ClaimVerificationAugmentation(BaseSingleChunkAugmentation[ClaimVerifications]):
    schema = ClaimVerifications
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains factual claims that can support entailment-style labels."
    adapters = (ClaimVerificationAdapter(),)
    temperature = 0.4
    instruction_templates = {
        INSTRUCTION: (
            INSTRUCTION,
            "Using only the passage, label the claim as supported, contradicted, or not enough information, and include brief evidence.",
            "Classify this claim from the passage and provide a short evidence explanation.",
            "Decide whether the passage supports, contradicts, or does not establish the claim; include evidence.",
            "Give the claim label and passage-grounded evidence.",
        ),
        LABEL_INSTRUCTION: (
            LABEL_INSTRUCTION,
            "Using only the passage, choose the claim label: supported, contradicted, or not enough information.",
            "Label this claim from the passage as supported, contradicted, or not enough information.",
            "Classify the claim with one of the allowed passage-grounded labels.",
            "Return only the claim label based on the passage.",
        ),
        EVIDENCE_INSTRUCTION: (
            EVIDENCE_INSTRUCTION,
            "Provide evidence from the passage for the given claim label.",
            "Explain the given claim label using only passage-grounded evidence.",
            "Write the evidence that justifies this label for the claim.",
            "Give a concise passage-based justification for the claim label.",
        ),
        JSON_INSTRUCTION: (
            JSON_INSTRUCTION,
            "Classify the claim from the passage and return JSON with 'label' and 'evidence'.",
            "Return a JSON object containing the passage-grounded claim label and evidence.",
            "Using only the passage, output JSON with keys 'label' and 'evidence' for this claim.",
            "Produce JSON for this claim verification with 'label' and 'evidence'.",
        ),
    }


claim_verification_augmentation = ClaimVerificationAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        claim_verification_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is based solely on attention mechanisms and dispenses with recurrence and convolutions entirely.
        """
    )
    print(len(dataset))
