from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class ErrorCorrection(BaseModel):
    corrupted_passage: str = Field(max_length=2000)
    correction_notes: str = Field(max_length=500)


SYSTEM_PROMPT = """
Given this passage, create a minimally corrupted version with a small number of factual, wording, punctuation, or ordering errors. The corrupted passage should still look plausible but should be corrected back to the original passage. Also summarize the corrections. Only generate answers.
    """

CORRECT_INSTRUCTION = (
    "Correct the errors in this passage while preserving the intended meaning."
)
NOTES_INSTRUCTION = "Explain what corrections were needed for this corrupted passage."


class ErrorCorrectionAdapter(BaseAlpacaAdapter[str, ErrorCorrection]):
    def convert(self, passages: str, output: ErrorCorrection) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=CORRECT_INSTRUCTION,
                input=output.corrupted_passage,
                output=passages,
            ),
            AlpacaDataset(
                instruction=NOTES_INSTRUCTION,
                input=output.corrupted_passage,
                output=output.correction_notes,
            ),
        ]


class ErrorCorrectionAugmentation(BaseSingleChunkAugmentation[ErrorCorrection]):
    schema = ErrorCorrection
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage is coherent enough to create a minimally corrupted correction example."
    adapters = (ErrorCorrectionAdapter(),)
    temperature = 0.6
    instruction_templates = {
        CORRECT_INSTRUCTION: (
            CORRECT_INSTRUCTION,
            "Fix the errors in this passage while preserving the intended meaning.",
            "Correct the corrupted passage so it matches the intended original meaning.",
            "Rewrite this passage with the errors corrected.",
            "Return the corrected passage without preserving the introduced errors.",
        ),
        NOTES_INSTRUCTION: (
            NOTES_INSTRUCTION,
            "Describe the corrections needed for this corrupted passage.",
            "Explain which errors in the corrupted passage need to be fixed.",
            "Summarize the changes required to correct this passage.",
            "State what was wrong in the corrupted passage and how to fix it.",
        ),
    }


error_correction_augmentation = ErrorCorrectionAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        error_correction_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
        """
    )
    print(len(dataset))
