from typing import Literal

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.tasks.extractive_qa import quote_in_passage
from text_albumentations.tasks.sentence_equivalence_plan import (
    EQUIVALENCE_INSTRUCTION,
    NLI_INSTRUCTION,
    SIMILARITY_INSTRUCTION,
    _nli_input,
    _pair_input,
)
from text_albumentations.utils import AlpacaDataset


ContrastType = Literal["entity", "number", "date", "attribute", "negation", "relation"]


class ContrastReplacement(BaseModel):
    original: str = Field(max_length=120)
    replacement: str = Field(max_length=120)
    contradiction_type: ContrastType


class SentenceContrastPlan(BaseModel):
    source_sentence: str = Field(max_length=500)
    replacements: list[ContrastReplacement] = Field(min_length=1, max_length=4)


SYSTEM_PROMPT = """
Given this passage, choose one self-contained factual sentence from the passage and propose controlled replacements that would make the sentence contradict the passage. The source sentence must be copied verbatim from the passage. Each original string must appear in the source sentence. Only generate replacement operators, not full rewritten sentences. Only generate answers.
    """

CORRECTION_INSTRUCTION = (
    "Correct the candidate sentence so it matches the reference sentence."
)
REPLACEMENT_INSTRUCTION = (
    "Identify the replacement that made the candidate sentence contradict the reference."
)


def _replace_once(text: str, original: str, replacement: str) -> str | None:
    if not original or not replacement or original == replacement:
        return None
    index = text.lower().find(original.lower())
    if index < 0:
        return None
    return f"{text[:index]}{replacement}{text[index + len(original):]}"


class SentenceContrastAdapter(BaseAlpacaAdapter[str, SentenceContrastPlan]):
    def convert(self, passages: str, output: SentenceContrastPlan) -> list[AlpacaDataset]:
        if not quote_in_passage(output.source_sentence, passages):
            return []

        rows = []
        for replacement in output.replacements:
            candidate = _replace_once(
                output.source_sentence,
                replacement.original,
                replacement.replacement,
            )
            if candidate is None or candidate == output.source_sentence:
                continue

            pair_input = _pair_input(output.source_sentence, candidate)
            nli_input = _nli_input(output.source_sentence, candidate)
            correction_input = (
                f"Reference sentence: {output.source_sentence}\n\n"
                f"Candidate sentence: {candidate}"
            )
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=SIMILARITY_INSTRUCTION,
                        input=pair_input,
                        output="0.25",
                    ),
                    AlpacaDataset(
                        instruction=EQUIVALENCE_INSTRUCTION,
                        input=pair_input,
                        output="not_equivalent",
                    ),
                    AlpacaDataset(
                        instruction=NLI_INSTRUCTION,
                        input=nli_input,
                        output="contradiction",
                    ),
                    AlpacaDataset(
                        instruction=CORRECTION_INSTRUCTION,
                        input=correction_input,
                        output=output.source_sentence,
                    ),
                    AlpacaDataset(
                        instruction=REPLACEMENT_INSTRUCTION,
                        input=correction_input,
                        output=f"{replacement.original} -> {replacement.replacement}",
                    ),
                ]
            )
        return rows


class SentenceContrastAugmentation(BaseSingleChunkAugmentation[SentenceContrastPlan]):
    schema = SentenceContrastPlan
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains factual sentences with entities, numbers, dates, attributes, or relations that can be contrastively swapped."
    adapters = (SentenceContrastAdapter(),)
    temperature = 0.4
    instruction_templates = {
        SIMILARITY_INSTRUCTION: (
            SIMILARITY_INSTRUCTION,
            "Score the semantic similarity between Sentence A and Sentence B from 0.0 to 1.0.",
            "Return a 0.0 to 1.0 semantic similarity score for the two sentences.",
            "Judge how semantically similar the two sentences are on a 0.0 to 1.0 scale.",
        ),
        EQUIVALENCE_INSTRUCTION: (
            EQUIVALENCE_INSTRUCTION,
            "Decide whether the two sentences are semantically equivalent. Answer equivalent or not_equivalent.",
            "Label the sentence pair as equivalent or not_equivalent.",
            "Answer whether Sentence A and Sentence B preserve the same meaning.",
        ),
        NLI_INSTRUCTION: (
            NLI_INSTRUCTION,
            "Label the premise-hypothesis relationship as entailment, neutral, or contradiction.",
            "Classify whether the premise entails, contradicts, or is neutral toward the hypothesis.",
            "Return the NLI label for the premise and hypothesis.",
        ),
        CORRECTION_INSTRUCTION: (
            CORRECTION_INSTRUCTION,
            "Fix the candidate sentence so it matches the reference sentence.",
            "Rewrite the candidate sentence to align with the reference sentence.",
            "Correct the contradicted candidate sentence using the reference.",
        ),
        REPLACEMENT_INSTRUCTION: (
            REPLACEMENT_INSTRUCTION,
            "Return the original-to-replacement pair that caused the contradiction.",
            "Identify the invalid replacement in original -> replacement form.",
            "State the replacement operator that made the candidate sentence wrong.",
        ),
    }


sentence_contrast_augmentation = SentenceContrastAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        sentence_contrast_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is a sequence transduction architecture based solely on attention mechanisms.
        """
    )
    print(len(dataset))
