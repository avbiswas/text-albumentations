from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.tasks.extractive_qa import quote_in_passage
from text_albumentations.utils import AlpacaDataset


class EquivalentRewrite(BaseModel):
    source_sentence: str = Field(max_length=500)
    equivalent_sentences: list[str] = Field(min_length=1, max_length=4)


SYSTEM_PROMPT = """
Given this passage, choose one self-contained sentence from the passage and write meaning-preserving rewrites of that sentence. The source sentence must be copied verbatim from the passage. Each rewrite must preserve the factual meaning of the source sentence and avoid adding new facts. Only generate answers.
    """

SIMILARITY_INSTRUCTION = (
    "Rate the semantic similarity of Sentence A and Sentence B from 0.0 to 1.0."
)
EQUIVALENCE_INSTRUCTION = (
    "Determine whether Sentence A and Sentence B are semantically equivalent. "
    "Answer equivalent or not_equivalent."
)
NLI_INSTRUCTION = (
    "Classify the relationship between the premise and hypothesis as entailment, "
    "neutral, or contradiction."
)


def _pair_input(sentence_a: str, sentence_b: str) -> str:
    return f"Sentence A: {sentence_a}\n\nSentence B: {sentence_b}"


def _nli_input(premise: str, hypothesis: str) -> str:
    return f"Premise: {premise}\n\nHypothesis: {hypothesis}"


class SentenceEquivalenceAdapter(BaseAlpacaAdapter[str, EquivalentRewrite]):
    def convert(self, passages: str, output: EquivalentRewrite) -> list[AlpacaDataset]:
        if not quote_in_passage(output.source_sentence, passages):
            return []

        rows = []
        for equivalent in output.equivalent_sentences:
            if not equivalent.strip() or equivalent.strip() == output.source_sentence.strip():
                continue
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=SIMILARITY_INSTRUCTION,
                        input=_pair_input(output.source_sentence, equivalent),
                        output="1.0",
                    ),
                    AlpacaDataset(
                        instruction=EQUIVALENCE_INSTRUCTION,
                        input=_pair_input(output.source_sentence, equivalent),
                        output="equivalent",
                    ),
                    AlpacaDataset(
                        instruction=NLI_INSTRUCTION,
                        input=_nli_input(output.source_sentence, equivalent),
                        output="entailment",
                    ),
                    AlpacaDataset(
                        instruction=NLI_INSTRUCTION,
                        input=_nli_input(equivalent, output.source_sentence),
                        output="entailment",
                    ),
                ]
            )
        return rows


class SentenceEquivalenceAugmentation(BaseSingleChunkAugmentation[EquivalentRewrite]):
    schema = EquivalentRewrite
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains a self-contained sentence that can be safely paraphrased without changing meaning."
    adapters = (SentenceEquivalenceAdapter(),)
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
    }


sentence_equivalence_augmentation = SentenceEquivalenceAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        sentence_equivalence_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is a sequence transduction architecture based solely on attention mechanisms.
        """
    )
    print(len(dataset))
