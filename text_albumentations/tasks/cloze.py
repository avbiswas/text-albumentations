import re

from pydantic import BaseModel

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import ModelRuntime, get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset

NUM_WORD_BLANKS = 3
MIN_MASKABLE_WORD_LENGTH = 6
MIN_PASSAGE_WORDS = 10


class ClozeSlices(BaseModel):
    word_masked_passage: str
    word_answers: str
    sentence_masked_passage: str
    sentence_answer: str


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def pick_maskable_words(text: str, count: int) -> list[str]:
    candidates = {
        word.strip(".,;:!?()\"'")
        for word in text.split()
    }
    ranked = sorted(
        (word for word in candidates if len(word) >= MIN_MASKABLE_WORD_LENGTH and word.isalpha()),
        key=lambda word: (-len(word), word.lower()),
    )
    return ranked[:count]


class ClozeAdapter(BaseAlpacaAdapter[str, ClozeSlices]):
    def convert(self, passages: str, output: ClozeSlices) -> list[AlpacaDataset]:
        dataset = [
            AlpacaDataset(
                instruction=(
                    "Fill in the blanks in this passage. Each blank is marked as [BLANK n]. "
                    "Answer with the missing words in order, one per line, as '[BLANK n]: word'."
                ),
                input=output.word_masked_passage,
                output=output.word_answers,
            ),
            AlpacaDataset(
                instruction=(
                    "One sentence in this passage has been replaced with [MISSING SENTENCE]. "
                    "Write the missing sentence so it fits the surrounding context."
                ),
                input=output.sentence_masked_passage,
                output=output.sentence_answer,
            ),
        ]
        return [row for row in dataset if row.input and row.output]


class ClozeAugmentation(BaseSingleChunkAugmentation[ClozeSlices]):
    schema = ClozeSlices
    system_prompt = "Generate cloze supervision slices from the provided passage."
    selection_hint = "the passage has multiple sentences with distinctive content words to mask."
    adapters = (ClozeAdapter(),)

    def generate_one(
        self,
        passages: str,
        runtime: ModelRuntime,
        response_format=None,
    ) -> ClozeSlices:
        empty = ClozeSlices(
            word_masked_passage="",
            word_answers="",
            sentence_masked_passage="",
            sentence_answer="",
        )

        if len(passages.split()) < MIN_PASSAGE_WORDS:
            return empty

        masked_words = pick_maskable_words(passages, NUM_WORD_BLANKS)
        word_masked_passage = passages
        word_answers = []
        for index, word in enumerate(masked_words, start=1):
            word_masked_passage = re.sub(
                rf"\b{re.escape(word)}\b",
                f"[BLANK {index}]",
                word_masked_passage,
                count=1,
            )
            word_answers.append(f"[BLANK {index}]: {word}")

        sentences = split_sentences(passages)
        sentence_masked_passage = ""
        sentence_answer = ""
        if len(sentences) >= 3:
            middle = len(sentences) // 2
            sentence_answer = sentences[middle]
            sentence_masked_passage = " ".join(
                "[MISSING SENTENCE]" if index == middle else sentence
                for index, sentence in enumerate(sentences)
            )

        return ClozeSlices(
            word_masked_passage=word_masked_passage if masked_words else "",
            word_answers="\n".join(word_answers),
            sentence_masked_passage=sentence_masked_passage,
            sentence_answer=sentence_answer,
        )

    async def agenerate_one(
        self,
        passages: str,
        runtime: ModelRuntime,
        response_format=None,
    ) -> ClozeSlices:
        return self.generate_one(passages, runtime)


cloze_augmentation = ClozeAugmentation()


def main(chunk: str) -> list[AlpacaDataset]:
    return run_augmentation(
        chunk,
        cloze_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks in an encoder-decoder configuration. The best performing models also connect
the encoder and decoder through an attention mechanism. We propose a new simple network
architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence
and convolutions entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly less time to train.
        """
    )

    print(len(dataset))
