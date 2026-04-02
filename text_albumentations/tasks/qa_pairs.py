import json

from pydantic import BaseModel, Field, create_model

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.response_formats import AlpacaResponseFormat
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class QA(BaseModel):
    question: str
    answer: str


class QAList(BaseModel):
    qa_pairs: list[QA] = Field(..., min_length=1, max_length=3)


SYSTEM_PROMPT = """
Given this passage of text, generate a list of important question answer pairs.
    """


def format_qa_pairs_markdown(qa_pairs: list[QA]) -> str:
    sections = []
    for idx, qa_pair in enumerate(qa_pairs, start=1):
        sections.append(
            f"### Q{idx}\n"
            f"**Question:** {qa_pair.question}\n\n"
            f"**Answer:** {qa_pair.answer}"
        )
    return "\n\n".join(sections)


def format_questions_markdown(qa_pairs: list[QA]) -> str:
    return "\n".join(
        f"{idx}. {qa_pair.question}"
        for idx, qa_pair in enumerate(qa_pairs, start=1)
    )


def format_facts_markdown(qa_pairs: list[QA]) -> str:
    return "\n".join(f"- {qa_pair.answer}" for qa_pair in qa_pairs)


def format_single_qa_markdown(qa_pair: QA) -> str:
    return (
        f"**Question:** {qa_pair.question}\n\n"
        f"**Answer:** {qa_pair.answer}"
    )


class MarkdownQaAdapter(BaseAlpacaAdapter[str, QAList]):
    def convert(self, passages: str, output: QAList) -> list[AlpacaDataset]:
        qa_pairs_markdown = format_qa_pairs_markdown(output.qa_pairs)
        questions_markdown = format_questions_markdown(output.qa_pairs)
        facts_markdown = format_facts_markdown(output.qa_pairs)
        dataset = [
            AlpacaDataset(
                instruction=SYSTEM_PROMPT,
                input=passages,
                output=qa_pairs_markdown,
            ),
            AlpacaDataset(
                instruction="Generate a set of questions from this passage in markdown format.",
                input=passages,
                output=questions_markdown,
            ),
            AlpacaDataset(
                instruction="List the important questions answered by this passage using markdown.",
                input=passages,
                output=questions_markdown,
            ),
            AlpacaDataset(
                instruction="Generate some important facts from this passage in markdown bullet points.",
                input=passages,
                output=facts_markdown,
            ),
        ]

        for qa_pair in output.qa_pairs:
            dataset.extend(
                [
                    AlpacaDataset(
                        instruction="Generate one question and its corresponding answer from this passage in markdown format.",
                        input=passages,
                        output=format_single_qa_markdown(qa_pair),
                    ),
                    AlpacaDataset(
                        instruction="Generate a question from this passage",
                        input=passages,
                        output=qa_pair.question,
                    ),
                    AlpacaDataset(
                        instruction="Generate an important fact or piece of information from this passage",
                        input=passages,
                        output=qa_pair.answer,
                    ),
                    AlpacaDataset(
                        instruction="Answer the user's question given the provided passage",
                        input=f"Passage: {passages}\n\nQuestion: {qa_pair.question}\nWhat is the answer?",
                        output=qa_pair.answer,
                    ),
                    AlpacaDataset(
                        instruction=f"Given the provided passage, answer the user's question. Passage: {passages}",
                        input=qa_pair.question,
                        output=qa_pair.answer,
                    ),
                ]
            )

        return dataset


class JsonQaAdapter(BaseAlpacaAdapter[str, QAList]):
    def convert(self, passages: str, output: QAList) -> list[AlpacaDataset]:
        dataset = [
            AlpacaDataset(
                instruction=SYSTEM_PROMPT + "Generate as a list of json containing 'question' and 'answer' keys",
                input=passages,
                output=json.dumps(
                    [qa_pair.model_dump() for qa_pair in output.qa_pairs],
                    ensure_ascii=False,
                ),
            ),
            AlpacaDataset(
                instruction="Generate a list of questions from this passage. Return a JSON array of strings.",
                input=passages,
                output=json.dumps(
                    [qa_pair.question for qa_pair in output.qa_pairs],
                    ensure_ascii=False,
                ),
            ),
            AlpacaDataset(
                instruction="List the important questions answered by this passage. Return a JSON array of strings.",
                input=passages,
                output=json.dumps(
                    [qa_pair.question for qa_pair in output.qa_pairs],
                    ensure_ascii=False,
                ),
            ),
            AlpacaDataset(
                instruction="Generate some facts from this passage. Return a JSON array of strings.",
                input=passages,
                output=json.dumps(
                    [qa_pair.answer for qa_pair in output.qa_pairs],
                    ensure_ascii=False,
                ),
            ),
        ]

        for qa_pair in output.qa_pairs:
            dataset.extend(
                [
                    AlpacaDataset(
                        instruction="Generate one question and it's corresponding answer from this passage. Return answer as a json of question and answer",
                        input=passages,
                        output=qa_pair.model_dump_json(),
                    ),
                    AlpacaDataset(
                        instruction="Generate a question from this passage",
                        input=passages,
                        output=qa_pair.question,
                    ),
                    AlpacaDataset(
                        instruction="Generate an important fact or piece of information from this passage",
                        input=passages,
                        output=qa_pair.answer,
                    ),
                    AlpacaDataset(
                        instruction="Answer the user's question given the provided passage",
                        input=f"Passage: {passages}\n\nQuestion: {qa_pair.question}\nWhat is the answer?",
                        output=qa_pair.answer,
                    ),
                    AlpacaDataset(
                        instruction=f"Given the provided passage, answer the user's question. Passage: {passages}",
                        input=qa_pair.question,
                        output=qa_pair.answer,
                    ),
                ]
            )

        return dataset


class QaPairAugmentation(BaseSingleChunkAugmentation[QAList]):
    schema = QAList
    system_prompt = SYSTEM_PROMPT
    response_formats = (
        AlpacaResponseFormat(
            name="markdown",
            adapter=MarkdownQaAdapter(),
            format_instruction=(
                "Represent the question-answer pairs using markdown."
            ),
        ),
        AlpacaResponseFormat(
            name="json",
            adapter=JsonQaAdapter(),
            format_instruction=(
                "Represent the question-answer pairs using JSON."
            ),
        ),
    )

    def __init__(
        self,
        *,
        max_qa_pairs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_qa_pairs = max_qa_pairs
        self._configured_schema: type[QAList] | None = None

    def get_schema(self) -> type[QAList]:
        if self.max_qa_pairs == 3:
            return self.schema
        if self._configured_schema is None:
            self._configured_schema = create_model(
                "ConfiguredQAList",
                qa_pairs=(list[QA], Field(..., min_length=1, max_length=self.max_qa_pairs)),
                __base__=BaseModel,
            )
        return self._configured_schema


qa_pair_augmentation = QaPairAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        qa_pair_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
             """
    )

    print(len(dataset))
