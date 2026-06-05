import asyncio
import json
import random

from pydantic import BaseModel, Field

from text_albumentations.base import BaseMultiChunkAugmentation
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import ModelRuntime, get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class UniqueQuestions(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=4)


class RetrievalReason(BaseModel):
    why: str = Field(..., max_length=300)


class NoAnswerReason(BaseModel):
    why: str = Field(..., max_length=300)


QUESTION_EXTRACTION_PROMPT = """
You will be given one passage.

Extract a small set of unique questions that can be answered directly from the passage.
Keep the questions short and specific.
Avoid vague questions such as "What is described?", "What does this talk about?", or similar generic wording.
Do not ask duplicate questions.
Do not ask questions that require outside knowledge.
Only return questions whose answers are clearly present in the passage.
"""


RETRIEVAL_REASON_PROMPT = """
You will be given one passage and one question.

Write a short reason explaining why the passage answers the question.
Keep the reason grounded in the passage.
Do not mention any passage numbers.
Do not use bullet points or JSON.
Be concise and specific.
"""


NO_ANSWER_REASON_PROMPT = """
You will be given several passages and one question.

Write a short reason explaining why none of the passages answer the question.
Keep the reason grounded in the passages.
Do not mention passage numbers.
Do not use bullet points or JSON.
Be concise and specific.
"""


RETRIEVAL_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return the passage number and a short justification in markdown."
)


RETRIEVAL_JSON_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return a JSON object with keys 'passage' and 'why'. "
    "Use a passage number when an answer is present, or null when none of the passages answer the question."
)


def extract_unique_questions(
    passage: str,
    runtime: ModelRuntime,
) -> UniqueQuestions:
    messages = [
        {"role": "system", "content": QUESTION_EXTRACTION_PROMPT},
        {"role": "user", "content": passage},
    ]
    return runtime.generate_structured(messages, UniqueQuestions, temperature=0.2)


async def aextract_unique_questions(
    passage: str,
    runtime: ModelRuntime,
) -> UniqueQuestions:
    messages = [
        {"role": "system", "content": QUESTION_EXTRACTION_PROMPT},
        {"role": "user", "content": passage},
    ]
    return await runtime.agenerate_structured(
        messages,
        UniqueQuestions,
        temperature=0.2,
    )


def generate_retrieval_reason(
    passage: str,
    question: str,
    runtime: ModelRuntime,
) -> RetrievalReason:
    messages = [
        {"role": "system", "content": RETRIEVAL_REASON_PROMPT},
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion: {question}",
        },
    ]
    return runtime.generate_structured(messages, RetrievalReason, temperature=0.2)


async def agenerate_retrieval_reason(
    passage: str,
    question: str,
    runtime: ModelRuntime,
) -> RetrievalReason:
    messages = [
        {"role": "system", "content": RETRIEVAL_REASON_PROMPT},
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion: {question}",
        },
    ]
    return await runtime.agenerate_structured(
        messages,
        RetrievalReason,
        temperature=0.2,
    )


def generate_no_answer_reason(
    passages: list[str],
    question: str,
    runtime: ModelRuntime,
) -> NoAnswerReason:
    messages = [
        {"role": "system", "content": NO_ANSWER_REASON_PROMPT},
        {
            "role": "user",
            "content": f"{format_passages(passages)}\n\nQuestion: {question}",
        },
    ]
    return runtime.generate_structured(messages, NoAnswerReason, temperature=0.2)


async def agenerate_no_answer_reason(
    passages: list[str],
    question: str,
    runtime: ModelRuntime,
) -> NoAnswerReason:
    messages = [
        {"role": "system", "content": NO_ANSWER_REASON_PROMPT},
        {
            "role": "user",
            "content": f"{format_passages(passages)}\n\nQuestion: {question}",
        },
    ]
    return await runtime.agenerate_structured(
        messages,
        NoAnswerReason,
        temperature=0.2,
    )


def format_passages(passages: list[str]) -> str:
    return "\n\n".join(
        f"Passage {idx}:\n{passage}"
        for idx, passage in enumerate(passages, start=1)
    )


def build_retrieval_input(passages: list[str], question: str) -> str:
    return f"{format_passages(passages)}\n\nQuestion: {question}"


def build_retrieval_output(passage_index: int, reason: str) -> str:
    return f"**Passage:** {passage_index + 1}\n\n**Why:** {reason}"


def build_no_answer_output(reason: str) -> str:
    return f"**Passage:** None\n\n**Why:** {reason}"


def build_retrieval_json_output(passage_index: int, reason: str) -> str:
    return json.dumps(
        {"passage": passage_index + 1, "why": reason},
        ensure_ascii=False,
    )


def build_no_answer_json_output(reason: str) -> str:
    return json.dumps(
        {"passage": None, "why": reason},
        ensure_ascii=False,
    )


class RetrievalAugmentation(BaseMultiChunkAugmentation[UniqueQuestions]):
    schema = UniqueQuestions
    system_prompt = QUESTION_EXTRACTION_PROMPT
    temperature = 0.2

    def __init__(
        self,
        *,
        max_questions_per_passage: int | None = None,
        max_passages: int | None = None,
        include_negative_examples: bool = True,
        **kwargs,
    ) -> None:
        if max_questions_per_passage is not None and max_questions_per_passage <= 0:
            raise ValueError("max_questions_per_passage must be greater than 0.")
        if max_passages is not None and max_passages <= 0:
            raise ValueError("max_passages must be greater than 0.")

        super().__init__(**kwargs)
        self.max_questions_per_passage = max_questions_per_passage
        self.max_passages = max_passages
        self.include_negative_examples = include_negative_examples

    def build_user_message(self, passages: list[str]) -> str:
        return format_passages(passages)

    def _prepare_passages(self, passages: list[str]) -> list[str]:
        shuffled_passages = passages[:]
        random.shuffle(shuffled_passages)
        if self.max_passages is not None:
            return shuffled_passages[: self.max_passages]
        return shuffled_passages

    def _limit_questions(self, questions: list[str]) -> list[str]:
        if self.max_questions_per_passage is None:
            return questions
        return questions[: self.max_questions_per_passage]

    def _build_positive_rows(
        self,
        passages: list[str],
        correct_idx: int,
        question: str,
        reason: str,
    ) -> list[AlpacaDataset]:
        full_input = build_retrieval_input(passages, question)
        return [
            AlpacaDataset(
                instruction=RETRIEVAL_INSTRUCTION,
                input=full_input,
                output=build_retrieval_output(correct_idx, reason),
            ),
            AlpacaDataset(
                instruction=RETRIEVAL_JSON_INSTRUCTION,
                input=full_input,
                output=build_retrieval_json_output(correct_idx, reason),
            ),
        ]

    def _build_negative_rows(
        self,
        negative_passages: list[str],
        question: str,
        reason: str,
    ) -> list[AlpacaDataset]:
        negative_input = build_retrieval_input(negative_passages, question)
        return [
            AlpacaDataset(
                instruction=RETRIEVAL_INSTRUCTION,
                input=negative_input,
                output=build_no_answer_output(reason),
            ),
            AlpacaDataset(
                instruction=RETRIEVAL_JSON_INSTRUCTION,
                input=negative_input,
                output=build_no_answer_json_output(reason),
            ),
        ]

    def _negative_passages(
        self,
        passages: list[str],
        correct_idx: int,
    ) -> list[str]:
        if not self.include_negative_examples:
            return []
        return [
            passage
            for idx, passage in enumerate(passages)
            if idx != correct_idx
        ]

    def build_dataset(
        self,
        passages: list[str],
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        shuffled_passages = self._prepare_passages(passages)

        extracted_questions = [
            extract_unique_questions(passage, runtime)
            for passage in shuffled_passages
        ]

        dataset = []
        for correct_idx, questions_for_passage in enumerate(extracted_questions):
            for question in self._limit_questions(questions_for_passage.questions):
                positive_reason = generate_retrieval_reason(
                    shuffled_passages[correct_idx],
                    question,
                    runtime,
                )

                dataset.extend(
                    self._build_positive_rows(
                        shuffled_passages,
                        correct_idx,
                        question,
                        positive_reason.why,
                    )
                )

                negative_passages = self._negative_passages(
                    shuffled_passages,
                    correct_idx,
                )
                if not negative_passages:
                    continue

                negative_reason = generate_no_answer_reason(
                    negative_passages,
                    question,
                    runtime,
                )
                dataset.extend(
                    self._build_negative_rows(
                        negative_passages,
                        question,
                        negative_reason.why,
                    )
                )

        return dataset

    async def abuild_dataset(
        self,
        passages: list[str],
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        shuffled_passages = self._prepare_passages(passages)

        extracted_questions = await asyncio.gather(
            *[
                aextract_unique_questions(passage, runtime)
                for passage in shuffled_passages
            ]
        )

        reason_jobs = []
        for correct_idx, questions_for_passage in enumerate(extracted_questions):
            for question in self._limit_questions(questions_for_passage.questions):
                reason_jobs.append(
                    (
                        "positive",
                        shuffled_passages,
                        correct_idx,
                        question,
                        agenerate_retrieval_reason(
                            shuffled_passages[correct_idx],
                            question,
                            runtime,
                        ),
                    )
                )

                negative_passages = self._negative_passages(
                    shuffled_passages,
                    correct_idx,
                )
                if not negative_passages:
                    continue

                reason_jobs.append(
                    (
                        "negative",
                        negative_passages,
                        correct_idx,
                        question,
                        agenerate_no_answer_reason(
                            negative_passages,
                            question,
                            runtime,
                        ),
                    )
                )

        reasons = await asyncio.gather(*[job[-1] for job in reason_jobs])

        dataset = []
        for job, reason in zip(reason_jobs, reasons):
            kind, job_passages, correct_idx, question, _ = job
            if kind == "positive":
                dataset.extend(
                    self._build_positive_rows(
                        job_passages,
                        correct_idx,
                        question,
                        reason.why,
                    )
                )
            else:
                dataset.extend(
                    self._build_negative_rows(
                        job_passages,
                        question,
                        reason.why,
                    )
                )

        return dataset


retrieval_augmentation = RetrievalAugmentation()


def main(passages: list[str]) -> list[AlpacaDataset]:
    return run_augmentation(
        passages,
        retrieval_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        [
            """
The Transformer replaces recurrence with attention mechanisms and achieves strong translation results while improving parallelization.
            """,
            """
Convolutional networks apply learned filters over local neighborhoods and are widely used in computer vision tasks.
            """,
            """
Retrieval-augmented generation combines a retriever with a generator so the model can ground its response in external documents.
            """,
        ]
    )

    print(len(dataset))
