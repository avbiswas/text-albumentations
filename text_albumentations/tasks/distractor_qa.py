import json

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class KeywordReplacement(BaseModel):
    keyword: str = Field(max_length=80)
    replacement: str = Field(max_length=80)


class MultipleChoiceQuestion(BaseModel):
    question: str = Field(max_length=300)
    correct_answer: str = Field(max_length=200)
    distractors: list[str] = Field(min_length=2, max_length=4)
    explanation: str = Field(max_length=500)
    keyword_replacements: list[KeywordReplacement] = Field(min_length=1, max_length=4)


class MultipleChoiceQuestions(BaseModel):
    questions: list[MultipleChoiceQuestion] = Field(min_length=1, max_length=4)


SYSTEM_PROMPT = """
Given this passage, generate multiple-choice questions. Each question must be answerable from the passage, have one correct answer, and include plausible but incorrect distractors. Explain why the correct answer follows from the passage. Also identify passage keywords and antagonizing replacements that would make a related corrupted passage invalid. Only generate answers.
    """

INSTRUCTION = (
    "Answer the multiple-choice question using the passage. "
    "Return the correct answer and a brief explanation."
)
ANSWER_ONLY_INSTRUCTION = "Answer the multiple-choice question using the passage."
EXPLANATION_INSTRUCTION = "Explain why the selected answer is correct based on the passage."
DISTRACTOR_INSTRUCTION = "List the incorrect answer choices from this multiple-choice question."
VALIDITY_INSTRUCTION = (
    "Determine whether the candidate passage is valid relative to the reference passage. "
    "Answer valid or invalid."
)
CORRECTION_INSTRUCTION = (
    "Correct the candidate passage so it matches the reference passage."
)
REPLACEMENT_INSTRUCTION = (
    "Identify the keyword replacement that made this candidate passage invalid."
)


def _choices(item: MultipleChoiceQuestion) -> list[str]:
    return [item.correct_answer, *item.distractors]


def _replace_once(text: str, keyword: str, replacement: str) -> str | None:
    if not keyword or not replacement or keyword == replacement:
        return None
    index = text.lower().find(keyword.lower())
    if index < 0:
        return None
    return f"{text[:index]}{replacement}{text[index + len(keyword):]}"


class DistractorQaAdapter(BaseAlpacaAdapter[str, MultipleChoiceQuestions]):
    def convert(self, passages: str, output: MultipleChoiceQuestions) -> list[AlpacaDataset]:
        rows = []
        for item in output.questions:
            choices = _choices(item)
            if len(set(choices)) != len(choices):
                continue
            question_input = (
                f"Passage:\n{passages}\n\n"
                f"Question: {item.question}\n\n"
                f"Choices: {json.dumps(choices, ensure_ascii=False)}"
            )
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=INSTRUCTION,
                        input=question_input,
                        output=(
                            f"Answer: {item.correct_answer}\n"
                            f"Explanation: {item.explanation}"
                        ),
                    ),
                    AlpacaDataset(
                        instruction=ANSWER_ONLY_INSTRUCTION,
                        input=question_input,
                        output=item.correct_answer,
                    ),
                    AlpacaDataset(
                        instruction=EXPLANATION_INSTRUCTION,
                        input=(
                            f"{question_input}\n\n"
                            f"Selected answer: {item.correct_answer}"
                        ),
                        output=item.explanation,
                    ),
                    AlpacaDataset(
                        instruction=DISTRACTOR_INSTRUCTION,
                        input=question_input,
                        output=json.dumps(item.distractors, ensure_ascii=False),
                    ),
                ]
            )
            for replacement in item.keyword_replacements:
                corrupted = _replace_once(
                    passages,
                    replacement.keyword,
                    replacement.replacement,
                )
                if corrupted is None:
                    continue
                validity_input = (
                    f"Reference passage:\n{passages}\n\n"
                    f"Candidate passage:\n{corrupted}"
                )
                rows.extend(
                    [
                        AlpacaDataset(
                            instruction=VALIDITY_INSTRUCTION,
                            input=validity_input,
                            output="invalid",
                        ),
                        AlpacaDataset(
                            instruction=CORRECTION_INSTRUCTION,
                            input=validity_input,
                            output=passages,
                        ),
                        AlpacaDataset(
                            instruction=REPLACEMENT_INSTRUCTION,
                            input=validity_input,
                            output=(
                                f"{replacement.keyword} -> {replacement.replacement}"
                            ),
                        ),
                    ]
                )
        return rows


class DistractorQaAugmentation(BaseSingleChunkAugmentation[MultipleChoiceQuestions]):
    schema = MultipleChoiceQuestions
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage supports factual questions with plausible incorrect alternatives."
    adapters = (DistractorQaAdapter(),)
    temperature = 0.5
    instruction_templates = {
        INSTRUCTION: (
            INSTRUCTION,
            "Use the passage to answer the multiple-choice question with the correct answer and a brief explanation.",
            "Choose the correct answer from the choices and explain it using the passage.",
            "Answer this passage-grounded multiple-choice question and include a short explanation.",
            "Return the correct choice and explain why it follows from the passage.",
        ),
        ANSWER_ONLY_INSTRUCTION: (
            ANSWER_ONLY_INSTRUCTION,
            "Choose the correct answer to the multiple-choice question using the passage.",
            "Return only the correct answer choice based on the passage.",
            "Select the passage-supported answer from the choices.",
            "Answer with the correct choice only.",
        ),
        EXPLANATION_INSTRUCTION: (
            EXPLANATION_INSTRUCTION,
            "Explain why the selected answer is correct, using only the passage.",
            "Justify the selected correct answer from the passage.",
            "Give a passage-grounded explanation for the selected answer.",
            "State why this answer follows from the passage.",
        ),
        DISTRACTOR_INSTRUCTION: (
            DISTRACTOR_INSTRUCTION,
            "List only the incorrect answer choices from this multiple-choice question.",
            "Return the distractors, meaning the choices that are incorrect.",
            "Identify the incorrect answer choices.",
            "List the wrong choices from the multiple-choice question.",
        ),
        VALIDITY_INSTRUCTION: (
            VALIDITY_INSTRUCTION,
            "Decide whether the candidate passage is valid or invalid relative to the reference passage. Answer valid or invalid.",
            "Compare the candidate to the reference and label it valid or invalid.",
            "Answer invalid if the candidate contradicts the reference passage; otherwise answer valid.",
            "Label the candidate passage as valid or invalid against the reference.",
        ),
        CORRECTION_INSTRUCTION: (
            CORRECTION_INSTRUCTION,
            "Correct the invalid candidate passage so it matches the reference passage.",
            "Rewrite the candidate passage to align with the reference passage.",
            "Fix the candidate passage using the reference passage.",
            "Return the corrected passage that matches the reference.",
        ),
        REPLACEMENT_INSTRUCTION: (
            REPLACEMENT_INSTRUCTION,
            "Identify the keyword replacement that made the candidate passage invalid.",
            "Return the incorrect keyword replacement responsible for the invalid candidate.",
            "State the keyword-to-replacement pair that corrupted the candidate passage.",
            "Name the replacement that caused the candidate to become invalid.",
        ),
    }


distractor_qa_augmentation = DistractorQaAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        distractor_qa_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer dispenses with recurrence and convolutions, relying solely on attention mechanisms.
        """
    )
    print(len(dataset))
