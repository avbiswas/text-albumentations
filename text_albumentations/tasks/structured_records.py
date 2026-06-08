import json

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class StructuredRecord(BaseModel):
    subject: str = Field(max_length=160)
    attribute: str = Field(max_length=160)
    value: str = Field(max_length=240)


class StructuredRecords(BaseModel):
    records: list[StructuredRecord] = Field(min_length=1, max_length=10)


SYSTEM_PROMPT = """
Given this passage, extract compact structured records as subject-attribute-value triples. Prefer dates, metrics, results, properties, roles, settings, and outcomes explicitly stated in the passage. Do not invent values. Only generate answers.
    """

JSON_INSTRUCTION = (
    "Extract structured records from this passage. "
    "Return a JSON array of subject, attribute, and value objects."
)
TEXT_INSTRUCTION = "State the extracted facts from this passage as concise sentences."
VALUE_LOOKUP_INSTRUCTION = (
    "Given the passage, subject, and attribute, return the corresponding value."
)
ATTRIBUTE_LOOKUP_INSTRUCTION = (
    "Given the passage, subject, and value, return the corresponding attribute."
)
SUBJECT_LOOKUP_INSTRUCTION = (
    "Given the passage, attribute, and value, return the corresponding subject."
)


def _sentence(record: StructuredRecord) -> str:
    return f"{record.subject} - {record.attribute}: {record.value}"


class StructuredRecordsAdapter(BaseAlpacaAdapter[str, StructuredRecords]):
    def convert(self, passages: str, output: StructuredRecords) -> list[AlpacaDataset]:
        rows = [
            AlpacaDataset(
                instruction=JSON_INSTRUCTION,
                input=passages,
                output=json.dumps(
                    [record.model_dump() for record in output.records],
                    ensure_ascii=False,
                ),
            ),
            AlpacaDataset(
                instruction=TEXT_INSTRUCTION,
                input=passages,
                output="\n".join(_sentence(record) for record in output.records),
            ),
        ]
        for record in output.records:
            rows.extend(
                [
                    AlpacaDataset(
                        instruction=VALUE_LOOKUP_INSTRUCTION,
                        input=(
                            f"Passage:\n{passages}\n\n"
                            f"Subject: {record.subject}\n"
                            f"Attribute: {record.attribute}"
                        ),
                        output=record.value,
                    ),
                    AlpacaDataset(
                        instruction=ATTRIBUTE_LOOKUP_INSTRUCTION,
                        input=(
                            f"Passage:\n{passages}\n\n"
                            f"Subject: {record.subject}\n"
                            f"Value: {record.value}"
                        ),
                        output=record.attribute,
                    ),
                    AlpacaDataset(
                        instruction=SUBJECT_LOOKUP_INSTRUCTION,
                        input=(
                            f"Passage:\n{passages}\n\n"
                            f"Attribute: {record.attribute}\n"
                            f"Value: {record.value}"
                        ),
                        output=record.subject,
                    ),
                ]
            )
        return rows


class StructuredRecordsAugmentation(BaseSingleChunkAugmentation[StructuredRecords]):
    schema = StructuredRecords
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains structured facts, measurements, dates, properties, roles, or outcomes."
    adapters = (StructuredRecordsAdapter(),)
    temperature = 0.2
    instruction_templates = {
        JSON_INSTRUCTION: (
            JSON_INSTRUCTION,
            "Extract subject, attribute, and value records from the passage as a JSON array.",
            "Return passage-grounded structured records as JSON objects with subject, attribute, and value.",
            "Convert the explicit facts in this passage into subject-attribute-value JSON records.",
            "Produce a JSON array of structured records from this passage.",
        ),
        TEXT_INSTRUCTION: (
            TEXT_INSTRUCTION,
            "List the extracted passage facts as concise subject-attribute-value lines.",
            "State the structured facts from the passage concisely.",
            "Write the extracted records as compact factual lines.",
            "Render the passage's structured facts as concise text.",
        ),
        VALUE_LOOKUP_INSTRUCTION: (
            VALUE_LOOKUP_INSTRUCTION,
            "Using the passage, return the value for the given subject and attribute.",
            "Find the value that matches this subject and attribute in the passage.",
            "Answer with the passage-grounded value for the subject and attribute.",
            "Look up the corresponding value from the passage.",
        ),
        ATTRIBUTE_LOOKUP_INSTRUCTION: (
            ATTRIBUTE_LOOKUP_INSTRUCTION,
            "Using the passage, return the attribute for the given subject and value.",
            "Find the attribute that links this subject to the given value in the passage.",
            "Answer with the passage-grounded attribute for the subject and value.",
            "Look up the corresponding attribute from the passage.",
        ),
        SUBJECT_LOOKUP_INSTRUCTION: (
            SUBJECT_LOOKUP_INSTRUCTION,
            "Using the passage, return the subject for the given attribute and value.",
            "Find the subject that matches this attribute and value in the passage.",
            "Answer with the passage-grounded subject for the attribute and value.",
            "Look up the corresponding subject from the passage.",
        ),
    }


structured_records_augmentation = StructuredRecordsAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        structured_records_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
Experiments on two machine translation tasks show the model is more parallelizable and requires less time to train.
        """
    )
    print(len(dataset))
