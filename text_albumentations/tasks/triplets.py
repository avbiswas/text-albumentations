import json

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.response_formats import AlpacaResponseFormat
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class Triplet(BaseModel):
    subject: str = Field(..., max_length=120)
    relation: str = Field(..., max_length=120)
    object: str = Field(..., max_length=160)


class TripletList(BaseModel):
    triplets: list[Triplet] = Field(..., min_length=1, max_length=2)


SYSTEM_PROMPT = """
Given this passage, extract a small set of knowledge graph triplets.
Each triplet must be directly supported by the passage.
Use short, clear phrases for the subject, relation, and object.
Do not invent facts not present in the passage.
"""


def format_triplets_markdown(triplets: list[Triplet]) -> str:
    return "\n".join(
        f"- ({triplet.subject}, {triplet.relation}, {triplet.object})"
        for triplet in triplets
    )


def format_triplets_json(triplets: list[Triplet]) -> str:
    return json.dumps(
        [triplet.model_dump() for triplet in triplets],
        ensure_ascii=False,
    )


class MarkdownTripletAdapter(BaseAlpacaAdapter[str, TripletList]):
    def convert(self, passages: str, output: TripletList) -> list[AlpacaDataset]:
        markdown_triplets = format_triplets_markdown(output.triplets)
        return [
            AlpacaDataset(
                instruction="Extract knowledge graph triplets from this passage in markdown format.",
                input=passages,
                output=markdown_triplets,
            ),
            AlpacaDataset(
                instruction="List the subject-relation-object triplets from this passage as markdown bullet points.",
                input=passages,
                output=markdown_triplets,
            ),
        ]


class JsonTripletAdapter(BaseAlpacaAdapter[str, TripletList]):
    def convert(self, passages: str, output: TripletList) -> list[AlpacaDataset]:
        json_triplets = format_triplets_json(output.triplets)
        return [
            AlpacaDataset(
                instruction="Extract knowledge graph triplets from this passage and return them as JSON.",
                input=passages,
                output=json_triplets,
            ),
            AlpacaDataset(
                instruction="Return a JSON array of subject-relation-object triplets supported by this passage.",
                input=passages,
                output=json_triplets,
            ),
        ]


class TripletAugmentation(BaseSingleChunkAugmentation[TripletList]):
    schema = TripletList
    system_prompt = SYSTEM_PROMPT
    response_formats = (
        AlpacaResponseFormat(
            name="markdown",
            adapter=MarkdownTripletAdapter(),
            format_instruction=(
                "Represent the triplets as markdown bullet points."
            ),
        ),
        AlpacaResponseFormat(
            name="json",
            adapter=JsonTripletAdapter(),
            format_instruction=(
                "Represent the triplets as a JSON array of objects with keys "
                "'subject', 'relation', and 'object'."
            ),
        ),
    )
    temperature = 0.2


triplet_augmentation = TripletAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        triplet_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer uses attention mechanisms.
The model achieves strong results on machine translation tasks.
The architecture removes recurrence and convolutions.
        """
    )

    print(len(dataset))
