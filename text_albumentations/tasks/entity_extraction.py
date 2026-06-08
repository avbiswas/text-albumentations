import json
from typing import Literal

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


EntityType = Literal[
    "person",
    "organization",
    "location",
    "method",
    "task",
    "dataset",
    "metric",
    "event",
    "concept",
    "other",
]


class Entity(BaseModel):
    name: str = Field(max_length=120)
    type: EntityType
    context: str = Field(max_length=300)


class EntityList(BaseModel):
    entities: list[Entity] = Field(min_length=1, max_length=12)


SYSTEM_PROMPT = """
Given this passage, extract important named entities and domain concepts. For each entity, choose the best type from the allowed labels and write a short passage-grounded context. Only generate answers.
    """

EXTRACT_INSTRUCTION = (
    "Extract the important entities and concepts from this passage. "
    "Return a JSON array with name, type, and context."
)
TYPE_INSTRUCTION = "Classify the entity from this passage using the allowed entity types."


class EntityExtractionAdapter(BaseAlpacaAdapter[str, EntityList]):
    def convert(self, passages: str, output: EntityList) -> list[AlpacaDataset]:
        rows = [
            AlpacaDataset(
                instruction=EXTRACT_INSTRUCTION,
                input=passages,
                output=json.dumps(
                    [entity.model_dump() for entity in output.entities],
                    ensure_ascii=False,
                ),
            )
        ]
        rows.extend(
            AlpacaDataset(
                instruction=TYPE_INSTRUCTION,
                input=f"Passage:\n{passages}\n\nEntity: {entity.name}",
                output=entity.type,
            )
            for entity in output.entities
        )
        return rows


class EntityExtractionAugmentation(BaseSingleChunkAugmentation[EntityList]):
    schema = EntityList
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage names people, organizations, methods, datasets, metrics, events, or concepts."
    adapters = (EntityExtractionAdapter(),)
    temperature = 0.2
    instruction_templates = {
        EXTRACT_INSTRUCTION: (
            EXTRACT_INSTRUCTION,
            "Extract important entities and concepts from the passage as JSON with name, type, and context.",
            "Return a JSON array of the passage's key entities and concepts with their types and contexts.",
            "Identify the important entities in this passage and output JSON records with name, type, and context.",
            "List passage-grounded entities and concepts as JSON objects with name, type, and context.",
        ),
        TYPE_INSTRUCTION: (
            TYPE_INSTRUCTION,
            "Choose the allowed entity type for this entity based on the passage.",
            "Classify this passage entity using one of the allowed entity type labels.",
            "Return the entity type that best fits this entity in the passage.",
            "Label the entity with its passage-grounded type.",
        ),
    }


entity_extraction_augmentation = EntityExtractionAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        entity_extraction_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer uses attention mechanisms for sequence transduction tasks.
        """
    )
    print(len(dataset))
