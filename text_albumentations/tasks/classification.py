from typing import Literal

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset

Tone = Literal[
    "formal",
    "informal",
    "technical",
    "persuasive",
    "narrative",
    "instructional",
]

Audience = Literal[
    "general public",
    "domain experts",
    "students",
    "professionals",
    "children",
]


class PassageLabels(BaseModel):
    topic: str = Field(max_length=80)
    tone: Tone
    audience: Audience


SYSTEM_PROMPT = """
Given this passage, classify it. Identify its main topic in a few words, its tone, and its most likely intended audience. Base your labels only on the passage content.

Tone options:
- formal: polished, impersonal register
- informal: conversational, casual register
- technical: specialized terminology aimed at practitioners or researchers
- persuasive: argues for a position or action
- narrative: tells a story or recounts events
- instructional: teaches or gives step-by-step guidance

Audience options:
- general public: no background assumed
- domain experts: assumes specialist knowledge of the field
- students: learners studying the subject
- professionals: practitioners applying the subject at work
- children: young readers

Only generate answers.
    """

TOPIC_INSTRUCTION = "Identify the main topic of this passage in a few words."
TONE_INSTRUCTION = (
    "Classify the tone of this passage as one of: "
    "formal, informal, technical, persuasive, narrative, instructional."
)
AUDIENCE_INSTRUCTION = (
    "Classify the most likely intended audience of this passage as one of: "
    "general public, domain experts, students, professionals, children."
)


class ClassificationAdapter(BaseAlpacaAdapter[str, PassageLabels]):
    def convert(self, passages: str, output: PassageLabels) -> list[AlpacaDataset]:
        return [
            AlpacaDataset(
                instruction=TOPIC_INSTRUCTION,
                input=passages,
                output=output.topic,
            ),
            AlpacaDataset(
                instruction=TONE_INSTRUCTION,
                input=passages,
                output=output.tone,
            ),
            AlpacaDataset(
                instruction=AUDIENCE_INSTRUCTION,
                input=passages,
                output=output.audience,
            ),
        ]


class ClassificationAugmentation(BaseSingleChunkAugmentation[PassageLabels]):
    schema = PassageLabels
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage is any coherent text; yields cheap topic/tone/audience labels."
    adapters = (ClassificationAdapter(),)
    temperature = 0.1
    max_tokens = 1024


classification_augmentation = ClassificationAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        classification_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
             """
    )

    print(len(dataset))
