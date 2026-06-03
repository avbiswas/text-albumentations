from __future__ import annotations

import pytest
from pydantic import BaseModel

from text_albumentations.runtime import ModelRuntime

PASSAGE = (
    "The dominant sequence transduction models are based on complex recurrent "
    "or convolutional neural networks in an encoder-decoder configuration. "
    "The best performing models also connect the encoder and decoder through "
    "an attention mechanism. We propose a new simple network architecture, "
    "the Transformer, based solely on attention mechanisms, dispensing with "
    "recurrence and convolutions entirely. Experiments on two machine "
    "translation tasks show these models to be superior in quality while "
    "being more parallelizable and requiring significantly less time to train."
)


class FakeModel(ModelRuntime):
    """Offline model: returns canned responses keyed by schema name.

    Dynamic schemas (e.g. ConfiguredSummary built via create_model) share
    canned responses with their base schema by stripping the 'Configured'
    prefix.
    """

    def __init__(self, responses: dict[str, BaseModel]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    def _lookup(self, output_type: type) -> BaseModel:
        name = output_type.__name__.removeprefix("Configured")
        if name not in self.responses:
            raise KeyError(f"FakeModel has no canned response for schema '{name}'")
        return self.responses[name]

    def generate_structured(self, messages, output_type, *, temperature=0.2, max_tokens=5000):
        self.calls.append((messages, output_type.__name__))
        canned = self._lookup(output_type)
        return output_type.model_validate(canned.model_dump())

    def generate_variation(self, output, output_type, *, context=None, temperature=0.5, max_tokens=5000):
        return output_type.model_validate(self._lookup(output_type).model_dump())


@pytest.fixture
def passage() -> str:
    return PASSAGE
