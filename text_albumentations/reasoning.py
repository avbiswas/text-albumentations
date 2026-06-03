from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset


class ReasoningTrace(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step reasoning trace explaining the logical thought process",
    )


REASONING_SYSTEM_PROMPT = """\
Given a passage and a response derived from it, produce the reasoning chain \
that logically connects them.

The reasoning should trace the deductions that lead from the passage content \
to each element of the response. Explain *why* each part of the response \
follows from the passage, not *how* one would go about constructing it.

Tone: impersonal, observational, deductive. Write as pure reasoning — no \
first-person pronouns, no agent language, no references to a task or process.

<example>
Passage: Water freezes at 0°C and boils at 100°C at standard pressure.
Response: - Freezing point: 0°C
- Boiling point: 100°C

Good reasoning:
The passage states two phase-transition thresholds for water under standard \
conditions. The first is the freezing point at 0°C, and the second is the \
boiling point at 100°C. Both values are explicit in the text and require no \
inference. Each maps directly to a bullet point.

Bad reasoning (do not do this):
I need to extract the key facts from the passage. First, I identify that the \
passage mentions freezing at 0°C. Then I note boiling at 100°C. I create a \
bullet for each one.
</example>

Do NOT include:
- First-person pronouns (I, we, my)
- Task-execution language ("extract", "identify the key points", "summarize")
- References to the instruction, prompt, task, or user request
- Formatting commentary ("as a bullet list", "in JSON")
- Meta-descriptions of what is being done ("The response lists...")

Instead, trace the logical dependencies: what the passage establishes, what \
follows from it, and why the response captures exactly those elements.

Return only the reasoning trace. No preamble, no sign-off.
"""


def _build_reasoning_messages(
    passage: str,
    instruction: str,
    response_output: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original Passage:\n{passage}\n\n"
                f"Instruction:\n{instruction}\n\n"
                f"Response:\n{response_output}"
            ),
        },
    ]


def generate_reasoning(
    passage: str,
    row: AlpacaDataset,
    runtime: ModelRuntime,
) -> AlpacaDataset:
    messages = _build_reasoning_messages(passage, row.instruction, row.output)
    trace = runtime.generate_structured(
        messages,
        ReasoningTrace,
        temperature=0.2,
        max_tokens=1000,
    )
    return AlpacaDataset(
        instruction=row.instruction,
        input=row.input,
        output=row.output,
        reasoning=trace.reasoning,
    )


async def agenerate_reasoning(
    passage: str,
    row: AlpacaDataset,
    runtime: ModelRuntime,
) -> AlpacaDataset:
    messages = _build_reasoning_messages(passage, row.instruction, row.output)
    trace = await runtime.agenerate_structured(
        messages,
        ReasoningTrace,
        temperature=0.2,
        max_tokens=1000,
    )
    return AlpacaDataset(
        instruction=row.instruction,
        input=row.input,
        output=row.output,
        reasoning=trace.reasoning,
    )


def add_reasoning_to_dataset(
    passage: str,
    dataset: list[AlpacaDataset],
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    return [generate_reasoning(passage, row, runtime) for row in dataset]


async def aadd_reasoning_to_dataset(
    passage: str,
    dataset: list[AlpacaDataset],
    runtime: ModelRuntime,
) -> list[AlpacaDataset]:
    results = await asyncio.gather(
        *[agenerate_reasoning(passage, row, runtime) for row in dataset]
    )
    return list(results)
