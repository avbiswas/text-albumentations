from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class MethodStep(BaseModel):
    step: str = Field(max_length=250)


class MethodSteps(BaseModel):
    process_name: str = Field(max_length=120)
    steps: list[MethodStep] = Field(min_length=2, max_length=8)


SYSTEM_PROMPT = """
Given this passage, name the process, workflow, method, or algorithm and extract it as ordered steps when the passage supports it. Keep each step faithful to the passage and do not invent missing steps. Only generate answers.
    """

INSTRUCTION = (
    "Convert the process or method described in this passage into ordered steps."
)
PROCESS_NAME_INSTRUCTION = "Name the process, method, workflow, or algorithm described in this passage."
NEXT_STEP_INSTRUCTION = "Given the completed steps so far, write the next step."
MISSING_STEP_INSTRUCTION = "Fill in the missing step in this ordered process."


def _format_steps(steps: list[MethodStep]) -> str:
    return "\n".join(
        f"{idx}. {step.step}"
        for idx, step in enumerate(steps, start=1)
    )


class MethodStepsAdapter(BaseAlpacaAdapter[str, MethodSteps]):
    def convert(self, passages: str, output: MethodSteps) -> list[AlpacaDataset]:
        rows = [
            AlpacaDataset(
                instruction=INSTRUCTION,
                input=passages,
                output=_format_steps(output.steps),
            ),
            AlpacaDataset(
                instruction=PROCESS_NAME_INSTRUCTION,
                input=passages,
                output=output.process_name,
            ),
        ]
        for index, step in enumerate(output.steps):
            if index > 0:
                rows.append(
                    AlpacaDataset(
                        instruction=NEXT_STEP_INSTRUCTION,
                        input=(
                            f"Process: {output.process_name}\n\n"
                            f"Completed steps:\n{_format_steps(output.steps[:index])}"
                        ),
                        output=step.step,
                    )
                )
            masked_steps = [
                MethodStep(step="[MISSING STEP]") if masked_index == index else masked_step
                for masked_index, masked_step in enumerate(output.steps)
            ]
            rows.append(
                AlpacaDataset(
                    instruction=MISSING_STEP_INSTRUCTION,
                    input=(
                        f"Process: {output.process_name}\n\n"
                        f"Steps:\n{_format_steps(masked_steps)}"
                    ),
                    output=step.step,
                )
            )
        return rows


class MethodStepsAugmentation(BaseSingleChunkAugmentation[MethodSteps]):
    schema = MethodSteps
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage explains a process, method, workflow, procedure, or algorithm."
    adapters = (MethodStepsAdapter(),)
    temperature = 0.3
    instruction_templates = {
        INSTRUCTION: (
            INSTRUCTION,
            "Extract the process described in this passage as ordered steps.",
            "Write the method from this passage as a numbered sequence of steps.",
            "Convert the passage's workflow into ordered steps.",
            "List the procedure described in the passage in order.",
        ),
        PROCESS_NAME_INSTRUCTION: (
            PROCESS_NAME_INSTRUCTION,
            "Name the process or method described in this passage.",
            "Identify the workflow, process, or algorithm in this passage.",
            "Give a concise name for the process described here.",
            "State the name of the method or procedure in the passage.",
        ),
        NEXT_STEP_INSTRUCTION: (
            NEXT_STEP_INSTRUCTION,
            "Write the next step after the completed steps shown.",
            "Continue the ordered process with the next step only.",
            "Given these completed steps, provide the following step.",
            "Predict the next step in this process.",
        ),
        MISSING_STEP_INSTRUCTION: (
            MISSING_STEP_INSTRUCTION,
            "Replace [MISSING STEP] with the correct step in this ordered process.",
            "Fill the missing step using the surrounding ordered steps.",
            "Provide only the step that belongs where [MISSING STEP] appears.",
            "Recover the omitted step from this process.",
        ),
    }


method_steps_augmentation = MethodStepsAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        method_steps_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
First tokenize the text, then encode the tokens, and finally decode the output sequence.
        """
    )
    print(len(dataset))
