from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from datasets import load_dataset

from text_albumentations.quality import QualityAssessment


DATASET_NAME = "paperbd/paper_instructions_300K-v1"
MODEL_NAME = "mlx-community/Qwen3.5-4B-MLX-4bit"
BASE_URL = "http://localhost:8080/v1"
DEFAULT_OUTPUT_DIR = Path("filtered_datasets/paper_instructions_300K-v1_quality_filtered")
DEFAULT_PROMPT = """\
A quality paper-instruction datapoint should be useful for supervised fine-tuning.
Keep the datapoint only if all of the following are true:
- The instruction is clear and answerable from the input.
- The input contains enough natural-language context to support the output.
- The output directly satisfies the instruction.
- The output is grounded in the input and does not introduce unsupported claims.
- The datapoint is not truncated in a way that makes the instruction impossible.
- The row is not boilerplate, markup-only, nonsensical, duplicated fragments, or malformed.

Reject the datapoint if the input/output pair is contradictory, incomplete, too vague,
too noisy, or not useful as a training example.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter paperbd/paper_instructions_300K-v1 with ta.quality_filter."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--response-format",
        choices=["json_object", "json_schema", "none"],
        default="json_object",
    )
    parser.add_argument(
        "--no-require-parameters",
        action="store_true",
        help=(
            "Allow OpenRouter to route to providers even if they do not "
            "advertise support for requested parameters such as response_format."
        ),
    )
    return parser.parse_args()


def read_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return DEFAULT_PROMPT
    return prompt_file.read_text()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as file:
        return sum(1 for _ in file)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Model response did not contain JSON: {text!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError(f"Model response contained incomplete JSON: {text!r}")


async def judge_row(
    row_index: int,
    row: dict[str, Any],
    prompt: str,
    *,
    client: AsyncOpenAI,
    model: str,
    response_format: str,
    require_parameters: bool,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request_response_format = None
            if response_format == "json_object":
                request_response_format = {"type": "json_object"}
            elif response_format == "json_schema":
                request_response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QualityAssessment",
                        "strict": True,
                        "schema": QualityAssessment.model_json_schema(),
                    },
                }

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data quality judge. Decide whether the datapoint "
                            "satisfies the user's quality criteria. Return only compact "
                            'JSON matching this schema: {"is_quality": boolean, '
                            '"reason": string}. The reason must be 25 words or fewer.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Quality criteria:\n{prompt.strip()}\n\n"
                            f"Datapoint:\n{json.dumps(row, ensure_ascii=False, indent=2)}"
                        ),
                    },
                ],
                response_format=request_response_format,
                temperature=0,
                max_tokens=128,
                extra_body=(
                    {"provider": {"require_parameters": True}}
                    if require_parameters
                    else None
                ),
            )
            if not response.choices:
                raise ValueError(f"Model response had no choices: {response}")
            content = response.choices[0].message.content or ""
            assessment = QualityAssessment.model_validate(extract_json_object(content))
            break
        except Exception as error:
            last_error = error
            if attempt == 2:
                return {
                    "row_index": row_index,
                    "is_quality": False,
                    "reason": f"Quality judge failed after retries: {error}",
                    "judge_error": True,
                    "row": row,
                }
            await asyncio.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError("quality assessment failed") from last_error

    return {
        "row_index": row_index,
        "is_quality": assessment.is_quality,
        "reason": assessment.reason,
        "row": row,
    }


async def filter_split(
    split: str,
    *,
    dataset_name: str,
    output_dir: Path,
    prompt: str,
    client: AsyncOpenAI,
    model: str,
    response_format: str,
    require_parameters: bool,
    concurrency: int,
    max_rows: int | None,
) -> None:
    kept_path = output_dir / f"{split}.jsonl"
    rejected_path = output_dir / f"{split}.rejected.jsonl"
    decisions_path = output_dir / f"{split}.decisions.jsonl"

    processed = count_jsonl(decisions_path)
    print(f"[{split}] resuming after {processed} processed rows")

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    semaphore = asyncio.Semaphore(concurrency)
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def schedule(row_index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            return await judge_row(
                row_index,
                row,
                prompt,
                client=client,
                model=model,
                response_format=response_format,
                require_parameters=require_parameters,
            )

    async def drain_one() -> None:
        done, pending_remaining = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        pending.clear()
        pending.update(pending_remaining)
        for task in done:
            decision = task.result()
            write_jsonl(decisions_path, decision)
            if decision["is_quality"]:
                write_jsonl(kept_path, decision["row"])
            else:
                write_jsonl(rejected_path, decision)
            if decision["row_index"] % 100 == 0:
                print(
                    f"[{split}] processed row_index={decision['row_index']} "
                    f"quality={decision['is_quality']}"
                )

    for row_index, row in enumerate(dataset):
        if row_index < processed:
            continue
        if max_rows is not None and row_index >= processed + max_rows:
            break

        pending.add(asyncio.create_task(schedule(row_index, dict(row))))
        if len(pending) >= concurrency:
            await drain_one()

    while pending:
        await drain_one()

    print(
        f"[{split}] done kept={count_jsonl(kept_path)} "
        f"rejected={count_jsonl(rejected_path)} decisions={count_jsonl(decisions_path)}"
    )


async def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = read_prompt(args.prompt_file)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    metadata = {
        "source_dataset": args.dataset,
        "model": args.model,
        "base_url": args.base_url,
        "splits": args.splits,
        "response_format": args.response_format,
        "require_parameters": not args.no_require_parameters,
        "prompt": prompt,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    for split in args.splits:
        await filter_split(
            split,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            prompt=prompt,
            client=client,
            model=args.model,
            response_format=args.response_format,
            require_parameters=not args.no_require_parameters,
            concurrency=args.concurrency,
            max_rows=args.max_rows_per_split,
        )


if __name__ == "__main__":
    asyncio.run(main())
