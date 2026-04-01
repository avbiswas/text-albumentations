# text-albumentations

`text-albumentations` is a synthetic data generation engine for text.

The goal is to help generate instruction-tuning and distillation datasets from existing text corpora by applying structured augmentations over passages.

This is built for the practical case where good supervised fine-tuning often requires more examples than you already have, and where synthetic data generation is one of the fastest ways to create task-shaped training data from raw documents.

## Why This Exists

Modern LLM workflows often need:

- synthetic SFT data
- task-specific distillation data
- multiple renderings of the same semantic content
- structured supervision generated from long-form text

If you already have long amounts of text, you can usually derive many useful supervision targets from it:

- bullet-point summaries
- QA pairs
- rephrasings
- continuation tasks
- retrieval examples
- comparisons
- knowledge graph triplets

Instead of treating synthetic data generation as one giant prompt, this project breaks it into explicit, composable pieces.


## Ideology

The core idea is:

`structured generation + simple priors -> dataset`

Structured generation gives you typed intermediate outputs using Pydantic schemas.

Simple priors give you the task shape:

- "extract bullets"
- "produce QA pairs"
- "find the answering passage"
- "serialize the response as markdown/json/etc"

That combination is easier to reason about than unstructured free-form prompting. It also makes the pipeline more extensible: you can swap prompts, schemas, response formats, runtimes, and adapters without rewriting the whole system.

## Current Capabilities

The project currently supports:

- single-chunk augmentations
- multi-chunk augmentations
- batched augmentation execution for many passages with one shared schema
- typed structured outputs with Pydantic
- Alpaca-format dataset generation
- response-format control for the Alpaca `output` field
- sync and async generation runtimes
- Outlines-backed local models
- Outlines-backed OpenAI models
- long-text ingestion with fixed-size character chunking
- JSONL dataset writing

Built-in augmentation families include:

- bullets
- QA pairs
- rephrase
- continuation
- retrieval
- comparison
- triplets

## Architecture

The main abstractions are:

- `BaseSingleChunkAugmentation` and `BaseMultiChunkAugmentation`
  These define the task contract: schema, prompt, response formats, generation knobs, and dataset construction.

- `BaseResponseFormat`
  This controls how the Alpaca `output` field should be represented and can also modify the system prompt with format-specific instructions.

- `BaseAlpacaAdapter`
  This converts typed structured outputs into Alpaca rows.

- `ModelRuntime`
  This is the model execution interface. Current implementations support local Outlines models and OpenAI-through-Outlines models.

- `AugmentationRunner`
  This binds together:
  1. input data
  2. a runtime
  3. an augmentation

## Usage

### Minimal Local Example

```python
import mlx_lm
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation

model = outlines.from_mlxlm(*mlx_lm.load("mlx-community/Qwen3.5-4B-OptiQ-4bit"))
runtime = OutlinesModel(model=model)

rows = run_augmentation(
    "The Transformer replaces recurrence with attention and improves parallelization.",
    bullet_augmentation,
    runtime,
)

for row in rows:
    print(row.model_dump_json())
```

See [`examples/minimal.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/minimal.py).

### OpenAI Sync

```python
import openai
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation

model = outlines.from_openai(openai.OpenAI(), "gpt-5.4-nano")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

rows = run_augmentation("some passage", bullet_augmentation, runtime)
```

See [`examples/openai_sync.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/openai_sync.py).

### OpenAI Async

```python
import asyncio
import openai
import outlines

from text_albumentations import OutlinesModel, arun_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation


async def main():
    model = outlines.from_openai(openai.AsyncOpenAI(), "gpt-5.4-nano")
    runtime = OutlinesModel(
        model,
        async_mode=True,
        total_concurrent_calls=4,
        max_tokens_parameter="max_completion_tokens",
    )

    rows = await arun_augmentation("some passage", bullet_augmentation, runtime)
    print(len(rows))


asyncio.run(main())
```

See [`examples/openai_async.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/openai_async.py).

### Transformers Local Model

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype="auto",
    device_map="auto",
)
hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

model = outlines.from_transformers(hf_model, hf_tokenizer)
runtime = OutlinesModel(model, max_tokens_parameter="max_new_tokens")

rows = run_augmentation("some passage", bullet_augmentation, runtime)
```

See the `examples/` directory for the current Transformers examples.

### Batch Augmentation Over Multiple Passages

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_albumentations import OutlinesModel, run_batch_augmentation
from text_albumentations.tasks.bullets import BulletAugmentation

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype="auto",
    device_map="auto",
)
hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

model = outlines.from_transformers(hf_model, hf_tokenizer)
runtime = OutlinesModel(model, max_tokens_parameter="max_new_tokens")
augmentation = BulletAugmentation(max_tokens=128, variations=0)

rows = run_batch_augmentation(
    [
        "The Transformer replaces recurrence with attention and improves parallelization.",
        "Outlines constrains generation so outputs match the expected structure.",
        "Synthetic supervision can be derived from raw documents with task-shaped prompts.",
        "Batch decoding is useful when many passages share the same schema and augmentation.",
    ],
    augmentation,
    runtime,
)
```

See [`examples/batch_augmentation.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/batch_augmentation.py).

### Long Text To JSONL

```python
import openai
import outlines

from text_albumentations import OutlinesModel, save_long_text_dataset
from text_albumentations.tasks.bullets import bullet_augmentation

model = outlines.from_openai(openai.OpenAI(), "gpt-5.4-nano")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

save_long_text_dataset(
    text=long_text,
    output_jsonl="out.jsonl",
    augmentation=bullet_augmentation,
    runtime=runtime,
    chunk_size_chars=300,
)
```

See [`examples/long_text_to_jsonl.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/long_text_to_jsonl.py).

### Multiple Augmentations Over The Same Passage

```python
import openai
import outlines

from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.rephrase import rephrase_augmentation

model = outlines.from_openai(openai.OpenAI(), "gpt-5.4-nano")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

rows = []
rows.extend(run_augmentation("some passage", bullet_augmentation, runtime))
rows.extend(run_augmentation("some passage", rephrase_augmentation, runtime))
```

See [`examples/multiple_augmentations.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/multiple_augmentations.py).

### Custom Preprocessing Model

You can also make the augmentation input itself be a custom Pydantic model instead of a raw string.

See [`examples/custom_preprocessing.py`](/Users/avishekbiswas/Projects/text-albumentations/examples/custom_preprocessing.py).

## Extensibility

The project is designed so users can extend it in layers.

### 1. Add A New Augmentation

Subclass one of:

- `BaseSingleChunkAugmentation`
- `BaseMultiChunkAugmentation`

Define:

- a Pydantic schema
- a system prompt
- `build_user_message(...)`
- one or more response formats

### 2. Add A New Response Format

Subclass `BaseResponseFormat` if you want to control:

- how the format modifies the system prompt
- how the final Alpaca `output` field is rendered

For common Alpaca row generation, `AlpacaResponseFormat` is usually enough.

### 3. Add A New Adapter

Subclass `BaseAlpacaAdapter` to convert a typed structured output into one or more Alpaca rows.

One structured output can expand into multiple rows.

### 4. Add A New Runtime

Implement `ModelRuntime` if you want to support a new backend.

That keeps model execution separate from:

- augmentation semantics
- prompt construction
- dataset adapters
- response serialization

This separation is intentional. The project should let you swap the model layer without rewriting the dataset logic.

## Philosophy On Synthetic Data

This project does not assume synthetic data is magic.

It assumes:

- synthetic data works best when the task shape is explicit
- typed intermediate representations are easier to control
- simple priors beat vague giant prompts
- extensibility matters because different teams want different schemas, formats, and runtimes

The aim is not "generate random data."

The aim is to turn raw text into useful supervision signals for SFT and distillation in a way that is structured, inspectable, and easy to extend.
