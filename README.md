# text-albumentations

`text-albumentations` is a synthetic data generation engine for text.

PyPI: https://pypi.org/project/text-albumentations/

The goal is to help generate instruction-tuning and distillation datasets from existing text corpora by applying structured augmentations over passages.

This is built for the practical case where good supervised fine-tuning often requires more examples than you already have, and where synthetic data generation is one of the fastest ways to create task-shaped training data from raw documents.

## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

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

- **auto-pick**: LLM-driven selection of which augmentations fit a given passage
- **quality filter**: automatic rejection of low-quality passages before generation
- **reasoning traces**: post-hoc CoT reasoning generated for each training row
- single-chunk augmentations
- multi-chunk augmentations
- batched augmentation execution for many passages with one shared schema
- typed structured outputs with Pydantic
- Alpaca-format dataset generation
- response-format control for the Alpaca `output` field
- sync and async generation runtimes
- Outlines-backed local models (MLX, Transformers)
- Outlines-backed OpenAI models (sync and async)
- OpenAI-compatible local servers (e.g. MLX server on localhost)
- long-text ingestion with fixed-size character chunking
- JSONL dataset writing

Built-in augmentations:

| Augmentation | Type | What it generates |
| --- | --- | --- |
| `bullets` | Single chunk | Extracts key points from a passage and renders them as bullet-style outputs. |
| `qa_pairs` | Single chunk | Produces question-answer pairs grounded in one passage. |
| `rephrase` | Single chunk | Rewrites a passage into a clearer or more elaborated version without changing meaning. |
| `continuation` | Single chunk | Produces continuation-style completions derived from the passage. |
| `triplets` | Single chunk | Extracts subject-relation-object knowledge graph triplets. |
| `comparison` | Multi chunk | Compares two passages and generates a structured comparison. |
| `retrieval` | Multi chunk | Builds retrieval-style supervision by pairing questions with the passage that answers them, or with no-answer cases. |

## Architecture

The main abstractions are:

- `BaseSingleChunkAugmentation` and `BaseMultiChunkAugmentation`
  These define the task contract: schema, prompt, response formats, generation knobs, and dataset construction.

- `BaseResponseFormat`
  This controls how the Alpaca `output` field should be represented and can also modify the system prompt with format-specific instructions.

- `BaseAlpacaAdapter`
  This converts typed structured outputs into Alpaca rows.

- `ModelRuntime`
  This is the model execution interface. Current implementations support local Outlines models, OpenAI-through-Outlines models, and OpenAI-compatible local servers.

- `AugmentationRunner`
  This binds together: input data, a runtime, and an augmentation.

- `MetaAugmentation`
  Auto-picks which augmentations to apply and filters low-quality passages.

## Usage

### Install

```bash
uv add text-albumentations
```

PyPI package: https://pypi.org/project/text-albumentations/

### Recommended: Auto-Pick With Quality Filter

The recommended way to generate datasets is `apply_best_augmentations`. It uses an LLM to automatically select which augmentations fit your passage and filters out low-quality input.

```python
import openai
import outlines

from text_albumentations import OutlinesModel, apply_best_augmentations
from text_albumentations.tasks.bullets import BulletAugmentation
from text_albumentations.tasks.qa_pairs import QaPairAugmentation
from text_albumentations.tasks.rephrase import RephraseAugmentation
from text_albumentations.tasks.triplets import TripletAugmentation

model = outlines.from_openai(openai.OpenAI(), "gpt-4o-mini")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")

rows = apply_best_augmentations(
    "The Transformer replaces recurrence with attention and improves parallelization. "
    "It achieved 28.4 BLEU on WMT 2014 English-to-German.",
    [
        ("bullets", BulletAugmentation(), "Extract key points as bullet points"),
        ("qa_pairs", QaPairAugmentation(), "Generate question-answer pairs"),
        ("rephrase", RephraseAugmentation(), "Rephrase and elaborate the passage"),
        ("triplets", TripletAugmentation(), "Extract knowledge graph triplets"),
    ],
    runtime,
)

for row in rows:
    print(row.model_dump_json())
```

The LLM first assesses passage quality (rejecting too-short, nonsensical, or boilerplate text), then selects only the augmentations that are well-suited to the passage's content and structure.

#### With Reasoning Traces

Add `add_reasoning=True` to generate a Chain-of-Thought reasoning trace for every training row:

```python
rows = apply_best_augmentations(
    passage,
    augmentations,
    runtime,
    enable_quality_filter=True,
    add_reasoning=True,
)
```

Each output row gets a `reasoning` field containing a step-by-step logical trace explaining how the response was derived from the passage and instruction.

#### Auto-Pick With Async

```python
import asyncio
import openai
import outlines

from text_albumentations import OutlinesModel, aapply_best_augmentations
from text_albumentations.tasks.bullets import BulletAugmentation
from text_albumentations.tasks.qa_pairs import QaPairAugmentation

async def main():
    model = outlines.from_openai(openai.AsyncOpenAI(), "gpt-4o-mini")
    runtime = OutlinesModel(model, async_mode=True, total_concurrent_calls=4,
                            max_tokens_parameter="max_completion_tokens")

    rows = await aapply_best_augmentations(passage, augmentations, runtime)
    print(len(rows))

asyncio.run(main())
```

### Runtime Setup

Pick the backend that fits your setup:

**OpenAI (sync)**
```python
model = outlines.from_openai(openai.OpenAI(), "gpt-4o-mini")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")
```

**OpenAI (async)**
```python
model = outlines.from_openai(openai.AsyncOpenAI(), "gpt-4o-mini")
runtime = OutlinesModel(model, async_mode=True, total_concurrent_calls=4,
                        max_tokens_parameter="max_completion_tokens")
```

**OpenAI-compatible local server**
```python
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
model = outlines.from_openai(client, "mlx-community/Qwen3.5-4B-MLX-4bit")
runtime = OutlinesModel(model, max_tokens_parameter="max_completion_tokens")
```

**MLX local model**
```python
import mlx_lm
model = outlines.from_mlxlm(*mlx_lm.load("mlx-community/Qwen3.5-4B-OptiQ-4bit"))
runtime = OutlinesModel(model=model)
```

**Transformers local model**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it",
                                                 torch_dtype="auto", device_map="auto")
hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = outlines.from_transformers(hf_model, hf_tokenizer)
runtime = OutlinesModel(model, max_tokens_parameter="max_new_tokens")
```

### Reasoning Traces (Standalone)

You can add reasoning traces to any existing dataset, even if you didn't generate them with `add_reasoning=True`:

```python
from text_albumentations.reasoning import add_reasoning_to_dataset

rows = run_augmentation(passage, bullet_augmentation, runtime)
rows_with_reasoning = add_reasoning_to_dataset(passage, rows, runtime)
```

Each row gets a `reasoning` field with a Chain-of-Thought trace. Available functions:

| Function | Description |
| --- | --- |
| `generate_reasoning(passage, row, runtime)` | Add reasoning to a single row |
| `add_reasoning_to_dataset(passage, dataset, runtime)` | Add reasoning to all rows |
| `agenerate_reasoning(...)` / `aadd_reasoning_to_dataset(...)` | Async variants |

### Batch Augmentation Over Multiple Passages

```python
from text_albumentations import OutlinesModel, run_batch_augmentation
from text_albumentations.tasks.bullets import BulletAugmentation

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

### Long Text To JSONL

```python
from text_albumentations import OutlinesModel, save_long_text_dataset
from text_albumentations.tasks.bullets import bullet_augmentation

save_long_text_dataset(
    text=long_text,
    output_jsonl="out.jsonl",
    augmentation=bullet_augmentation,
    runtime=runtime,
    chunk_size_chars=300,
)
```

### Advanced: Manual Augmentation Selection

If you want precise control over which augmentations run, use `run_augmentation` directly with individual augmentation instances.

```python
from text_albumentations import OutlinesModel, run_augmentation
from text_albumentations.tasks.bullets import BulletAugmentation

rows = run_augmentation(passage, BulletAugmentation(max_bullets=4, variations=2), runtime)
```

#### Multiple Augmentations Over The Same Passage

```python
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.rephrase import rephrase_augmentation

rows = []
rows.extend(run_augmentation("some passage", bullet_augmentation, runtime))
rows.extend(run_augmentation("some passage", rephrase_augmentation, runtime))
```

#### Augmentation Knobs

Every augmentation accepts these parameters to control generation behavior:

| Parameter | Default | Description |
| --- | --- | --- |
| `temperature` | 0.2 | Sampling temperature for base generation |
| `max_tokens` | 5000 | Max tokens for base generation |
| `num_generations` | 1 | Number of independent base generations |
| `variations` | varies | Number of variations per base generation (uses higher `variation_temperature`) |
| `variation_temperature` | 0.5 | Temperature used for variation generation |

Customize per-augmentation parameters:

```python
aug = BulletAugmentation(
    max_bullets=4,
    temperature=0.3,
    variations=3,
    variation_temperature=0.7,
)
```

#### Custom Preprocessing Model

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
