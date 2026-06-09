# text-albumentations

`text-albumentations` is a synthetic data generation engine for text.
The goal is to help generate instruction-tuning and distillation datasets from existing text corpora by applying structured augmentations over passages.

This is built for the practical case where good supervised fine-tuning often requires more examples than you already have, and where synthetic data generation is one of the fastest ways to create task-shaped training data from raw documents.

![text-albumentations pipeline](assets/pipeline.svg)


If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

## Quickstart

Install via pip (or uv):
```
pip install text_albumentations
```

And use:

```python
import text_albumentations as ta

# A model is its own thing — build it once, point it anywhere:
model = ta.OpenAIModel(
    "gpt-5-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

# A smart switch picks the right augmentations and prefilters low-quality passages:
rows = ta.augment("Your passage of text here...", model=model)

# Or choose tasks explicitly:
rows = ta.augment(
    "Your passage of text here...",
    tasks=["summarize", "qa_pairs", "title"],
    model=model,
)

# Or auto-pick only from a whitelist:
rows = ta.augment(
    "Your passage of text here...",
    tasks=["summarize", "qa_pairs", "extractive_qa"],
    selection_mode="auto",
    model=model,
)

ta.save(rows, "train.jsonl")   # appends Alpaca-format rows as JSONL
```

Model primitives:

- `ta.OpenAIModel(...)` — any OpenAI-compatible endpoint (OpenAI, local MLX server, vLLM, Ollama). Falls back to the `TEXT_ALBUMENTATIONS_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` environment variables, so a configured shell needs only `ta.augment(text)`.
- `ta.LocalMLXModel("mlx-community/...")` — an MLX model loaded in-process (Apple Silicon).
- `ta.LocalHFModel("Qwen/Qwen3.5-2B")` — a Hugging Face Transformers model loaded in-process.

By default, the final Alpaca `instruction` strings are sampled from curated
per-task template pools. This adds wording variety across rows without changing
the internal generation prompts, `input`, `output`, or row count. Pass
`sample_instruction_template=False` when you need the exact default instruction
strings.

`ta.list_tasks()` returns every built-in single-passage task name with a one-line hint of when it fits. `ta.get_task(...)` and `ta.resolve_tasks(...)` return the augmentation objects when downstream builders want to own scheduling. When you need more control, `tasks=` also accepts configured augmentation instances in explicit mode (e.g. `BulletAugmentation(max_bullets=4)`) — everything below stays available.

Defining your own task takes a schema and a prompt:

```python
from pydantic import BaseModel, Field

class KeyStat(BaseModel):
    statistic: str = Field(max_length=200)
    context: str = Field(max_length=300)

key_stat = ta.task(
    prompt="Extract the single most important statistic from this passage.",
    schema=KeyStat,
    output="{statistic} — {context}",   # template over schema fields
    instruction_variants=[
        "Identify the key statistic in this passage.",
        "Extract the most important number from this passage.",
    ],
)

rows = ta.augment(passage, tasks=[key_stat, "summarize"], model=model)
```

The schema is enforced exactly via structured generation. `output=` also takes a callable, `rows=` gives full control over emitted rows, and generation knobs (`temperature=`, `variations=`, ...) pass straight through. Custom tasks can provide `instruction_variants=` for final-row instruction sampling. Subclassing `BaseSingleChunkAugmentation` remains available for tasks that need custom input types or programmatic generation.


## Why This Exists

Modern LLM workflows often need:

- synthetic SFT data
- task-specific distillation data
- multiple renderings of the same semantic content
- structured supervision generated from long-form text

If you already have long amounts of text, you can usually derive many useful supervision targets from it:

- bullet-point summaries
- QA pairs (free-form and extractive)
- rephrasings and style transfers
- summaries, titles, and headlines
- continuation and cloze (fill-in-the-blank) tasks
- retrieval examples
- comparisons
- knowledge graph triplets
- classification labels
- backtranslated instructions and counterfactuals

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

That combination is easier to reason about than unstructured free-form prompting. It also makes the pipeline more extensible: you can swap prompts, schemas, response formats, models, and adapters without rewriting the whole system.

## Current Capabilities

The project currently supports:

- **one-call generation**: `ta.augment(text, model=model)` is the whole pipeline
- **auto-pick (smart switch)**: LLM-driven selection of which augmentations fit a given passage, guided by per-task `selection_hint`s and grammar-constrained to real task names
- **prefilter**: lightweight dedicated LLM call (`PassageQuality`) that rejects low-quality passages before any generation runs — active in both `auto` and `sample` modes
- **standalone postfilter**: optional one-off judging of individual datapoints with `ta.postfilter(...)`
- **reasoning traces**: post-hoc CoT reasoning generated for each training row
- **terse custom tasks**: `ta.task(prompt=..., schema=...)` defines a new augmentation without classes
- single-chunk and multi-chunk augmentations
- batched augmentation execution for many passages with one shared schema
- typed structured outputs with Pydantic
- Alpaca-format dataset generation
- response-format control for the Alpaca `output` field
- sync and async generation
- model primitives: `OpenAIModel` (any OpenAI-compatible endpoint), `LocalMLXModel`, `LocalHFModel`
- long-text ingestion with fixed-size character chunking
- JSONL dataset writing

Built-in augmentations:

| Augmentation | What it generates |
| --- | --- |
| `bullets` | Extracts key points from a passage and renders them as bullet-style outputs. |
| `qa_pairs` | Produces question-answer pairs grounded in one passage. |
| `rephrase` | Rewrites a passage into a clearer or more elaborated version without changing meaning. |
| `continuation` | Programmatically slices the passage into prefix/suffix continuation rows (LLM-free). |
| `triplets` | Extracts subject-relation-object knowledge graph triplets. |
| `summarize` | Produces a one-sentence TLDR and a short prose summary of the passage. |
| `title` | Generates a short descriptive title and a one-sentence headline. |
| `cloze` | Programmatically masks salient words and a middle sentence, producing fill-in-the-blank rows (LLM-free). |
| `extractive_qa` | Generates questions answered by verbatim quotes from the passage; quotes are verified against the source before a row is kept. |
| `classification` | Labels the passage with topic, tone, and intended audience via constrained generation. |
| `style_transfer` | Rewrites the passage in a target style (`eli5`, `formal`, `casual`, or a custom description). |
| `backtranslation` | Generates the instruction the passage would be the ideal answer to (instruction backtranslation); the passage itself becomes the output. |
| `counterfactual` | Alters a central claim of the passage and asks what would follow, with a passage-grounded answer. |
| `evidence_selection` | Builds claim-to-evidence examples where the model chooses the supporting quote from candidate quotes. |
| `claim_verification` | Generates supported, contradicted, or not-enough-information claim verification rows. |
| `entity_extraction` | Extracts named entities and domain concepts with typed labels and short contexts. |
| `definition_extraction` | Extracts passage-grounded term definitions with verified supporting quotes. |
| `method_steps` | Converts process, method, workflow, or algorithm passages into ordered steps. |
| `structured_records` | Extracts subject-attribute-value records as JSON and concise text facts. |
| `section_heading` | Generates local section or subsection headings with short rationales. |
| `query_generation` | Generates search queries that could retrieve the passage and reverse query-to-passage rows. |
| `distractor_qa` | Produces multiple-choice QA rows with plausible distractors and explanations. |
| `error_correction` | Creates corrupted passage variants and correction rows targeting the original passage. |
| `comparison` | Compares two passages and generates a structured comparison. |
| `retrieval` | Builds retrieval-style supervision by pairing questions with the passage that answers them, or with no-answer cases. |

## Architecture

The main abstractions are:

- **Models** (`OpenAIModel`, `LocalMLXModel`, `LocalHFModel`)
  Where generation runs. Build one, pass it everywhere — both `ta.augment` and every lower-level API take the same object. All implement the `ModelRuntime` interface, which you can implement yourself for a new backend.

- **Tasks / Augmentations** (`ta.task(...)`, `BaseSingleChunkAugmentation`, `BaseMultiChunkAugmentation`)
  What to generate from a passage. A task is a Pydantic schema (enforced exactly via structured generation), a `system_prompt` for the augmenter model, a `selection_hint` for the smart switch, and adapters that turn outputs into training rows.

- `BaseAlpacaAdapter`
  Converts one typed structured output into one or more Alpaca rows.

- `BaseResponseFormat`
  Controls how the Alpaca `output` field is represented and can modify the system prompt with format-specific instructions.

- `PassageQuality` / `prefilter_passage` (the quality gate)
  A single tiny structured call (`is_quality: bool`, `max_tokens=20`) that runs before any generation. Used by both `auto` and `sample` modes when `prefilter=True`. Cheap enough to run on every passage without burning generation budget.

- `MetaAugmentation` (the smart switch)
  Auto-picks which augmentations to apply, reading each task's `selection_hint`. Runs only after the passage passes the quality gate. Its task choices are grammar-constrained to the actual task names, so it cannot select something that doesn't exist.

## Usage

### Install

```bash
uv add text-albumentations
```

PyPI package: https://pypi.org/project/text-albumentations/

### Models

Build a model once and pass it everywhere. Every API in the library — from `ta.augment` down to `run_augmentation` — takes the same model object.

**Any OpenAI-compatible endpoint** (OpenAI, local MLX server, vLLM, Ollama):
```python
model = ta.OpenAIModel("gpt-5-mini",
                       base_url="https://api.openai.com/v1",
                       api_key="sk-...")
```

All three arguments fall back to environment variables (`TEXT_ALBUMENTATIONS_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`), so a configured shell can just call `ta.OpenAIModel()` — or skip the model entirely and call `ta.augment(text)`.

`OpenAIModel` chooses a structured-output mode automatically from the model name (`response_format="auto"`). OpenAI/GPT and Gemini models default to strict JSON Schema mode. DeepSeek, Claude, Minimax, GLM, and unknown model families default to JSON-object mode. For every runtime backend, the Pydantic schema is also appended to the system prompt and validated locally after generation.

Override when needed:

```python
model = ta.OpenAIModel(..., response_format="auto")          # default
model = ta.OpenAIModel(..., response_format="json_schema")
model = ta.OpenAIModel(..., response_format="json_object")
```

`OpenAIModel` also passes `reasoning_effort="low"` by default for
OpenAI-compatible reasoning models. OpenAI Chat Completions accepts this as a
top-level parameter; OpenRouter also accepts it, and additionally supports a
provider-neutral `reasoning` object through `extra_body`.

```python
model = ta.OpenAIModel(..., reasoning_effort="low")      # default
model = ta.OpenAIModel(..., reasoning_effort="high")
model = ta.OpenAIModel(..., reasoning_effort=None)       # disable for local servers

model = ta.OpenAIModel(
    ...,
    completion_kwargs={"extra_body": {"reasoning": {"effort": "low"}}},
)
```

Use `completion_kwargs` for any other OpenAI/OpenRouter chat-completion
argument. Values in `completion_kwargs` override the library defaults.

**In-process local models:**
```python
model = ta.LocalMLXModel("mlx-community/Qwen3.5-4B-OptiQ-4bit")   # Apple Silicon
model = ta.LocalHFModel("google/gemma-3-1b-it")                   # Transformers
```

**Google Vertex AI:**

Install the optional dependency and authenticate with Application Default
Credentials (ADC):

```bash
pip install "text-albumentations[vertex]"
gcloud auth application-default login
```

```python
model = ta.VertexAIModel(
    "gemini-2.5-flash",
    project="your-gcp-project",
    location="us-central1",
)
rows = ta.augment(text, tasks=["bullets", "qa_pairs"], model=model)
```

`project` can also come from `GOOGLE_CLOUD_PROJECT` and `location` from
`GOOGLE_CLOUD_LOCATION`. Vertex AI uses Gemini's native structured output.

For async pipelines, `OpenAIModel` takes `async_mode=True` and `total_concurrent_calls=`:
```python
model = ta.OpenAIModel("gpt-5-mini", base_url=..., api_key=...,
                       async_mode=True, total_concurrent_calls=4)
```
The default async OpenAI concurrency is 100. For local OpenAI-compatible
servers, keep this lower unless you have capacity to spare; around 8 is a
reasonable starting point.

A new backend is one class away: implement the `ModelRuntime` interface and pass your object anywhere a model is accepted.

### Recommended: Auto-Pick With Prefilter

The default mode of `ta.augment` is the smart switch. First, a lightweight `PassageQuality` call rejects passages that are too short, nonsensical, or boilerplate — with `max_tokens=20`, it costs almost nothing. If the passage passes, a second call (`MetaAugmentation`) selects only the augmentations well-suited to the passage's content and structure.

```python
import text_albumentations as ta

model = ta.OpenAIModel("gpt-5-mini", base_url=..., api_key=...)

rows = ta.augment(
    "The Transformer replaces recurrence with attention and improves parallelization. "
    "It achieved 28.4 BLEU on WMT 2014 English-to-German.",
    model=model,
)

for row in rows:
    print(row.model_dump_json())
```

#### How the selector decides

Each augmentation carries a `selection_hint` — a one-liner describing *when* the task fits a passage. This is deliberately distinct from `system_prompt`: the hint is read only by the selector LLM, while the system prompt is what the augmenter model sees. The selector's menu looks like:

```
- triplets: pick when the passage states relationships between named entities or concepts.
- extractive_qa: pick when specific facts in the passage can be quoted verbatim as answers.
```

Its choices are grammar-constrained (via a `Literal` over the actual task names), so it cannot hallucinate a task that doesn't exist.

Use `selection_mode="auto"` with `tasks=` to auto-pick only from a whitelist:

```python
rows = ta.augment(
    passage,
    tasks=["qa_pairs", "summarize", "extractive_qa"],
    selection_mode="auto",
    model=model,
    prefilter=True,
)
```

Use `select_tasks` when you want to log the prefilter and task-selection decision before generating rows:

```python
selection = ta.select_tasks(
    passage,
    tasks=["qa_pairs", "summarize", "extractive_qa"],
    model=model,
    prefilter=True,
)

print(selection.selected_tasks)
```

For downstream builders that want to own scheduling, resolve task objects directly:

```python
augmentations = ta.resolve_tasks(["qa_pairs", "summarize", "extractive_qa"])
```

#### With Reasoning Traces

Add `add_reasoning=True` to generate a Chain-of-Thought reasoning trace for every training row:

```python
rows = ta.augment(passage, model=model, add_reasoning=True)
```

Each output row gets a `reasoning` field containing a step-by-step logical trace explaining how the response was derived from the passage and instruction.

#### High-Level Async

```python
import asyncio
import text_albumentations as ta

async def main():
    model = ta.OpenAIModel("gpt-5-mini",
                           base_url="https://api.openai.com/v1",
                           api_key="sk-...",
                           async_mode=True, total_concurrent_calls=4)
    rows = await ta.aaugment(
        passage,
        tasks=["bullets", "qa_pairs"],
        selection_mode="auto",
        model=model,
    )
    print(len(rows))

asyncio.run(main())
```

### Choosing Tasks Explicitly

Pass `tasks=` to skip the smart switch. Names and configured instances mix freely in explicit mode:

```python
from text_albumentations.tasks.bullets import BulletAugmentation

rows = ta.augment(
    passage,
    tasks=["summarize", "qa_pairs", BulletAugmentation(max_bullets=4, variations=2)],
    model=model,
)
```

`ta.list_tasks()` returns every built-in task name with its selection hint. The lower-level equivalent is `run_augmentation(passage, augmentation, model)` for one augmentation at a time.

For stochastic dataset mixtures, pass probabilities with `selection_mode="sample"`. Each task is sampled independently. The lightweight `PassageQuality` prefilter still runs (unless `prefilter=False`), so garbage passages bail before any generation calls. `[]` is returned if no tasks are sampled or the passage fails the quality gate:

```python
rows = ta.augment(
    passage,
    tasks={
        "qa_pairs": 0.25,
        "summarize": 0.25,
        "extractive_qa": 0.25,
        "classification": 0.10,
    },
    selection_mode="sample",
    model=model,
)
```

Instruction-template sampling is enabled by default for `augment`, `aaugment`,
`run_augmentation`, and batch runners. It only changes the emitted Alpaca
`instruction` wording:

```python
rows = ta.augment(passage, tasks=["qa_pairs"], model=model)

rows = ta.augment(
    passage,
    tasks=["qa_pairs"],
    model=model,
    sample_instruction_template=False,  # exact canonical instructions
)
```

Built-in format-specific rows keep their format explicit in every variant, so
markdown rows stay markdown, JSON rows stay JSON, and Python-list rows stay
Python-list rows.

Note: `ta.augment` operates on a single passage, so it covers the single-chunk tasks. The multi-chunk tasks (`comparison`, `retrieval`) take a list of passages and run through `run_augmentation` directly. Use `ta.list_multi_tasks()`, `ta.get_multi_task(...)`, and `ta.resolve_multi_tasks(...)` for stable named access:

```python
from text_albumentations import run_augmentation

rows = run_augmentation([passage_a, passage_b], ta.get_multi_task("comparison"), model)
```

Retrieval can generate many internal model calls because it extracts questions
and then writes positive and no-answer reasons. Use `RetrievalAugmentation` to
cap work for high-throughput runs:

```python
from text_albumentations.tasks.retrieval import RetrievalAugmentation

retrieval = RetrievalAugmentation(
    max_questions_per_passage=2,
    max_passages=8,
    include_negative_examples=False,
)
rows = run_augmentation(passages, retrieval, model)
```

### Reasoning Traces (Standalone)

You can add reasoning traces to any existing dataset, even if you didn't generate them with `add_reasoning=True`:

```python
from text_albumentations.reasoning import add_reasoning_to_dataset

rows = ta.augment(passage, tasks=["bullets"], model=model)
rows_with_reasoning = add_reasoning_to_dataset(passage, rows, model)
```

Each row gets a `reasoning` field with a Chain-of-Thought trace. Available functions:

| Function | Description |
| --- | --- |
| `generate_reasoning(passage, row, model)` | Add reasoning to a single row |
| `add_reasoning_to_dataset(passage, dataset, model)` | Add reasoning to all rows |
| `agenerate_reasoning(...)` / `aadd_reasoning_to_dataset(...)` | Async variants |

### Postfilter (Standalone)

Use `ta.postfilter(...)` to judge one generated training datapoint against your own quality criteria. The datapoint can be a string or JSON-like Python value, and the result is a typed `PostfilterAssessment` with `is_quality` and `reason` fields:

```python
assessment = ta.postfilter(
    {
        "instruction": "Answer the question.",
        "input": "What does the Transformer replace?",
        "output": "It replaces recurrence with attention.",
    },
    prompt="A quality datapoint is correct, grounded in the input, and self-contained.",
    model=model,
)

if assessment.is_quality:
    print("keep")
else:
    print(assessment.reason)
```

Async pipelines can use `await ta.apostfilter(...)`.

`augment` / `aaugment` do not run row-level postfiltering. This keeps the main
generation path from making one extra judge call per emitted row. Use
`ta.postfilter(...)` explicitly on selected rows when you need row-level review.

### Batch Augmentation Over Multiple Passages

```python
from text_albumentations import run_batch_augmentation
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
    model,
)
```

### Long Text To JSONL

```python
from text_albumentations import save_long_text_dataset
from text_albumentations.tasks.bullets import bullet_augmentation

save_long_text_dataset(
    text=long_text,
    output_jsonl="out.jsonl",
    augmentation=bullet_augmentation,
    runtime=model,
    chunk_size_chars=300,
)
```

### Augmentation Knobs

Every augmentation accepts these parameters to control generation behavior:

| Parameter | Default | Description |
| --- | --- | --- |
| `temperature` | 0.2 | Sampling temperature for base generation (tasks override: e.g. `rephrase` 0.5, `counterfactual` 0.7) |
| `max_tokens` | 5000 | Max tokens for base generation |
| `num_generations` | 1 | Number of independent base generations |
| `variations` | 0 | Extra paraphrased variations per base generation (`bullets` defaults to 1) |
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

### Custom Preprocessing Model

You can also make the augmentation input itself be a custom Pydantic model instead of a raw string.

See [`examples/custom_preprocessing.py`](examples/custom_preprocessing.py).

## Extensibility

The project is designed so users can extend it in layers.

### 1. Define A Task With `ta.task`

For most custom tasks, no class is needed — a schema and a prompt are enough:

```python
key_stat = ta.task(
    prompt="Extract the single most important statistic from this passage.",
    schema=KeyStat,
    output="{statistic} — {context}",          # template over schema fields
    selection_hint="the passage contains a notable number or metric.",
)
```

- `output=` — a template string, a callable `(output) -> str`, or omitted when the schema has exactly one field
- `instruction=` — the training row's instruction, when it should differ from the prompt
- `rows=` — full control: a callable `(passage, output) -> list[AlpacaDataset]` emitting any number of rows
- `selection_hint=` — lets the smart switch know when to pick this task
- generation knobs (`temperature=`, `variations=`, ...) pass straight through

### 2. Add A New Augmentation Class

Subclass `BaseSingleChunkAugmentation` or `BaseMultiChunkAugmentation` when a task needs custom input types, programmatic (LLM-free) generation, or verification logic. Define:

- a Pydantic schema
- a `system_prompt` (sent to the augmenter model)
- a `selection_hint` (read by the smart switch — never sent to the augmenter)
- adapters and/or response formats
- optionally `build_user_message(...)`, `validate_passages(...)`, or `generate_one(...)`

### 3. Add A New Response Format

Subclass `BaseResponseFormat` if you want to control:

- how the format modifies the system prompt
- how the final Alpaca `output` field is rendered

For common Alpaca row generation, `AlpacaResponseFormat` is usually enough.

### 4. Add A New Adapter

Subclass `BaseAlpacaAdapter` to convert a typed structured output into one or more Alpaca rows.

One structured output can expand into multiple rows.

### 5. Add A New Model Backend

Implement the `ModelRuntime` interface if you want a backend beyond the built-in `OpenAIModel` / `LocalMLXModel` / `LocalHFModel` primitives. Your object then works everywhere a model is accepted.

That keeps model execution separate from:

- augmentation semantics
- prompt construction
- dataset adapters
- response serialization

This separation is intentional. The project should let you swap the model layer without rewriting the dataset logic.

## Tests

```bash
uv run pytest -m "not integration"   # offline suite (no model needed)
uv run pytest                        # + integration tests against a local OpenAI-compatible server on :8080
```

## Philosophy On Synthetic Data

This project does not assume synthetic data is magic.

It assumes:

- synthetic data works best when the task shape is explicit
- typed intermediate representations are easier to control
- simple priors beat vague giant prompts
- extensibility matters because different teams want different schemas, formats, and runtimes

The aim is not "generate random data."

The aim is to turn raw text into useful supervision signals for SFT and distillation in a way that is structured, inspectable, and easy to extend.
