from __future__ import annotations

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.runner import arun_augmentation, run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset, save_dataset


def chunk_text_by_chars(
    text: str,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
) -> list[str]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than 0.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be greater than or equal to 0.")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars.")

    stripped_text = text.strip()
    if not stripped_text:
        return []

    step = chunk_size_chars - overlap_chars
    return [
        stripped_text[idx : idx + chunk_size_chars]
        for idx in range(0, len(stripped_text), step)
        if stripped_text[idx : idx + chunk_size_chars].strip()
    ]


def generate_rows_from_long_text(
    text: str,
    augmentation: BaseSingleChunkAugmentation,
    runtime: ModelRuntime,
    *,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
) -> list[AlpacaDataset]:
    dataset = []
    chunks = chunk_text_by_chars(
        text,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )

    for chunk in chunks:
        dataset.extend(run_augmentation(chunk, augmentation, runtime))

    return dataset


async def agenerate_rows_from_long_text(
    text: str,
    augmentation: BaseSingleChunkAugmentation,
    runtime: ModelRuntime,
    *,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
) -> list[AlpacaDataset]:
    dataset = []
    chunks = chunk_text_by_chars(
        text,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )

    for chunk in chunks:
        dataset.extend(await arun_augmentation(chunk, augmentation, runtime))

    return dataset


def save_long_text_dataset(
    text: str,
    output_jsonl: str,
    augmentation: BaseSingleChunkAugmentation,
    runtime: ModelRuntime,
    *,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
) -> list[AlpacaDataset]:
    dataset = generate_rows_from_long_text(
        text,
        augmentation,
        runtime,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )
    save_dataset(dataset, output_jsonl)
    return dataset


async def asave_long_text_dataset(
    text: str,
    output_jsonl: str,
    augmentation: BaseSingleChunkAugmentation,
    runtime: ModelRuntime,
    *,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
) -> list[AlpacaDataset]:
    dataset = await agenerate_rows_from_long_text(
        text,
        augmentation,
        runtime,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )
    save_dataset(dataset, output_jsonl)
    return dataset
