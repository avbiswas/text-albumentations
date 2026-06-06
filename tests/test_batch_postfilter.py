"""Tests for batch_postfilter optimization."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from text_albumentations.postfilter import PostfilterAssessment
from text_albumentations.batch_postfilter import (
    BatchPostfilterAssessment,
    batch_postfilter,
    batch_filter_rows,
    _build_batch_postfilter_messages,
)
from text_albumentations.utils import AlpacaDataset


def _make_runtime(assessments: list[PostfilterAssessment]) -> MagicMock:
    """Create a mock runtime that returns a BatchPostfilterAssessment."""
    runtime = MagicMock()
    runtime.generate_structured = MagicMock(
        return_value=BatchPostfilterAssessment(assessments=assessments)
    )
    return runtime


class TestBatchPostfilterMessages:
    def test_includes_all_datapoints(self):
        datapoints = [
            {"instruction": "Q1", "input": "text", "output": "A1"},
            {"instruction": "Q2", "input": "text", "output": "A2"},
        ]
        messages = _build_batch_postfilter_messages(datapoints, "quality criteria")
        assert len(messages) == 2
        assert "2" in messages[1]["content"]  # count mentioned
        assert "Q1" in messages[1]["content"]
        assert "Q2" in messages[1]["content"]

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValueError):
            _build_batch_postfilter_messages([{"a": 1}], "  ")


class TestBatchPostfilter:
    def test_returns_correct_count(self):
        datapoints = [{"q": f"Q{i}"} for i in range(5)]
        expected = [
            PostfilterAssessment(is_quality=True, reason="ok")
            for _ in range(5)
        ]
        runtime = _make_runtime(expected)

        results = batch_postfilter(datapoints, "test prompt", model=runtime)
        assert len(results) == 5
        assert all(r.is_quality for r in results)

    def test_mixed_quality(self):
        datapoints = [{"q": f"Q{i}"} for i in range(3)]
        expected = [
            PostfilterAssessment(is_quality=True, reason="good"),
            PostfilterAssessment(is_quality=False, reason="bad"),
            PostfilterAssessment(is_quality=True, reason="ok"),
        ]
        runtime = _make_runtime(expected)

        results = batch_postfilter(datapoints, "test", model=runtime)
        assert results[0].is_quality is True
        assert results[1].is_quality is False
        assert results[2].is_quality is True

    def test_batches_correctly(self):
        """8 items with batch_size=3 → 3 batches (3, 3, 2)."""
        datapoints = [{"q": f"Q{i}"} for i in range(8)]
        call_count = 0

        def _gen(messages, output_type, **kwargs):
            nonlocal call_count
            call_count += 1
            # Extract count from the user message
            content = messages[1]["content"]
            # Parse the datapoints from the JSON in the content
            count = 3 if call_count <= 2 else 2
            return BatchPostfilterAssessment(
                assessments=[
                    PostfilterAssessment(is_quality=True, reason="ok")
                    for _ in range(count)
                ]
            )

        runtime = MagicMock()
        runtime.generate_structured = MagicMock(side_effect=_gen)

        results = batch_postfilter(
            datapoints, "test", model=runtime, batch_size=3
        )
        assert len(results) == 8
        assert call_count == 3  # 3 batches

    def test_pads_missing_assessments(self):
        """Model returns fewer assessments than inputs → pad with False."""
        datapoints = [{"q": f"Q{i}"} for i in range(5)]
        # Model returns only 3 assessments
        runtime = _make_runtime(
            [PostfilterAssessment(is_quality=True, reason="ok") for _ in range(3)]
        )

        results = batch_postfilter(datapoints, "test", model=runtime, batch_size=10)
        assert len(results) == 5
        assert results[3].is_quality is False  # padded
        assert results[4].is_quality is False  # padded

    def test_truncates_extra_assessments(self):
        """Model returns more assessments than inputs → truncate."""
        datapoints = [{"q": "Q1"}]
        runtime = _make_runtime(
            [
                PostfilterAssessment(is_quality=True, reason="ok"),
                PostfilterAssessment(is_quality=True, reason="extra"),
            ]
        )

        results = batch_postfilter(datapoints, "test", model=runtime)
        assert len(results) == 1  # truncated to input count


class TestBatchFilterRows:
    def test_filters_correctly(self):
        rows = [
            AlpacaDataset(instruction=f"Q{i}", input="text", output=f"A{i}")
            for i in range(4)
        ]
        expected = [
            PostfilterAssessment(is_quality=True, reason="ok"),
            PostfilterAssessment(is_quality=False, reason="bad"),
            PostfilterAssessment(is_quality=True, reason="ok"),
            PostfilterAssessment(is_quality=False, reason="bad"),
        ]
        runtime = _make_runtime(expected)

        kept = batch_filter_rows(rows, "test prompt", runtime)
        assert len(kept) == 2
        assert kept[0].instruction == "Q0"
        assert kept[1].instruction == "Q2"

    def test_none_prompt_returns_all(self):
        rows = [
            AlpacaDataset(instruction="Q", input="text", output="A")
        ]
        kept = batch_filter_rows(rows, None, MagicMock())
        assert len(kept) == 1

    def test_empty_rows(self):
        runtime = _make_runtime([])
        kept = batch_filter_rows([], "test", runtime)
        assert kept == []

    def test_single_row(self):
        rows = [AlpacaDataset(instruction="Q", input="text", output="A")]
        runtime = _make_runtime(
            [PostfilterAssessment(is_quality=True, reason="ok")]
        )
        kept = batch_filter_rows(rows, "test", runtime)
        assert len(kept) == 1


class TestBatchPostfilterAssessment:
    def test_schema_valid(self):
        a = BatchPostfilterAssessment(
            assessments=[
                PostfilterAssessment(is_quality=True, reason="ok"),
                PostfilterAssessment(is_quality=False, reason="bad"),
            ]
        )
        assert len(a.assessments) == 2
