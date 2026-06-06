"""Tests for the Economy of Minds (EOM) routing layer.

Covers: AgentState, WealthBank, Auction, BucketBrigade, PopulationEvolution,
EconomyRouter, and backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import BaseModel

from text_albumentations.economy import (
    AgentState,
    Auction,
    BucketBrigade,
    PopulationEvolution,
    WakeUpCheck,
    WealthBank,
)
from text_albumentations.economy_router import EconomyRouter, EconomySelection
from text_albumentations.utils import AlpacaDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeSchema(BaseModel):
    text: str


def _make_aug(name: str, hint: str = "test hint") -> MagicMock:
    """Create a mock augmentation with a selection_hint."""
    aug = MagicMock()
    aug.selection_hint = hint
    aug.__class__.__name__ = f"{name}Aug"
    return aug


def _make_runtime(wake_results: list[bool] | None = None) -> MagicMock:
    """Create a mock ModelRuntime.

    Args:
        wake_results: Sequence of should_wake values for successive calls.
    """
    runtime = MagicMock()

    if wake_results is not None:
        call_count = 0

        def _gen_structured(messages, output_type, **kwargs):
            nonlocal call_count
            if output_type is WakeUpCheck:
                idx = min(call_count, len(wake_results) - 1)
                result = WakeUpCheck(should_wake=wake_results[idx])
                call_count += 1
                return result
            # Default: return a valid instance of whatever schema
            return output_type.model_validate({"text": "fake"})

        runtime.generate_structured = MagicMock(side_effect=_gen_structured)
    else:
        runtime.generate_structured = MagicMock(
            return_value=FakeSchema(text="fake")
        )

    runtime.agenerate_structured = MagicMock(
        return_value=asyncio.coroutine(lambda *a, **k: FakeSchema(text="fake"))()
    )

    return runtime


# ===========================================================================
# AgentState tests
# ===========================================================================


class TestAgentState:
    def test_init_defaults(self):
        aug = _make_aug("test")
        state = AgentState(name="test", augmentation=aug)
        assert state.wealth == 100.0
        assert state.frozen_bid == 10.0
        assert state.episodes_played == 0
        assert state.generation == 0
        assert not state.is_bankrupt

    def test_bankrupt_below_threshold(self):
        aug = _make_aug("test")
        state = AgentState(name="test", augmentation=aug, wealth=-1.0)
        assert state.is_bankrupt

    def test_avg_reward_zero_episodes(self):
        aug = _make_aug("test")
        state = AgentState(name="test", augmentation=aug)
        assert state.avg_reward == 0.0

    def test_avg_reward_with_episodes(self):
        aug = _make_aug("test")
        state = AgentState(
            name="test", augmentation=aug, total_reward=7.5, episodes_played=3
        )
        assert state.avg_reward == pytest.approx(2.5)

    def test_to_dict_and_from_dict_roundtrip(self):
        aug = _make_aug("test")
        state = AgentState(
            name="test",
            augmentation=aug,
            wealth=50.0,
            frozen_bid=5.0,
            trigger_prompt="wake for X",
            episodes_played=10,
            total_reward=25.0,
        )
        d = state.to_dict()
        restored = AgentState.from_dict(d, aug)

        assert restored.name == state.name
        assert restored.wealth == state.wealth
        assert restored.frozen_bid == state.frozen_bid
        assert restored.trigger_prompt == state.trigger_prompt
        assert restored.episodes_played == state.episodes_played
        assert restored.total_reward == state.total_reward


# ===========================================================================
# WealthBank tests
# ===========================================================================


class TestWealthBank:
    def test_roundtrip(self):
        """Save → load preserves agent states."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            aug = _make_aug("qa")

            bank = WealthBank(path)
            agent = AgentState(name="qa", augmentation=aug, wealth=75.0)
            bank.update(agent)
            bank.save()

            bank2 = WealthBank(path)
            bank2.load({"qa": aug})
            assert "qa" in bank2.agents
            assert bank2.get("qa").wealth == 75.0

    def test_latest_entry_wins(self):
        """When multiple entries exist for same agent, last one is used."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            aug = _make_aug("qa")

            bank = WealthBank(path)
            bank.update(AgentState(name="qa", augmentation=aug, wealth=50.0))
            bank.save()

            bank.update(AgentState(name="qa", augmentation=aug, wealth=90.0))
            bank.save()

            bank2 = WealthBank(path)
            bank2.load({"qa": aug})
            assert bank2.get("qa").wealth == 90.0

    def test_bankrupt_agents(self):
        aug = _make_aug("a")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="a", augmentation=aug, wealth=-5.0))
        bank.update(AgentState(name="b", augmentation=aug, wealth=50.0))
        assert bank.bankrupt_agents() == ["a"]

    def test_top_agents(self):
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="low", augmentation=aug, wealth=10.0))
        bank.update(AgentState(name="high", augmentation=aug, wealth=200.0))
        bank.update(AgentState(name="mid", augmentation=aug, wealth=50.0))

        top = bank.top_agents(2)
        assert [a.name for a in top] == ["high", "mid"]

    def test_active_agents_excludes_bankrupt(self):
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="alive", augmentation=aug, wealth=10.0))
        bank.update(AgentState(name="dead", augmentation=aug, wealth=-1.0))
        assert len(bank.active_agents()) == 1
        assert bank.active_agents()[0].name == "alive"


# ===========================================================================
# Auction tests
# ===========================================================================


class TestAuction:
    def test_single_winner(self):
        """One eligible agent wins."""
        aug = _make_aug("qa")
        agents = [AgentState(name="qa", augmentation=aug, trigger_prompt="")]
        runtime = _make_runtime()

        auction = Auction(agents, runtime, max_winners=5)
        winners = auction.run("some passage")
        assert winners == ["qa"]

    def test_multiple_eligible_ranked_by_bid(self):
        """Multiple eligible agents, ranked by frozen_bid descending."""
        aug = _make_aug("x")
        agents = [
            AgentState(name="low_bid", augmentation=aug, frozen_bid=5.0, trigger_prompt=""),
            AgentState(name="high_bid", augmentation=aug, frozen_bid=20.0, trigger_prompt=""),
            AgentState(name="mid_bid", augmentation=aug, frozen_bid=10.0, trigger_prompt=""),
        ]
        runtime = _make_runtime()

        auction = Auction(agents, runtime, max_winners=5)
        winners = auction.run("passage")
        assert winners == ["high_bid", "mid_bid", "low_bid"]

    def test_max_winners_limits_results(self):
        """Only top-k winners returned."""
        aug = _make_aug("x")
        agents = [
            AgentState(name=f"a{i}", augmentation=aug, frozen_bid=float(i), trigger_prompt="")
            for i in range(10)
        ]
        runtime = _make_runtime()

        auction = Auction(agents, runtime, max_winners=3)
        winners = auction.run("passage")
        assert len(winners) == 3

    def test_no_eligible_returns_empty(self):
        """All agents decline to wake → empty list."""
        aug = _make_aug("x")
        agents = [AgentState(name="qa", augmentation=aug, trigger_prompt="only for code")]
        runtime = _make_runtime(wake_results=[False])

        auction = Auction(agents, runtime, max_winners=5)
        winners = auction.run("natural language passage")
        assert winners == []

    def test_async_auction(self):
        """Async auction returns same results as sync."""
        aug = _make_aug("x")
        agents = [
            AgentState(name="a", augmentation=aug, frozen_bid=5.0, trigger_prompt=""),
            AgentState(name="b", augmentation=aug, frozen_bid=15.0, trigger_prompt=""),
        ]
        runtime = _make_runtime()

        async def _run():
            auction = Auction(agents, runtime, max_winners=5)
            return await auction.arun("passage")

        winners = asyncio.get_event_loop().run_until_complete(_run())
        assert winners == ["b", "a"]


# ===========================================================================
# BucketBrigade tests
# ===========================================================================


class TestBucketBrigade:
    def test_settle_basic(self):
        """Winners collect reward, pay bids, rent charged."""
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="w1", augmentation=aug, wealth=100.0, frozen_bid=10.0))
        bank.update(AgentState(name="w2", augmentation=aug, wealth=100.0, frozen_bid=10.0))

        BucketBrigade.settle(
            winners=["w1", "w2"],
            rewards={"w1": 0.8, "w2": 0.5},
            bank=bank,
            rent_rate=1.0,
        )

        w1 = bank.get("w1")
        w2 = bank.get("w2")

        # w1: +0.8 reward, -10.0 bid, -1.0 rent = 89.8
        # w2: +0.5 reward, -10.0 bid, pays 10.0 to w1, -1.0 rent = 99.5
        # But wait: w2 pays bid to w1, so w1 also gets +10.0 from w2
        assert w1.wealth == pytest.approx(100.0 + 0.8 - 10.0 + 10.0 - 1.0)  # 99.8
        assert w2.wealth == pytest.approx(100.0 + 0.5 - 10.0 - 1.0)  # 89.5

        assert w1.episodes_played == 1
        assert w2.episodes_played == 1
        assert w1.total_reward == pytest.approx(0.8)
        assert w2.total_reward == pytest.approx(0.5)

    def test_first_winner_pays_house(self):
        """First winner's bid goes to the house (no previous winner)."""
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        initial = 100.0
        bank.update(
            AgentState(name="only", augmentation=aug, wealth=initial, frozen_bid=10.0)
        )

        BucketBrigade.settle(["only"], {"only": 0.9}, bank, rent_rate=0.0)

        agent = bank.get("only")
        # +0.9 reward, -10.0 bid (to house), 0.0 rent
        assert agent.wealth == pytest.approx(initial + 0.9 - 10.0)

    def test_rent_decay(self):
        """All agents lose rent each episode."""
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="a", augmentation=aug, wealth=50.0))
        bank.update(AgentState(name="b", augmentation=aug, wealth=30.0))

        BucketBrigade.settle(["a"], {"a": 0.0}, bank, rent_rate=2.0)

        assert bank.get("a").wealth == pytest.approx(50.0 + 0.0 - 10.0 - 2.0)
        assert bank.get("b").wealth == pytest.approx(30.0 - 2.0)

    def test_bankruptcy_after_settle(self):
        """Agent goes bankrupt after negative wealth."""
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(
            AgentState(name="weak", augmentation=aug, wealth=1.0, frozen_bid=10.0)
        )

        BucketBrigade.settle(["weak"], {"weak": 0.0}, bank, rent_rate=0.0)
        assert bank.get("weak").is_bankrupt


# ===========================================================================
# PopulationEvolution tests
# ===========================================================================


class TestPopulationEvolution:
    def test_should_evolve_below_threshold(self):
        runtime = _make_runtime()
        evo = PopulationEvolution(runtime, evolve_every=100)
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(
            AgentState(name="a", augmentation=aug, episodes_played=10)
        )
        assert not evo.should_evolve(bank)

    def test_should_evolve_at_threshold(self):
        runtime = _make_runtime()
        evo = PopulationEvolution(runtime, evolve_every=50)
        aug = _make_aug("x")
        bank = WealthBank(":memory:")
        bank.update(
            AgentState(name="a", augmentation=aug, episodes_played=50)
        )
        assert evo.should_evolve(bank)

    def test_evolve_prunes_bankrupt(self):
        runtime = _make_runtime()
        # Return a mutation output for the explore call
        runtime.generate_structured = MagicMock(
            return_value=MagicMock(mutated_prompt="new prompt")
        )
        evo = PopulationEvolution(runtime, evolve_every=10, max_population=5)
        aug = _make_aug("qa")
        bank = WealthBank(":memory:")
        bank.update(AgentState(name="qa", augmentation=aug, wealth=-5.0))

        evo.evolve(bank, {"qa": aug})
        assert "qa" not in bank.agents or bank.get("qa").wealth > 0

    def test_evolve_mutates_top_agents(self):
        runtime = _make_runtime()
        runtime.generate_structured = MagicMock(
            return_value=MagicMock(mutated_prompt="evolved prompt")
        )
        evo = PopulationEvolution(runtime, evolve_every=10, top_k_parents=1)
        aug = _make_aug("qa")
        bank = WealthBank(":memory:")
        bank.update(
            AgentState(
                name="qa",
                augmentation=aug,
                wealth=200.0,
                trigger_prompt="original",
                episodes_played=20,
            )
        )

        new_names = evo.evolve(bank, {"qa": aug})
        assert len(new_names) >= 1
        # The child should have a different name and the mutated prompt
        child = bank.agents[new_names[0]]
        assert child.trigger_prompt == "evolved prompt"
        assert child.generation == 1


# ===========================================================================
# EconomyRouter integration tests
# ===========================================================================


class TestEconomyRouter:
    def _make_router(
        self,
        augmentations: list[tuple[str, MagicMock]] | None = None,
    ) -> EconomyRouter:
        if augmentations is None:
            augmentations = [
                ("qa", _make_aug("qa", "passage has Q&A content")),
                ("rephrase", _make_aug("rephrase", "passage is rephrasable")),
            ]

        entries = [(name, aug) for name, aug in augmentations]

        with tempfile.TemporaryDirectory() as tmp:
            bank_path = str(Path(tmp) / "test_bank.jsonl")
            router = EconomyRouter(entries, bank_path=bank_path)
            # Patch bank_path to use temp dir
            router._bank_path = bank_path
            # Re-seed in the temp path
            router._bank = WealthBank(bank_path)
            router._seed_agents()
            return router

    def test_backward_compat_economy_mode_false(self):
        """When economy_mode is not used, MetaAugmentation path is unchanged."""
        # This test just verifies EconomyRouter doesn't break the import chain.
        from text_albumentations.meta import MetaAugmentation

        aug = _make_aug("qa", "test hint")
        meta = MetaAugmentation([("qa", aug)])
        assert meta._aug_map["qa"] == aug

    def test_economy_router_builds_dataset(self):
        """EconomyRouter returns dataset from winning agents."""
        router = self._make_router()
        runtime = _make_runtime()

        # Mock run_augmentation to return rows
        mock_rows = [
            AlpacaDataset(instruction="Q", input="passage", output="A")
        ]
        with patch("text_albumentations.economy_router.run_augmentation", return_value=mock_rows):
            with patch("text_albumentations.economy_router.postfilter") as mock_pf:
                mock_pf.return_value = MagicMock(is_quality=True)
                dataset = router.build_dataset("some passage about things", runtime)

        assert len(dataset) > 0

    def test_economy_router_no_winners(self):
        """When no agents wake up, returns empty dataset."""
        router = self._make_router()
        runtime = _make_runtime(wake_results=[False, False])

        dataset = router.build_dataset("some passage", runtime)
        assert dataset == []

    def test_economy_router_persistence(self):
        """Wealth survives across two build_dataset calls."""
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = str(Path(tmp) / "persist_bank.jsonl")
            aug = _make_aug("qa", "hint")
            entries = [("qa", aug)]

            router1 = EconomyRouter(entries, bank_path=bank_path)
            # Seed the bank
            router1._bank = WealthBank(bank_path)
            router1._seed_agents()

            # Simulate one episode
            agent = router1._bank.get("qa")
            agent.wealth = 75.0
            agent.episodes_played = 5
            router1._bank.save()

            # Load fresh
            router2 = EconomyRouter(entries, bank_path=bank_path)
            router2._bank = WealthBank(bank_path)
            router2._bank.load({"qa": aug})

            assert router2._bank.get("qa").wealth == 75.0
            assert router2._bank.get("qa").episodes_played == 5

    def test_economy_selection_schema(self):
        """EconomySelection schema works as expected."""
        sel = EconomySelection(winners=["qa", "rephrase"])
        assert sel.winners == ["qa", "rephrase"]

        empty = EconomySelection()
        assert empty.winners == []


# ===========================================================================
# Backward compatibility tests
# ===========================================================================


class TestBackwardCompat:
    def test_meta_augmentation_unchanged(self):
        """MetaAugmentation still works exactly as before."""
        from text_albumentations.meta import MetaAugmentation

        aug = _make_aug("qa", "passage has Q&A content")
        meta = MetaAugmentation([("qa", aug)])

        assert "qa" in meta._aug_map
        assert len(meta._aug_options) == 1
        assert meta._aug_options[0].selection_hint == "passage has Q&A content"

    def test_selection_mode_auto_still_works(self):
        """selection_mode='auto' path in augment() is not affected."""
        # We just test the import and that _infer_selection_mode works.
        from text_albumentations.easy import _infer_selection_mode

        assert _infer_selection_mode(None, None) == "auto"
        assert _infer_selection_mode(["qa"], None) == "explicit"
        assert _infer_selection_mode({"qa": 0.5}, None) == "sample"
        assert _infer_selection_mode(None, "explicit") == "explicit"
