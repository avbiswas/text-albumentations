"""EconomyRouter — drop-in replacement for MetaAugmentation.

Uses market mechanisms (auctions, wealth, bucket-brigade credit assignment,
population evolution) instead of a single-shot LLM call to select
augmentation tasks for each passage.

Usage::

    from text_albumentations.economy_router import EconomyRouter

    router = EconomyRouter(entries, bank_path="economy_bank.jsonl")
    dataset = run_augmentation(passage, router, runtime)

Or via ``ta.augment()`` with ``economy_mode=True``.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.economy import (
    AgentState,
    Auction,
    BucketBrigade,
    PopulationEvolution,
    WealthBank,
)
from text_albumentations.meta import AugmentationEntry
from text_albumentations.runner import arun_augmentation, run_augmentation
from text_albumentations.runtime import ModelRuntime
from text_albumentations.utils import AlpacaDataset

logger = logging.getLogger(__name__)

# Default postfilter prompt used as the reward signal when none is provided.
_DEFAULT_REWARD_PROMPT = (
    "A quality generated training datapoint should be useful for "
    "supervised fine-tuning. Keep the row only if the instruction is "
    "clear, the input contains enough context, and the output directly "
    "satisfies the instruction."
)


class EconomySelection(BaseModel):
    """Minimal schema — actual routing is auction-based, not LLM-selected."""

    winners: list[str] = Field(
        default_factory=list,
        description="Names of winning agents selected by the auction.",
    )


class EconomyRouter(BaseSingleChunkAugmentation[EconomySelection]):
    """Market-based augmentation router replacing MetaAugmentation.

    Each augmentation task is wrapped in an ``AgentState`` with wealth, a
    frozen bid, and an evolving trigger prompt. On each passage:

    1. Agents "wake up" via a lightweight structured call on their trigger.
    2. Eligible agents bid; the auction returns top-k winners.
    3. Winners generate rows via their augmentations.
    4. Postfilter quality scores become environment rewards.
    5. Bucket-brigade settles wealth between winners.
    6. Population evolution runs if the episode threshold is reached.

    The WealthBank persists across calls so the router learns over time.
    """

    schema = EconomySelection
    temperature = 0.1
    max_tokens = 64

    def __init__(
        self,
        augmentations: list[AugmentationEntry],
        *,
        bank_path: str = "economy_bank.jsonl",
        evolve_every: int = 50,
        max_winners: int = 5,
        rent_rate: float = 1.0,
        reward_prompt: str | None = None,
        mutation_temperature: float = 0.5,
        max_population: int = 20,
        top_k_parents: int = 3,
        sample_instruction_template: bool = True,
    ) -> None:
        super().__init__()
        self._aug_map: dict[str, BaseSingleChunkAugmentation] = {}
        self._bank_path = bank_path
        self._max_winners = max_winners
        self._rent_rate = rent_rate
        self._reward_prompt = reward_prompt or _DEFAULT_REWARD_PROMPT
        self._sample_instruction_template = sample_instruction_template

        for entry in augmentations:
            name, aug = entry[0], entry[1]
            hint = entry[2] if len(entry) > 2 else aug.selection_hint or ""
            self._aug_map[name] = aug

        # Build initial agent states if the bank is empty.
        self._bank = self._load_bank()
        if not self._bank.agents:
            self._seed_agents()

        self._evolution: PopulationEvolution | None = None  # lazy-init with runtime

    # -- public API ----------------------------------------------------------

    def build_dataset(
        self,
        passages: str,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        """Sync path: auction → generate → reward → settle → evolve."""
        validated = self.validate_passages(passages)
        bank = self._ensure_bank(runtime)

        # Auction
        auction = Auction(
            list(bank.agents.values()),
            runtime,
            max_winners=self._max_winners,
        )
        winners = auction.run(validated)
        if not winners:
            return []

        # Generate rows per winner
        dataset: list[AlpacaDataset] = []
        for name in winners:
            agent = bank.get(name)
            rows = run_augmentation(
                validated,
                agent.augmentation,
                runtime,
                sample_instruction_template=self._sample_instruction_template,
            )
            dataset.extend(rows)

        # Compute reward per agent (fraction of rows passing postfilter)
        rewards = self._compute_rewards(winners, dataset, runtime)

        # Bucket-brigade settlement
        BucketBrigade.settle(winners, rewards, bank, rent_rate=self._rent_rate)

        # Population evolution
        self._maybe_evolve(bank, runtime)

        # Persist
        bank.save()

        logger.info(
            "EconomyRouter: winners=%s rewards=%s bank_snapshot=%s",
            winners,
            {k: f"{v:.2f}" for k, v in rewards.items()},
            f"active={len(bank.active_agents())} total={len(bank.agents)}",
        )

        return dataset

    async def abuild_dataset(
        self,
        passages: str,
        runtime: ModelRuntime,
    ) -> list[AlpacaDataset]:
        """Async path: auction → generate → reward → settle → evolve."""
        import asyncio

        validated = self.validate_passages(passages)
        bank = self._ensure_bank(runtime)

        # Auction (async)
        auction = Auction(
            list(bank.agents.values()),
            runtime,
            max_winners=self._max_winners,
        )
        winners = await auction.arun(validated)
        if not winners:
            return []

        # Generate rows per winner (parallel)
        tasks = [
            arun_augmentation(
                validated,
                bank.get(name).augmentation,
                runtime,
                sample_instruction_template=self._sample_instruction_template,
            )
            for name in winners
        ]
        results = await asyncio.gather(*tasks)
        dataset: list[AlpacaDataset] = []
        for rows in results:
            dataset.extend(rows)

        # Compute rewards (async)
        rewards = await self._acompute_rewards(winners, dataset, runtime)

        # Settle
        BucketBrigade.settle(winners, rewards, bank, rent_rate=self._rent_rate)

        # Evolve
        self._maybe_evolve(bank, runtime)

        # Persist
        bank.save()
        return dataset

    # -- internals -----------------------------------------------------------

    def _load_bank(self) -> WealthBank:
        bank = WealthBank(self._bank_path)
        bank.load(self._aug_map)
        return bank

    def _seed_agents(self) -> None:
        """Create initial agent states from the augmentation map."""
        for name, aug in self._aug_map.items():
            hint = aug.selection_hint or ""
            agent = AgentState(
                name=name,
                augmentation=aug,
                trigger_prompt=hint,
            )
            self._bank.update(agent)
        self._bank.save()

    def _ensure_bank(self, runtime: ModelRuntime) -> WealthBank:
        """Ensure bank is loaded and evolution is initialized."""
        if not self._bank.agents:
            self._bank = self._load_bank()
        if not self._bank.agents:
            self._seed_agents()
        if self._evolution is None:
            self._evolution = PopulationEvolution(
                runtime,
                evolve_every=self._evolution_every,
                mutation_temperature=self._mutation_temperature,
                max_population=self._max_population,
                top_k_parents=self._top_k_parents,
            )
        return self._bank

    @property
    def _evolution_every(self) -> int:
        return getattr(self, "__evolve_every", 50)

    @_evolution_every.setter
    def _evolution_every(self, value: int) -> None:
        self.__evolve_every = value

    @property
    def _mutation_temperature(self) -> float:
        return getattr(self, "__mutation_temp", 0.5)

    @property
    def _max_population(self) -> int:
        return getattr(self, "__max_pop", 20)

    @property
    def _top_k_parents(self) -> int:
        return getattr(self, "__top_k", 3)

    def _compute_rewards(
        self,
        winners: list[str],
        dataset: list[AlpacaDataset],
        runtime: ModelRuntime,
    ) -> dict[str, float]:
        """Reward = fraction of each agent's rows that pass postfilter."""
        if not dataset:
            return {name: 0.0 for name in winners}

        from text_albumentations.batch_postfilter import batch_postfilter

        # Batch-assess ALL rows in one call (or a few batched calls).
        all_assessments = batch_postfilter(
            [row.model_dump() for row in dataset],
            self._reward_prompt,
            model=runtime,
        )

        # Attribute rows to agents round-robin.
        rewards: dict[str, float] = {}
        per_agent = len(dataset) // max(len(winners), 1)
        remainder = len(dataset) % max(len(winners), 1)

        idx = 0
        for i, name in enumerate(winners):
            count = per_agent + (1 if i < remainder else 0)
            agent_assessments = all_assessments[idx : idx + count]
            idx += count

            if not agent_assessments:
                rewards[name] = 0.0
                continue

            passed = sum(1 for a in agent_assessments if a.is_quality)
            rewards[name] = passed / len(agent_assessments)

        return rewards

    async def _acompute_rewards(
        self,
        winners: list[str],
        dataset: list[AlpacaDataset],
        runtime: ModelRuntime,
    ) -> dict[str, float]:
        """Async variant of reward computation."""
        if not dataset:
            return {name: 0.0 for name in winners}

        from text_albumentations.batch_postfilter import abatch_postfilter

        all_assessments = await abatch_postfilter(
            [row.model_dump() for row in dataset],
            self._reward_prompt,
            model=runtime,
        )

        rewards: dict[str, float] = {}
        per_agent = len(dataset) // max(len(winners), 1)
        remainder = len(dataset) % max(len(winners), 1)

        idx = 0
        for i, name in enumerate(winners):
            count = per_agent + (1 if i < remainder else 0)
            agent_assessments = all_assessments[idx : idx + count]
            idx += count

            if not agent_assessments:
                rewards[name] = 0.0
                continue

            passed = sum(1 for a in agent_assessments if a.is_quality)
            rewards[name] = passed / len(agent_assessments)

        return rewards

    def _maybe_evolve(self, bank: WealthBank, runtime: ModelRuntime) -> None:
        """Run population evolution if the episode threshold is reached."""
        if self._evolution is None:
            self._evolution = PopulationEvolution(runtime)
        if self._evolution.should_evolve(bank):
            new_names = self._evolution.evolve(bank, self._aug_map)
            if new_names:
                logger.info("Population evolved: new agents %s", new_names)
