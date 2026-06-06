"""Economy of Minds (EOM) primitives for market-based augmentation routing.

Each augmentation is an "agent" with wealth, a frozen bid, and an evolving
trigger prompt. Agents compete in auctions for the right to act on a passage.
Postfilter quality scores become environment rewards. Bucket-brigade credit
assignment flows value backward through action chains. Population evolution
prunes bankrupt agents and mutates wealthy ones across episodes.

Reference: arXiv 2606.02859 — Economy of Minds (Harvard, 2026).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.runtime import ModelRuntime


# ---------------------------------------------------------------------------
# Structured outputs used by the economy
# ---------------------------------------------------------------------------


class WakeUpCheck(BaseModel):
    should_wake: bool = Field(
        ...,
        description="True if this agent should participate in the auction for this passage.",
    )


class MutationOutput(BaseModel):
    mutated_prompt: str = Field(
        ...,
        description="A slightly modified version of the trigger prompt.",
    )


# ---------------------------------------------------------------------------
# AgentState — per-agent economic state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Economic wrapper around a single augmentation task."""

    name: str
    augmentation: BaseSingleChunkAugmentation
    wealth: float = 100.0
    frozen_bid: float = 10.0
    trigger_prompt: str = ""
    lineage_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    episodes_played: int = 0
    total_reward: float = 0.0
    bankruptcy_threshold: float = 0.0
    rent_rate: float = 1.0

    # -- helpers -------------------------------------------------------------

    @property
    def is_bankrupt(self) -> bool:
        return self.wealth <= self.bankruptcy_threshold

    @property
    def avg_reward(self) -> float:
        if self.episodes_played == 0:
            return 0.0
        return self.total_reward / self.episodes_played

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (augmentation omitted)."""
        return {
            "name": self.name,
            "wealth": self.wealth,
            "frozen_bid": self.frozen_bid,
            "trigger_prompt": self.trigger_prompt,
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "episodes_played": self.episodes_played,
            "total_reward": self.total_reward,
            "bankruptcy_threshold": self.bankruptcy_threshold,
            "rent_rate": self.rent_rate,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        augmentation: BaseSingleChunkAugmentation,
    ) -> AgentState:
        """Reconstruct from a persisted dict + live augmentation."""
        return cls(
            name=d["name"],
            augmentation=augmentation,
            wealth=d.get("wealth", 100.0),
            frozen_bid=d.get("frozen_bid", 10.0),
            trigger_prompt=d.get("trigger_prompt", ""),
            lineage_id=d.get("lineage_id", uuid.uuid4().hex[:12]),
            generation=d.get("generation", 0),
            episodes_played=d.get("episodes_played", 0),
            total_reward=d.get("total_reward", 0.0),
            bankruptcy_threshold=d.get("bankruptcy_threshold", 0.0),
            rent_rate=d.get("rent_rate", 1.0),
        )


# ---------------------------------------------------------------------------
# WealthBank — persistence layer
# ---------------------------------------------------------------------------


class WealthBank:
    """Persist agent states across ``augment()`` calls.

    Storage format: one JSON object per agent per save, appended as JSONL.
    The latest entry for each agent name wins on load.
    """

    def __init__(self, path: str | Path = "economy_bank.jsonl") -> None:
        self.path = Path(path)
        self.agents: dict[str, AgentState] = {}

    # -- I/O ----------------------------------------------------------------

    def load(
        self,
        augmentation_map: dict[str, BaseSingleChunkAugmentation] | None = None,
    ) -> None:
        """Load agent states from disk. ``augmentation_map`` supplies live
        augmentation instances for deserialized agents."""
        if not self.path.exists():
            return

        latest: dict[str, dict] = {}
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    latest[record["name"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue

        for name, record in latest.items():
            if augmentation_map and name in augmentation_map:
                self.agents[name] = AgentState.from_dict(
                    record, augmentation_map[name]
                )

    def save(self) -> None:
        """Append current agent states to the bank file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for agent in self.agents.values():
                f.write(json.dumps(agent.to_dict(), ensure_ascii=False) + "\n")

    # -- query helpers -------------------------------------------------------

    def get(self, name: str) -> AgentState:
        return self.agents[name]

    def update(self, agent: AgentState) -> None:
        self.agents[agent.name] = agent

    def remove(self, name: str) -> None:
        self.agents.pop(name, None)

    def bankrupt_agents(self) -> list[str]:
        return [name for name, a in self.agents.items() if a.is_bankrupt]

    def top_agents(self, n: int) -> list[AgentState]:
        ranked = sorted(
            self.agents.values(), key=lambda a: a.wealth, reverse=True
        )
        return ranked[:n]

    def active_agents(self) -> list[AgentState]:
        return [a for a in self.agents.values() if not a.is_bankrupt]

    def snapshot(self) -> dict:
        """Full state snapshot for logging / analysis."""
        return {
            "agents": {n: a.to_dict() for n, a in self.agents.items()},
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents()),
            "total_wealth": sum(a.wealth for a in self.agents.values()),
        }


# ---------------------------------------------------------------------------
# Auction — bidding mechanism
# ---------------------------------------------------------------------------


class Auction:
    """First-price sealed-bid auction among eligible agents.

    Each agent's ``trigger_prompt`` is used in a lightweight structured call
    to decide if the agent "wakes up" (is eligible to bid). Eligible agents
    submit their frozen bid. The auction returns all eligible agents ranked
    by bid, enabling multiple augmentations per passage.
    """

    def __init__(
        self,
        agents: list[AgentState],
        runtime: ModelRuntime,
        *,
        temperature: float = 0.1,
        max_tokens: int = 64,
        max_winners: int = 5,
    ) -> None:
        self.agents = agents
        self.runtime = runtime
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_winners = max_winners

    def _check_wake_up(self, agent: AgentState, passage: str) -> bool:
        """Structured call: should this agent wake up for this passage?"""
        if not agent.trigger_prompt:
            # No trigger prompt = always eligible (sensible default).
            return True

        result = self.runtime.generate_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "Decide whether the described augmentation is suitable "
                        "for the given passage. Return should_wake=true only "
                        f"if the augmentation fits.\n\nAgent description: {agent.trigger_prompt}"
                    ),
                },
                {"role": "user", "content": passage},
            ],
            WakeUpCheck,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return result.should_wake

    async def _acheck_wake_up(self, agent: AgentState, passage: str) -> bool:
        """Async variant of wake-up check."""
        if not agent.trigger_prompt:
            return True

        result = await self.runtime.agenerate_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "Decide whether the described augmentation is suitable "
                        "for the given passage. Return should_wake=true only "
                        f"if the augmentation fits.\n\nAgent description: {agent.trigger_prompt}"
                    ),
                },
                {"role": "user", "content": passage},
            ],
            WakeUpCheck,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return result.should_wake

    def run(self, passage: str) -> list[str]:
        """Run auction. Returns ordered list of winning agent names (top-k)."""
        eligible: list[AgentState] = []
        for agent in self.agents:
            if self._check_wake_up(agent, passage):
                eligible.append(agent)

        # Sort by frozen bid descending (higher bid = higher priority).
        eligible.sort(key=lambda a: a.frozen_bid, reverse=True)
        return [a.name for a in eligible[: self.max_winners]]

    async def arun(self, passage: str) -> list[str]:
        """Async variant of auction."""
        import asyncio

        wake_results = await asyncio.gather(
            *[self._acheck_wake_up(agent, passage) for agent in self.agents]
        )

        eligible = [
            agent
            for agent, woke in zip(self.agents, wake_results)
            if woke
        ]
        eligible.sort(key=lambda a: a.frozen_bid, reverse=True)
        return [a.name for a in eligible[: self.max_winners]]


# ---------------------------------------------------------------------------
# BucketBrigade — credit assignment
# ---------------------------------------------------------------------------


class BucketBrigade:
    """Post-episode wealth transfer using bucket-brigade credit assignment.

    1. Each winner collects the environment reward ``r_t``.
    2. Each winner pays its bid to the *previous* winner (or house if first).
    3. Rent is charged to *all* agents (inactivity penalty).
    """

    @staticmethod
    def settle(
        winners: list[str],
        rewards: dict[str, float],
        bank: WealthBank,
        *,
        rent_rate: float | None = None,
    ) -> None:
        """Settle wealth after one episode."""
        for i, name in enumerate(winners):
            agent = bank.get(name)
            reward = rewards.get(name, 0.0)

            # Collect environment reward.
            agent.wealth += reward
            agent.total_reward += reward
            agent.episodes_played += 1

            # Pay bid to previous winner (or house for first).
            agent.wealth -= agent.frozen_bid
            if i > 0:
                prev = bank.get(winners[i - 1])
                prev.wealth += agent.frozen_bid

        # Rent: decay all agents to penalize inactivity.
        for agent in bank.agents.values():
            rate = rent_rate if rent_rate is not None else agent.rent_rate
            agent.wealth -= rate


# ---------------------------------------------------------------------------
# PopulationEvolution — cross-episode adaptation
# ---------------------------------------------------------------------------

_MUTATE_SYSTEM_PROMPT = (
    "You are an augmentation-trigger optimizer. Given the current trigger "
    "prompt and the agent's performance, produce a slightly modified version "
    "that preserves what works but adds controlled variation. "
    "Keep the prompt concise (one sentence)."
)

_EXPLORE_SYSTEM_PROMPT = (
    "You are an augmentation-trigger designer. A previous agent with this "
    "trigger failed (went bankrupt). Create a new trigger prompt that takes "
    "a different approach, correcting likely failure modes. "
    "Keep the prompt concise (one sentence)."
)


class PopulationEvolution:
    """After enough episodes, evolve the population:

    - **Prune** bankrupt agents (wealth <= threshold).
    - **Exploit**: mutate wealthy agents' trigger prompts to produce children.
    - **Explore**: create new agents with corrected prompts for failure modes.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        evolve_every: int = 50,
        mutation_temperature: float = 0.5,
        max_population: int = 20,
        top_k_parents: int = 3,
    ) -> None:
        self.runtime = runtime
        self.evolve_every = evolve_every
        self.mutation_temperature = mutation_temperature
        self.max_population = max_population
        self.top_k_parents = top_k_parents

    def should_evolve(self, bank: WealthBank) -> bool:
        total_played = sum(a.episodes_played for a in bank.agents.values())
        return total_played >= self.evolve_every

    def evolve(
        self,
        bank: WealthBank,
        augmentation_map: dict[str, BaseSingleChunkAugmentation],
    ) -> list[str]:
        """Run one evolution cycle. Returns list of new agent names."""
        new_names: list[str] = []

        # 1. Prune bankrupt agents.
        for name in bank.bankrupt_agents():
            bank.remove(name)

        # 2. Exploit: mutate top-k parents.
        parents = bank.top_agents(self.top_k_parents)
        for parent in parents:
            if len(bank.agents) >= self.max_population:
                break
            child = self._mutate(parent, augmentation_map)
            bank.update(child)
            new_names.append(child.name)

        # 3. Explore: for each removed slot, create a new variant.
        while len(bank.agents) < min(self.max_population, len(augmentation_map)):
            # Pick a task not yet in the population.
            for task_name, augmentation in augmentation_map.items():
                if task_name not in bank.agents:
                    new_agent = self._explore(task_name, augmentation, bank)
                    bank.update(new_agent)
                    new_names.append(new_agent.name)
                    break
            else:
                break  # All tasks already represented.

        return new_names

    def _mutate(
        self,
        parent: AgentState,
        augmentation_map: dict[str, BaseSingleChunkAugmentation],
    ) -> AgentState:
        """Produce a child agent with a slightly varied trigger prompt."""
        child_suffix = f"_g{parent.generation + 1}_{uuid.uuid4().hex[:4]}"
        child_name = f"{parent.name}{child_suffix}"

        mutated_prompt = self._generate_mutated_prompt(parent.trigger_prompt)

        child_aug = augmentation_map.get(parent.name.rstrip("_0123456789abcdef_g"))
        if child_aug is None:
            child_aug = parent.augmentation

        return AgentState(
            name=child_name,
            augmentation=child_aug,
            wealth=parent.wealth * 0.3,  # Child gets 30% of parent's wealth.
            frozen_bid=parent.frozen_bid,
            trigger_prompt=mutated_prompt,
            lineage_id=parent.lineage_id,
            generation=parent.generation + 1,
            rent_rate=parent.rent_rate,
        )

    def _explore(
        self,
        task_name: str,
        augmentation: BaseSingleChunkAugmentation,
        bank: WealthBank,
    ) -> AgentState:
        """Create a new agent exploring a different behavioral region."""
        existing = bank.agents.get(task_name)
        failed_prompt = existing.trigger_prompt if existing else ""

        new_prompt = self._generate_explore_prompt(
            failed_prompt, augmentation.selection_hint or ""
        )

        return AgentState(
            name=f"{task_name}_e{uuid.uuid4().hex[:4]}",
            augmentation=augmentation,
            wealth=50.0,  # Fresh start.
            frozen_bid=10.0,
            trigger_prompt=new_prompt,
            lineage_id=uuid.uuid4().hex[:12],
            generation=0,
        )

    # -- LLM helpers ---------------------------------------------------------

    def _generate_mutated_prompt(self, current_prompt: str) -> str:
        result = self.runtime.generate_structured(
            [
                {"role": "system", "content": _MUTATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Current trigger prompt:\n{current_prompt}",
                },
            ],
            MutationOutput,
            temperature=self.mutation_temperature,
            max_tokens=128,
        )
        return result.mutated_prompt

    def _generate_explore_prompt(
        self, failed_prompt: str, fallback_hint: str
    ) -> str:
        user_content = (
            f"Failed trigger prompt:\n{failed_prompt}\n\n"
            f"Original task description (for context):\n{fallback_hint}"
            if failed_prompt
            else f"Task description:\n{fallback_hint}"
        )
        result = self.runtime.generate_structured(
            [
                {"role": "system", "content": _EXPLORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            MutationOutput,
            temperature=self.mutation_temperature,
            max_tokens=128,
        )
        return result.mutated_prompt
