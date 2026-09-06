from __future__ import annotations

import random
from copy import deepcopy
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from apc.core.genome import PromptGenome


def _get_by_path(genome: PromptGenome, path: str):
    obj = genome
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_by_path(genome: PromptGenome, path: str, value) -> None:
    parts = path.split(".")
    obj = genome
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class GenomeMutator:
    """只允许 search_space 内的点路径基因变异（FR-6）。"""

    def __init__(self, search_space: dict[str, list], rng: random.Random):
        self.search_space = search_space
        self.rng = rng

    def mutate(self, genome: PromptGenome) -> tuple[PromptGenome, str]:
        g = deepcopy(genome)
        path = self.rng.choice(list(self.search_space.keys()))
        choices = [v for v in self.search_space[path] if v != _get_by_path(g, path)]
        if not choices:
            # 该路径无可用取值：退化为原体（保持原 id，不产生伪变异记录）
            return genome, ""
        value = self.rng.choice(choices)
        _set_by_path(g, path, value)
        note = f"{path}: {_get_by_path(genome, path)} -> {value}"
        g.parent_genome_id = genome.genome_id
        g.mutation_note = note
        # 内容哈希 id：同内容同 id（可复现），不同内容必不同（防 key 碰撞）
        g.genome_id = g.fresh_content_id()
        return g, note


class TrialTrace(BaseModel):
    genome_id: str
    parent_genome_id: str | None = None
    mutation_note: str | None = None
    phase: str
    score: float


class OptimizationReport(BaseModel):
    task_id: str
    model_id: str
    baseline_score: float
    champion_score: float
    champion_genome: dict = Field(default_factory=dict)
    champion_genome_id: str = ""
    history: list[dict] = Field(default_factory=list)
    trials: list[TrialTrace] = Field(default_factory=list)
    budget_used: int = 0
    improvement: float = 0.0


class EvolutionaryOptimizer:
    """进化搜索 + 规则变异 + Successive Halving（FR-6 / KR-4 / KR-5）。

    evaluator(genome, phase) -> float：phase ∈ dev_r1 / dev_r2 / dev_full / validation，
    由调用方决定各阶段的数据集切片；预算按每次评估计 1。
    """

    def __init__(self, task_spec, model_profile, genome_root: PromptGenome,
                 evaluator: Callable[[PromptGenome, str], float], *,
                 generations: int = 3, population_size: int = 20, elite_k: int = 5,
                 budget: int = 100, seed: int = 42):
        self.task_spec = task_spec
        self.model_profile = model_profile
        self.root = genome_root
        self.evaluator = evaluator
        self.generations = generations
        self.population_size = population_size
        self.elite_k = elite_k
        self.budget = budget
        self.rng = random.Random(seed)
        self.mutator = GenomeMutator(genome_root.search_space or {}, self.rng)
        self.budget_used = 0
        self.trials: list[TrialTrace] = []

    # ---------- 预算 ----------
    def _eval(self, genome: PromptGenome, phase: str) -> float:
        if self.budget_used >= self.budget:
            raise BudgetExhausted(f"预算已耗尽（{self.budget}），无法继续评估")
        score = float(self.evaluator(genome, phase))
        self.budget_used += 1
        self.trials.append(TrialTrace(
            genome_id=genome.genome_id, parent_genome_id=genome.parent_genome_id,
            mutation_note=genome.mutation_note, phase=phase, score=round(score, 4)))
        return score

    # ---------- 搜索 ----------
    def optimize(self) -> OptimizationReport:
        baseline_score = self._eval(self.root, "dev_full")
        population = self._initial_population()
        history = []
        best_genome, best_score = self.root, baseline_score

        for gen in range(self.generations):
            if self.budget_used >= self.budget:
                break
            try:
                # Successive Halving：dev_r1 小子集快速淘汰一半，dev_full 全量复评
                scored = [(g, self._eval(g, "dev_r1")) for g in population]
                scored.sort(key=lambda x: x[1], reverse=True)
                survivors = [g for g, _ in scored[: max(1, len(scored) // 2)]]
                gen_scored = [(g, self._eval(g, "dev_full")) for g in survivors]
            except BudgetExhausted:
                break
            gen_scored.sort(key=lambda x: x[1], reverse=True)
            if gen_scored and gen_scored[0][1] > best_score:
                best_genome, best_score = gen_scored[0]
            # 精英繁殖下一代
            elite = gen_scored[: self.elite_k] or [(best_genome, best_score)]
            population = [deepcopy(g) for g, _ in elite]
            while len(population) < self.population_size:
                mutated, _ = self.mutator.mutate(self.rng.choice(elite)[0])
                population.append(mutated)
            history.append({
                "generation": gen + 1,
                "best_score": round(max((s for _, s in gen_scored), default=best_score), 4),
                "avg_score": round(sum(s for _, s in gen_scored) / len(gen_scored), 4) if gen_scored else None,
                "budget_used": self.budget_used,
                "survivors": len(survivors),
            })

        finalists = [best_genome, self.root]
        val_scores = {}
        for g in finalists:
            try:
                val_scores[g.genome_id] = self._eval(g, "validation")
            except BudgetExhausted:
                break
        champion = max(finalists, key=lambda g: val_scores.get(g.genome_id, -1))

        return OptimizationReport(
            task_id=self.task_spec.task_id,
            model_id=self.model_profile.model_id,
            baseline_score=round(baseline_score, 4),
            champion_score=round(val_scores.get(champion.genome_id, best_score), 4),
            champion_genome=champion.model_dump(mode="json"),
            champion_genome_id=champion.genome_id,
            history=history,
            trials=self.trials,
            budget_used=self.budget_used,
            improvement=round(val_scores.get(champion.genome_id, best_score) - baseline_score, 4),
        )


    def _initial_population(self) -> list[PromptGenome]:
        pop = [deepcopy(self.root)]
        while len(pop) < self.population_size and self.budget_used < self.budget:
            mutated, _ = self.mutator.mutate(self.root)
            pop.append(mutated)
        return pop


class BudgetExhausted(Exception):
    pass
