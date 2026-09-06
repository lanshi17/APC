from __future__ import annotations

from copy import deepcopy
from typing import Callable

from pydantic import BaseModel, Field

from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.task_spec import TaskSpec
from apc.optimizer.evolutionary import EvolutionaryOptimizer


class CapabilityDelta(BaseModel):
    """源/目标模型的 15 维能力差（target − source），驱动 seed genome 调整（FR-7）。"""

    source_model_id: str
    target_model_id: str
    diffs: dict[str, float] = Field(default_factory=dict)


class CapabilityDeltaCalculator:
    def calculate(self, source: ModelProfile, target: ModelProfile) -> CapabilityDelta:
        s = source.capability.model_dump()
        t = target.capability.model_dump()
        return CapabilityDelta(source_model_id=source.model_id, target_model_id=target.model_id,
                               diffs={k: round(t[k] - s[k], 4) for k in s})


class MigrationMutator:
    """CapabilityDelta → seed genome 调整规则（只调 search_space 内基因，FR-7）。"""

    def apply_delta(self, genome: PromptGenome, delta: CapabilityDelta) -> PromptGenome:
        g = deepcopy(genome)
        d = delta.diffs
        notes: list[str] = []
        if d.get("few_shot_benefit", 0) > 0.15:
            g.examples.enabled = True
            g.examples.count = max(g.examples.count, 2)
            notes.append("examples.enabled -> True")
        if d.get("json_reliability", 0) < -0.05:
            g.output.strictness = "high"
            g.output.include_schema_in_prompt = True
            g.output.forbid_extra_fields = True
            notes.append("output.strictness -> high")
        if d.get("schema_strictness", 0) > 0.05 and g.output.strictness == "high":
            g.output.strictness = "medium"
            notes.append("output.strictness -> medium")
        if d.get("self_verification_benefit", 0) > 0.10:
            g.verification.enabled = True
            g.verification.type = "constraint_check"
            notes.append("verification.enabled -> True")
        if d.get("math", 0) < -0.05 or d.get("reasoning", 0) < -0.05:
            g.verification.enabled = True
            notes.append("verification.enabled -> True (weak reasoning)")
        if notes:
            # 血统：seed 记录来源与能力差触发的调整（KR-3/KR-6）
            g.parent_genome_id = genome.genome_id
            g.mutation_note = "migration: " + "; ".join(notes)
            g.genome_id = g.fresh_content_id()
        return g


class MigrationReport(BaseModel):
    """迁移对比报告与采纳决策（FR-7 / KR-6：目标 holdout ≥ 源分数 90%）。"""

    source_model_id: str
    target_model_id: str
    capability_delta: dict[str, float] = Field(default_factory=dict)
    source_holdout_score: float
    target_baseline_score: float
    target_adapted_score: float
    target_format_error_rate: float
    target_avg_latency_ms: float
    target_total_output_tokens: int
    budget_used: int = 0
    decision: str = "keep_source"
    adapted_genome: dict = Field(default_factory=dict)
    adapted_genome_id: str = ""
    trials: list[dict] = Field(default_factory=list)


class PromptMigrationPipeline:
    """源模型冠军 genome → 目标模型 seed 调整 → 目标模型快速优化 → holdout 对比决策。"""

    def __init__(self, compiler, evaluator: Callable[[PromptGenome, str], float],
                 holdout_evaluator: Callable[[PromptGenome, str], object],
                 optimizer_factory: Callable[[PromptGenome], EvolutionaryOptimizer],
                 delta_calculator: CapabilityDeltaCalculator | None = None,
                 migration_mutator: MigrationMutator | None = None):
        self.compiler = compiler
        self.evaluator = evaluator
        self.holdout_evaluator = holdout_evaluator
        self.optimizer_factory = optimizer_factory
        self.delta_calculator = delta_calculator or CapabilityDeltaCalculator()
        self.migration_mutator = migration_mutator or MigrationMutator()

    def migrate(self, task_spec: TaskSpec, source_profile: ModelProfile, target_profile: ModelProfile,
                source_champion_genome: PromptGenome) -> MigrationReport:
        delta = self.delta_calculator.calculate(source_profile, target_profile)
        # seed：源冠军 genome 按目标能力差异调整 + 规则适配目标模型
        seed = self.migration_mutator.apply_delta(source_champion_genome, delta)
        seed = CompilerRules.apply(seed, target_profile)

        # 目标模型快速优化（小预算）
        opt = self.optimizer_factory(seed)
        report = opt.optimize()

        # holdout 对比：目标模型分别用「源冠军直迁」与「适配后冠军」
        adapted_genome = PromptGenome.model_validate(report.champion_genome)
        baseline_trial = self.holdout_evaluator(source_champion_genome, "holdout")
        adapted_trial = self.holdout_evaluator(adapted_genome, "holdout")
        source_trial = self.holdout_evaluator(source_champion_genome, "source_holdout")

        baseline_score = float(getattr(baseline_trial, "score"))
        adapted_score = float(getattr(adapted_trial, "score"))
        source_score = float(getattr(source_trial, "score"))

        decision = "adopt" if adapted_score >= 0.9 * source_score else "keep_source"
        return MigrationReport(
            source_model_id=source_profile.model_id,
            target_model_id=target_profile.model_id,
            capability_delta=delta.diffs,
            source_holdout_score=round(source_score, 4),
            target_baseline_score=round(baseline_score, 4),
            target_adapted_score=round(adapted_score, 4),
            target_format_error_rate=float(getattr(adapted_trial, "format_error_rate", 0.0)),
            target_avg_latency_ms=float(getattr(adapted_trial, "avg_latency_ms", 0.0)),
            target_total_output_tokens=int(getattr(adapted_trial, "total_output_tokens", 0)),
            budget_used=report.budget_used,
            decision=decision,
            adapted_genome=report.champion_genome,
            adapted_genome_id=report.champion_genome_id,
            trials=[t.model_dump() for t in report.trials],
        )
