"""迁移测试：15 维差、决策阈值（KR-6）、报告字段。"""
from __future__ import annotations

from apc.core.model_profile import ModelProfile
from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.transfer.migration import (
    CapabilityDeltaCalculator,
    MigrationMutator,
    PromptMigrationPipeline,
)


def test_capability_delta_fifteen_dims():
    src, tgt = ModelProfile(model_id="glm"), ModelProfile(model_id="qwen")
    src.capability.math, tgt.capability.math = 0.0, 1.0
    delta = CapabilityDeltaCalculator().calculate(src, tgt)
    assert len(delta.diffs) == 15
    assert delta.diffs["math"] == 1.0


def test_delta_drives_seed_adjustment():
    src, tgt = ModelProfile(model_id="glm"), ModelProfile(model_id="qwen")
    src.capability.few_shot_benefit, tgt.capability.few_shot_benefit = 0.0, 0.5
    delta = CapabilityDeltaCalculator().calculate(src, tgt)
    genome = PromptGenome.from_json(__import__("pathlib").Path(__file__).resolve().parents[2]
                                    / "configs" / "genomes" / "base.json")
    seed = MigrationMutator().apply_delta(genome, delta)
    assert seed.examples.enabled is True and seed.examples.count >= 2


class _StubTrial:
    def __init__(self, score):
        self.score = score
        self.format_error_rate = 0.0
        self.avg_latency_ms = 10.0
        self.total_output_tokens = 100


def _pipeline(adapted_score, source_score):
    spec = TaskSpec.from_yaml(__import__("pathlib").Path(__file__).resolve().parents[2]
                              / "configs" / "tasks" / "financial_analysis.yaml")

    def holdout_eval(genome, phase):
        if phase == "source_holdout":
            return _StubTrial(source_score)
        return _StubTrial(adapted_score)

    def optimizer_factory(seed_genome):
        class _FakeOpt:
            def optimize(self):
                from apc.optimizer.evolutionary import OptimizationReport

                return OptimizationReport(task_id=spec.task_id, model_id="qwen",
                                          baseline_score=0.0, champion_score=0.0,
                                          champion_genome=seed_genome.model_dump(mode="json"),
                                          champion_genome_id=seed_genome.genome_id)

        return _FakeOpt()

    return PromptMigrationPipeline(compiler=None, evaluator=None, holdout_evaluator=holdout_eval,
                                   optimizer_factory=optimizer_factory)


def test_decision_adopt_above_threshold(task_spec):
    src, tgt = ModelProfile(model_id="glm"), ModelProfile(model_id="qwen")
    genome = PromptGenome.from_json(__import__("pathlib").Path(__file__).resolve().parents[2]
                                    / "configs" / "genomes" / "base.json")
    report = _pipeline(adapted_score=0.90, source_score=1.0).migrate(
        task_spec, src, tgt, genome)  # 0.90 >= 0.9*1.0 → adopt
    assert report.decision == "adopt"
    assert report.source_holdout_score == 1.0
    assert report.target_adapted_score == 0.90


def test_decision_keep_source_below_threshold(task_spec):
    src, tgt = ModelProfile(model_id="glm"), ModelProfile(model_id="qwen")
    genome = PromptGenome.from_json(__import__("pathlib").Path(__file__).resolve().parents[2]
                                    / "configs" / "genomes" / "base.json")
    report = _pipeline(adapted_score=0.89, source_score=1.0).migrate(
        task_spec, src, tgt, genome)  # 0.89 < 0.9 → keep_source
    assert report.decision == "keep_source"
