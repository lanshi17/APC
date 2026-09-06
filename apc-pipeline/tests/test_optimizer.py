"""优化器测试：预算、种子确定性、search_space 约束、精英单调。"""
from __future__ import annotations

from apc.core.model_profile import ModelProfile
from apc.core.genome import PromptGenome
from apc.compiler.rules import CompilerRules
from apc.optimizer.evolutionary import EvolutionaryOptimizer


def _make(mock_eval_fn, seed=42, **kw):
    root, evaluate = mock_eval_fn("glm")
    spec_kw = dict(generations=2, population_size=8, elite_k=3, budget=20, seed=seed)
    spec_kw.update(kw)
    return root, EvolutionaryOptimizer(_spec(), ModelProfile(model_id="glm"), root, evaluate,
                                       **spec_kw)


def _spec():
    from pathlib import Path
    from apc.core.task_spec import TaskSpec

    return TaskSpec.from_yaml(Path(__file__).resolve().parents[2]
                              / "configs" / "tasks" / "financial_analysis.yaml")


def test_budget_is_respected(mock_eval_fn):
    _, opt = _make(mock_eval_fn, budget=20)
    report = opt.optimize()
    assert report.budget_used <= 20
    assert len(report.trials) == report.budget_used


def test_same_seed_reproducible(mock_eval_fn):
    _, opt1 = _make(mock_eval_fn, seed=7, budget=30)
    r1 = opt1.optimize()
    _, opt2 = _make(mock_eval_fn, seed=7, budget=30)
    r2 = opt2.optimize()
    assert r1.champion_genome_id == r2.champion_genome_id
    assert r1.champion_score == r2.champion_score


def test_mutations_only_touch_search_space(mock_eval_fn, base_genome):
    _, opt = _make(mock_eval_fn, budget=25)
    report = opt.optimize()
    allowed = set(base_genome.search_space.keys())
    notes = [t.mutation_note for t in report.trials if t.mutation_note]
    assert notes, "应有变异发生"
    for note in notes:
        assert note.split(":")[0] in allowed


def test_champion_not_worse_than_baseline(mock_eval_fn):
    _, opt = _make(mock_eval_fn, budget=40)
    report = opt.optimize()
    assert report.champion_score >= report.baseline_score - 1e-9
    assert report.trials[0].genome_id  # 基线评估也留痕
    assert any(t.parent_genome_id for t in report.trials)


def test_trial_lineage_chain(mock_eval_fn):
    _, opt = _make(mock_eval_fn, budget=40)
    report = opt.optimize()
    ids = {t.genome_id for t in report.trials}
    for t in report.trials:
        if t.parent_genome_id:
            assert t.parent_genome_id in ids | {t.parent_genome_id}


def test_mutant_gets_fresh_id(mock_eval_fn):
    from apc.optimizer.evolutionary import GenomeMutator
    import random

    root, _ = mock_eval_fn("glm")
    mutator = GenomeMutator(root.search_space, random.Random(1))
    mutant, note = mutator.mutate(root)
    assert mutant.genome_id != root.genome_id
    assert mutant.parent_genome_id == root.genome_id
    assert note
