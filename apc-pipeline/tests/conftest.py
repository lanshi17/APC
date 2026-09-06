"""共享 fixtures：真实 configs/datasets + Mock 客户端，全部可复现（无网络）。"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_SPEC = REPO_ROOT / "configs" / "tasks" / "financial_analysis.yaml"
BASE_GENOME = REPO_ROOT / "configs" / "genomes" / "base.json"


@pytest.fixture()
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def task_spec():
    from apc.core.task_spec import TaskSpec

    return TaskSpec.from_yaml(TASK_SPEC)


@pytest.fixture()
def base_genome():
    from apc.core.genome import PromptGenome

    return PromptGenome.from_json(BASE_GENOME)


@pytest.fixture()
def mock_eval_fn(task_spec, base_genome, tmp_path):
    """Mock 闭包 evaluator：genome + phase → score（与 CLI/graph 相同口径）。"""
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.compiler.rules import CompilerRules
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile
    from apc.evaluation.checker import RuleBasedChecker
    from apc.evaluation.dataset import load_dataset
    from apc.evaluation.judge import RuleBasedJudge
    from apc.evaluation.runner import EvaluationRunner
    from apc.models.mock_client import MockClient

    dataset = load_dataset(REPO_ROOT / "datasets" / "financial_analysis" / "dev.jsonl")

    def factory(model_id: str):
        compiler = DefaultPromptCompiler()
        profile = ModelProfile(model_id=model_id)
        root = CompilerRules.apply(base_genome, profile)
        client = MockClient(model_id)
        runner = EvaluationRunner(client, judge=RuleBasedJudge(), checker=RuleBasedChecker(),
                                  artifacts_dir=tmp_path)

        def evaluate(genome: PromptGenome, phase: str) -> float:
            cp = compiler.compile(genome, task_spec, profile, apply_rules=False)
            return runner.evaluate(task_spec, cp, dataset, save_outputs=False).score

        return root, evaluate

    return factory
