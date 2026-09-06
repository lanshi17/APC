"""评测链测试：checker / judge 值等价 / scorer 公式 / runner 追溯字段。"""
from __future__ import annotations

from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.evaluation.scorer import TrialScorer
from apc.models.mock_client import MockClient

VALID_OUTPUT = ('{"summary": "公司营收增长", "metrics": [{"name": "营收", "value": "12.5亿", '
                '"change": "+10%", "comment": "增长"}], "risks": ["r1", "r2", "r3"], '
                '"confidence": 0.9}')


def test_checker_json_parse_failure():
    checker = RuleBasedChecker()
    spec = _spec()
    r = checker.check("噪声前缀 {\"summary\": 1}", spec)
    assert r["format_score"] == 0.0 and not r["json_parsable"]
    assert checker.format_error(r)


def test_checker_missing_and_extra_fields():
    checker = RuleBasedChecker()
    r = checker.check('{"summary": "x"}', _spec())
    assert "metrics" in r["missing_fields"]
    assert checker.format_error(r)
    r2 = checker.check('{"summary": "x", "metrics": [], "risks": [], "confidence": 0.5, "zzz": 1}',
                       _spec())
    assert r2["extra_fields"] == ["zzz"]


def test_checker_constraints_pass():
    checker = RuleBasedChecker()
    r = checker.check(VALID_OUTPUT, _spec())
    assert r["format_score"] == 1.0 and r["constraint_score"] == 1.0
    assert not checker.format_error(r)


def test_judge_unit_equivalence(task_spec):
    judge = RuleBasedJudge()
    expected = {"metrics": [{"name": "营收", "value": "12.5亿", "change": "+10%", "comment": "增长"}]}
    actual = ('{"metrics": [{"name": "营收", "value": "12.5亿元", "change": "+10%", '
              '"comment": "增长"}]}')
    out = judge.judge("输入", expected, actual, task_spec)
    # 值等价（12.5亿元 == 12.5亿）→ 指标命中
    assert out["accuracy"] > 0.9


def test_runner_produces_traceable_trial(task_spec, repo_root, tmp_path):
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.core.genome import PromptGenome
    from apc.core.model_profile import ModelProfile

    dataset = load_dataset(repo_root / "datasets" / "financial_analysis" / "dev.jsonl")
    genome = PromptGenome.from_json(repo_root / "configs" / "genomes" / "base.json")
    cp = DefaultPromptCompiler().compile(genome, task_spec, ModelProfile(model_id="glm"))
    trial = EvaluationRunner(MockClient("glm"), artifacts_dir=tmp_path).evaluate(
        task_spec, cp, dataset)
    # KR-3 全字段追溯
    assert trial.task_id == task_spec.task_id
    assert trial.model_id == "glm" and trial.genome_id == genome.genome_id
    assert trial.prompt_id == cp.prompt_id and trial.dataset_version
    assert trial.judge_id == "rule_based_v1"
    assert trial.parent_genome_id is None and trial.mutation_note is None
    # 原始输出落盘
    assert (tmp_path / "outputs" / f"{trial.trial_id}.jsonl").exists()


def _spec():
    from apc.core.task_spec import TaskSpec
    from pathlib import Path

    return TaskSpec.from_yaml(Path(__file__).resolve().parents[2]
                              / "configs" / "tasks" / "financial_analysis.yaml")
