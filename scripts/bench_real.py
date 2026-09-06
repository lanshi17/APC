# -*- coding: utf-8 -*-
"""真实 LLM 烟囱测试：用与 Mock 基准完全同口径的评测链，在真实模型上跑 2 个样本。

用法：
  cp .env.example .env   # 填入任一供应商 key
  .venv/bin/python scripts/bench_real.py --model glm --task financial_report_analysis_v1

无凭证时 create_client 回退 MockClient → 本脚本明确报告 SKIP（不伪造真实结论）。
有凭证时输出 2 样本的 TrialResult（accuracy/format/约束/token/延迟），
证明真实链路（factory → client → runner → scorer）端到端可用。
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.compiler.renderer import DefaultPromptCompiler
from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.task_spec import TaskSpec
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.factory import create_client
from apc.models.mock_client import MockClient

TASKS = {
    "financial_report_analysis_v1": (
        REPO / "configs" / "tasks" / "financial_analysis.yaml",
        REPO / "datasets" / "financial_analysis" / "dev.jsonl"),
    "contract_extraction_v1": (
        REPO / "configs" / "tasks" / "contract_extraction.yaml",
        REPO / "datasets" / "contract_extraction" / "dev.jsonl"),
    "math_reasoning_v1": (
        REPO / "configs" / "tasks" / "math_reasoning.yaml",
        REPO / "datasets" / "math_reasoning" / "dev.jsonl"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm")
    ap.add_argument("--task", default="financial_report_analysis_v1", choices=list(TASKS))
    args = ap.parse_args()
    client = create_client(args.model, prefer_mock=False)
    if isinstance(client, MockClient):
        print(f"SKIP: 模型 {args.model} 无可用凭证（回退 MockClient）。填 .env 后重跑。")
        return 2
    task_path, ds_path = TASKS[args.task]
    spec = TaskSpec.from_yaml(task_path)
    base = PromptGenome.from_json(REPO / "configs" / "genomes" / "base.json")
    genome = CompilerRules.apply(deepcopy(base), ModelProfile(model_id=args.model))
    cp = DefaultPromptCompiler().compile(genome, spec, ModelProfile(model_id=args.model))
    ds = load_dataset(ds_path)
    from apc.evaluation.dataset import Dataset
    tiny = Dataset(dataset_id=ds.dataset_id, version=ds.version, samples=ds.samples[:2])
    trial = EvaluationRunner(client, judge=RuleBasedJudge(), checker=RuleBasedChecker(),
                             artifacts_dir=Path("/tmp/bench-real")).evaluate(
        spec, cp, tiny, save_outputs=False)
    print(f"REAL RUN: model={trial.model_id} version={trial.model_version} "
          f"score={trial.score} acc={trial.accuracy} fmt={trial.format_score} "
          f"in_tok={trial.total_input_tokens} out_tok={trial.total_output_tokens} "
          f"lat={trial.avg_latency_ms}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
