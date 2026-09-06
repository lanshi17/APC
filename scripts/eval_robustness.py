# -*- coding: utf-8 -*-
"""稳健性评测：b100 各冠军 genome 在 perturbation 分集上的分数与下降量。

输入 experiments/apcbench/bench_results_b100.json（含 champion_genome），
输出 experiments/apcbench/robustness_results.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.compiler.renderer import DefaultPromptCompiler
from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.mock_client import MockClient
from apc.profiler.probe_runner import ProbeRunner
from apc.profiler.profile_builder import ProfileBuilder

TASK = REPO / "configs" / "tasks" / "financial_analysis.yaml"
DS = REPO / "datasets" / "financial_analysis"


def build_profile(model_id: str):
    results = ProbeRunner(MockClient(model_id)).run_suite(REPO / "probes" / "v1")
    dicts = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in results]
    return ProfileBuilder().build(model_id, dicts)


def main():
    spec = TaskSpec.from_yaml(TASK)
    pert = load_dataset(DS / "perturbation.jsonl")
    comp = DefaultPromptCompiler()
    profiles = {m: build_profile(m) for m in ("glm", "qwen", "gpt")}
    rows = json.loads((REPO / "experiments" / "apcbench" / "bench_results_b100.json")
                      .read_text(encoding="utf-8"))
    out = []
    for r in rows:
        g = PromptGenome.model_validate(r["champion_genome"])
        prof = profiles[r["model_id"]]
        cp = comp.compile(g, spec, prof, apply_rules=False)
        t = EvaluationRunner(MockClient(r["model_id"]), judge=RuleBasedJudge(),
                             checker=RuleBasedChecker(),
                             artifacts_dir=Path("/tmp/bm-r")).evaluate(
            spec, cp, pert, save_outputs=False)
        out.append({"method": r["method"], "model_id": r["model_id"], "seed": r["seed"],
                    "holdout": r["holdout_score"], "perturbation": round(t.score, 4),
                    "drop": round(r["holdout_score"] - t.score, 4),
                    "pert_robustness": t.robustness})
        print(f"{r['method']:15s} {r['model_id']:5s} seed={r['seed']} "
              f"hold={r['holdout_score']:.4f} pert={t.score:.4f} drop={r['holdout_score']-t.score:+.4f}",
              flush=True)
    p = REPO / "experiments" / "apcbench" / "robustness_results.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(out)} rows -> {p}")


if __name__ == "__main__":
    main()
