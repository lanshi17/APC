# -*- coding: utf-8 -*-
"""真实 LLM 全协议基准 + 跨任务迁移实验（同判分、同方法族，只需一个真实凭证）。

每任务方法族：zero-shot | manual | apc-full(进化搜索, 预算内)
  profile = 真实探针画像(ProbeRunner+ProfileBuilder, 缓存 artifacts/profiles/<model>_probe.json)
  evals   = dev_r 搜索期 / validation 冠军选择 / holdout 终测
  输出    = experiments/apcbench/real_<task>.json，冠军 genome 存 artifacts/optimizations/real_<task>_champ.json

用法:
  .venv/bin/python scripts/bench_real_full.py --task contract --model qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.compiler.renderer import DefaultPromptCompiler
from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.task_spec import TaskSpec
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import Dataset, load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.factory import create_client
from apc.models.mock_client import MockClient
from apc.optimizer.evolutionary import EvolutionaryOptimizer

BASE_GENOME = REPO / "configs" / "genomes" / "base.json"
PROFILES_DIR = REPO / "artifacts" / "profiles"
CHAMPS_DIR = REPO / "artifacts" / "optimizations"

TASK_CFG = {
    "contract": ("configs/tasks/contract_extraction.yaml", "contract_extraction", "contract_rule_v1"),
    "math": ("configs/tasks/math_reasoning.yaml", "math_reasoning", "math_rule_v1"),
    "financial": ("configs/tasks/financial_analysis.yaml", "financial_analysis", "financial_rule_v1"),
}


def real_profile(client) -> ModelProfile:
    """真实模型探针画像；同模型缓存复用（探针是模型属性，与任务无关）。"""
    cache = PROFILES_DIR / f"{client.model_id}_probe.json"
    if cache.exists():
        return ModelProfile.model_validate_json(cache.read_text(encoding="utf-8"))
    from apc.profiler.probe_runner import ProbeRunner
    from apc.profiler.profile_builder import ProfileBuilder
    results = ProbeRunner(client).run_suite(REPO / "probes" / "v1")
    dicts = [r if isinstance(r, dict) else r.model_dump(mode="json") for r in results]
    prof = ProfileBuilder().build(client.model_id, dicts)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(prof.model_dump_json(indent=1), encoding="utf-8")
    return prof


def manual_genome(base: PromptGenome) -> PromptGenome:
    g = deepcopy(base)
    g.examples.enabled, g.examples.count = True, 2
    g.output.strictness, g.output.include_schema_in_prompt = "high", True
    g.output.forbid_extra_fields, g.verification.enabled = True, True
    return g


class RealEnv:
    """一个 (task, model) 的真实评测环境：封装 spec/compiler/profile/client + score_of/ev。"""

    def __init__(self, task: str, model: str, dev_r_n=5, val_n=8, hold_n=20):
        self.task, self.model = task, model
        self.client = create_client(model, prefer_mock=False)
        if isinstance(self.client, MockClient):
            raise RuntimeError(f"模型 {model} 无真实凭证（回退 MockClient），本脚本仅走真实链路。")
        task_yaml, ds_name, self.judge_id = TASK_CFG[task]
        self.spec = TaskSpec.from_yaml(REPO / task_yaml)
        self.base = PromptGenome.from_json(BASE_GENOME)
        self.compiler = DefaultPromptCompiler()
        self.profile = real_profile(self.client)
        ds_dir = REPO / "datasets" / ds_name
        self.dev = Dataset(dataset_id="dev_s", version="real",
                           samples=load_dataset(ds_dir / "dev.jsonl").samples[:8])
        self.val = Dataset(dataset_id="val_r", version="real",
                           samples=load_dataset(ds_dir / "validation.jsonl").samples[:val_n])
        self.hold = Dataset(dataset_id="hold_r", version="real",
                            samples=load_dataset(ds_dir / "holdout.jsonl").samples[:hold_n])
        self.dev_r = Dataset(dataset_id="dev_r", version="real", samples=self.dev.samples[:dev_r_n])
        if task == "financial":
            runner = EvaluationRunner(self.client, judge=RuleBasedJudge(),
                                      checker=RuleBasedChecker(), artifacts_dir=Path("/tmp/bench-real-full"))

            def score_of(g: PromptGenome, ds) -> float:
                cp = self.compiler.compile(g, self.spec, self.profile, apply_rules=False)
                return runner.evaluate(self.spec, cp, ds, save_outputs=False).score
        else:
            from bench_contract import run_task
            def score_of(g: PromptGenome, ds) -> float:
                return run_task(self.spec, self.compiler, self.profile, self.client, ds, g).score
        self.score_of = score_of

    def ev(self, g: PromptGenome, phase: str) -> float:
        ds = self.val if phase == "validation" else (self.dev_r if phase == "dev_r1" else self.dev)
        return self.score_of(g, ds)

    def optimize(self, root: PromptGenome, budget: int, seed=42):
        return EvolutionaryOptimizer(self.spec, self.profile, root, self.ev, generations=2,
                                     population_size=8, elite_k=3, budget=budget, seed=seed).optimize()

    def row(self, method, baseline, validation, holdout, budget, t0, **extra):
        d = {"method": method, "model_id": self.client.model_id, "model_version": self.client.model_version,
             "task": self.task, "seed": 42, "baseline_score": round(float(baseline), 4),
             "validation_score": round(float(validation), 4), "holdout_score": round(float(holdout), 4),
             "budget_used": budget, "elapsed_s": round(time.time() - t0, 1)}
        d.update(extra)
        return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--dev-r", type=int, default=5)
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--hold-n", type=int, default=20)
    args = ap.parse_args()

    env = RealEnv(args.task, args.model, args.dev_r, args.val_n, args.hold_n)
    rows, t_start = [], time.time()

    for method in ("zero-shot", "manual"):
        t0 = time.time()
        g = env.base if method == "zero-shot" else manual_genome(env.base)
        r = env.row(method, env.score_of(g, env.dev_r), env.score_of(g, env.val),
                    env.score_of(g, env.hold), 0, t0)
        rows.append(r)
        print(f"{method:10s} hold={r['holdout_score']:.4f} ({r['elapsed_s']}s)", flush=True)

    t0 = time.time()
    root = CompilerRules.apply(deepcopy(env.base), env.profile)
    rep = env.optimize(root, args.budget)
    champ = PromptGenome.model_validate(rep.champion_genome)
    CHAMPS_DIR.mkdir(parents=True, exist_ok=True)
    (CHAMPS_DIR / f"real_{args.task}_champ.json").write_text(champ.model_dump_json(indent=1), encoding="utf-8")
    r = env.row("apc-full", rep.baseline_score, env.score_of(champ, env.val),
                env.score_of(champ, env.hold), rep.budget_used, t0,
                champion_genome_id=champ.genome_id, mutation_note=champ.mutation_note)
    rows.append(r)
    print(f"{'apc-full':10s} hold={r['holdout_score']:.4f} budget={r['budget_used']} ({r['elapsed_s']}s)", flush=True)

    out = REPO / "experiments" / "apcbench" / f"real_{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"protocol": "real-llm", "total_s": round(time.time() - t_start, 1),
                                        "dev_r": args.dev_r, "val_n": args.val_n, "hold_n": args.hold_n,
                                        "judge": env.judge_id}, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
