# -*- coding: utf-8 -*-
"""PGAM 真实验证：ProfileGuidedMutator vs 均匀变异（GenomeMutator），同任务同预算同 seed。

判别格选择 rationale：financial 是无余量格（两变异器必然同带，无信息量）；
math 上 gpt-6 冷搜 .6336 与冠军直迁 .8443 之间有真实方差——搜索器质量可分辨。
uniform 臂 = bench_real_transfer.py 的 cold 行（同 task/model/budget/seed/协议，直接复用）。
本脚本只跑 PGAM 臂，输出 real_<task>_<model>_pgam.json。

用法:
  .venv/bin/python scripts/bench_real_pgam.py --task math --model gpt6 --budget 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from apc.optimizer.evolutionary import EvolutionaryOptimizer, ProfileGuidedMutator
from bench_real_full import TASK_CFG, RealEnv
from copy import deepcopy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--model", default="gpt6")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = RealEnv(args.task, args.model)
    env.seed = args.seed
    capability = env.profile.capability.model_dump()
    print(f"capability: {json.dumps({k: round(v, 3) for k, v in capability.items()})}", flush=True)

    t0 = time.time()
    root = CompilerRules.apply(deepcopy(env.base), env.profile)
    mut = ProfileGuidedMutator(root.search_space or {}, random.Random(args.seed), capability)
    rep = EvolutionaryOptimizer(env.spec, env.profile, root, env.ev, generations=2,
                                population_size=8, elite_k=3, budget=args.budget,
                                seed=args.seed, mutator=mut).optimize()
    champ = PromptGenome.model_validate(rep.champion_genome)
    r = env.row("pgam-cold", rep.baseline_score, env.score_of(champ, env.val),
                env.score_of(champ, env.hold), rep.budget_used, t0,
                champion_genome_id=champ.genome_id)
    print(f"pgam-cold    hold={r['holdout_score']:.4f} budget={r['budget_used']}", flush=True)

    out = REPO / "experiments" / "apcbench" / f"real_{args.task}_{args.model}_pgam.json"
    out.write_text(json.dumps({"meta": {"protocol": "real-pgam", "model": args.model,
                                        "task": args.task, "budget": args.budget,
                                        "seed": args.seed, "judge": env.judge_id,
                                        "note": "PGAM=ProfileGuidedMutator(画像先验+bandit); uniform 对照=real_transfer_*_to_*.json 的 cold 行(同 task/model/budget/seed)"},
                               "rows": [r]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
