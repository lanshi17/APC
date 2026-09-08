# -*- coding: utf-8 -*-
"""真实 LLM 跨任务 PromptGenome 迁移实验（单凭证即可跑，APC「可迁移」核心验证）。

问题：在 task A 上进化出的冠军 genome，直接迁移到 task B（同模型），
      是否在更少 eval 预算下达到/超过 B 从零搜索？

三臂对比（同 B 预算，同真实 qwen 模型）:
  cold        —— B 从 base genome 进化（基准）
  transfer-0  —— 直接套用 A 的冠军 genome 到 B（零适配，测原始可迁移性）
  transfer-ws —— 从 A 的冠军 genome warm-start 在 B 上继续进化（预算内适配）

用法:
  .venv/bin/python scripts/bench_real_transfer.py --source contract --target financial --budget 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.compiler.rules import CompilerRules
from apc.core.genome import PromptGenome
from bench_real_full import CHAMPS_DIR, RealEnv, BASE_GENOME


def load_champ(task: str) -> PromptGenome:
    p = CHAMPS_DIR / f"real_{task}_champ.json"
    if not p.exists():
        raise FileNotFoundError(f"源任务冠军不存在: {p}（先跑 bench_real_full --task {task}）")
    return PromptGenome.from_json(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["contract", "math", "financial"])
    ap.add_argument("--target", required=True, choices=["contract", "math", "financial"])
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.source == args.target:
        print("source 与 target 必须不同"); return 2

    tenv = RealEnv(args.target, args.model)
    champ_a = load_champ(args.source)
    rows = []

    # ---- transfer-0：A 冠军零适配直评 B ----
    t0 = time.time()
    r0 = tenv.row("transfer-0", tenv.score_of(champ_a, tenv.dev_r),
                  tenv.score_of(champ_a, tenv.val), tenv.score_of(champ_a, tenv.hold), 0, t0,
                  source_genome_id=champ_a.genome_id)
    rows.append(r0)
    print(f"transfer-0   B_hold={r0['holdout_score']:.4f}", flush=True)

    # ---- cold：B 从 base 进化 ----
    t0 = time.time()
    root_c = CompilerRules.apply(deepcopy(tenv.base), tenv.profile)
    rep_c = tenv.optimize(root_c, args.budget, args.seed)
    champ_c = PromptGenome.model_validate(rep_c.champion_genome)
    rc = tenv.row("cold", rep_c.baseline_score, tenv.score_of(champ_c, tenv.val),
                  tenv.score_of(champ_c, tenv.hold), rep_c.budget_used, t0)
    rows.append(rc)
    print(f"cold         B_hold={rc['holdout_score']:.4f} budget={rc['budget_used']}", flush=True)

    # ---- transfer-ws：A 冠军 warm-start 进化 B ----
    t0 = time.time()
    root_w = deepcopy(champ_a)
    root_w.parent_genome_id = champ_a.genome_id
    root_w.task_id = tenv.spec.task_id
    root_w.mutation_note = f"transfer_init:{args.source}->{args.target}"
    rep_w = tenv.optimize(root_w, args.budget, args.seed)
    champ_w = PromptGenome.model_validate(rep_w.champion_genome)
    rw = tenv.row("transfer-ws", rep_w.baseline_score, tenv.score_of(champ_w, tenv.val),
                  tenv.score_of(champ_w, tenv.hold), rep_w.budget_used, t0,
                  source_genome_id=champ_a.genome_id, champion_genome_id=champ_w.genome_id)
    rows.append(rw)
    print(f"transfer-ws  B_hold={rw['holdout_score']:.4f} budget={rw['budget_used']}", flush=True)

    gain = round(rw["holdout_score"] - rc["holdout_score"], 4)
    print(f"\n迁移收益 transfer-ws - cold = {gain:+.4f}（同预算 {args.budget}）")
    out = REPO / "experiments" / "apcbench" / f"real_transfer_{args.source}_to_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"source": args.source, "target": args.target,
                                        "model": tenv.client.model_id, "model_version": tenv.client.model_version,
                                        "budget": args.budget, "seed": args.seed, "judge": tenv.judge_id},
                               "transfer_gain_vs_cold": gain, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
