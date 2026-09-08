"""真实 LLM 稳健性评测：base/manual/champ(rule)/champ_safe 四种 genome 在
perturbation 分集上的分数与相对各自 holdout 的下降量(drop)。
回答 F1 遗留问题——搜索的价值若在格式/约束维度，须以扰动下 drop 更小来兑现。

用法：  .venv/bin/python scripts/bench_real_robust.py --task financial --model qwen --n 20
输出：  experiments/apcbench/real_robust_<task>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.evaluation.dataset import Dataset, load_dataset

from bench_real_full import RealEnv, manual_genome  # noqa: E402

CHAMPS = REPO / "artifacts" / "optimizations"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="financial")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    env = RealEnv(args.task, args.model, 5, 8, args.n)
    pf = REPO / "datasets" / "financial_analysis" / "perturbation.jsonl"
    if not pf.exists():
        print(f"任务 {args.task} 无 perturbation 数据集（当前仅 financial 有）", file=sys.stderr)
        return 2
    pert = Dataset(dataset_id="pert_r", version="real",
                   samples=load_dataset(pf).samples[:args.n])

    genomes = {"base": env.base, "manual": manual_genome(env.base)}
    for arm, fn in (("apc-full", f"real_{args.task}_champ.json"), ("apc-safe", f"real_{args.task}_champ_safe.json")):
        p = CHAMPS / fn
        if p.exists():
            genomes[arm] = PromptGenome.model_validate_json(p.read_text(encoding="utf-8"))

    res = json.loads((REPO / "experiments" / "apcbench" / f"real_{args.task}.json").read_text(encoding="utf-8"))
    hold_by = {r["method"]: r["holdout_score"] for r in res["rows"]}
    hold_by.setdefault("base", hold_by.get("zero-shot"))
    hold_by.setdefault("manual", hold_by.get("manual"))

    out = []
    for name, g in genomes.items():
        t0 = time.time()
        s = env.score_of(g, pert)
        hold = hold_by.get(name)
        row = {"genome": name, "perturbation_score": round(s, 4), "n": len(pert.samples),
               "elapsed_s": round(time.time() - t0, 1)}
        if hold is not None:
            row["holdout"] = hold
            row["drop"] = round(hold - s, 4)
        out.append(row)
        print(f"{name:11s} pert={s:.4f} " + (f"drop={row.get('drop'):+.4f}" if "drop" in row else ""), flush=True)
        dest = REPO / "experiments" / "apcbench" / f"real_robust_{args.task}.json"
        dest.write_text(json.dumps({"task": args.task, "model": args.model, "judge": env.judge_id,
                                    "rows": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
