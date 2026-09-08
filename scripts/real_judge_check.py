"""真实 LLM 上的 Judge 一致性抽检：rule-judge vs LLM-judge（同一被评模型 ⇒ 非独立，
诚实标注 self-judge 弱效度）。对每个样本比较两个 judge 的 accuracy/constraint 分数，
输出 Pearson + Spearman 与分歧样本清单，供论文 §3.4 判分有效性论证。

用法（仓库根目录）：
  .venv/bin/python scripts/real_judge_check.py --task financial --model qwen --genome champ
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.judge import LLMJudge, RuleBasedJudge
from apc.evaluation.runner import EvaluationRunner
from apc.models.factory import create_client

from bench_real_full import RealEnv  # noqa: E402


def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def corr(a: list[float], b: list[float]) -> float:
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return statistics.correlation(a, b)


def spearman(a: list[float], b: list[float]) -> float:
    return corr(rank(a), rank(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="financial")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--genome", choices=["base", "champ"], default="champ")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    env = RealEnv(args.task, args.model, 5, 8, args.n)
    if args.genome == "champ":
        champ = REPO / "artifacts" / "optimizations" / f"real_{args.task}_champ.json"
        genome = PromptGenome.model_validate_json(champ.read_text(encoding="utf-8"))
    else:
        genome = env.base
    cp = env.compiler.compile(genome, env.spec, env.profile, apply_rules=False)
    runner = EvaluationRunner(env.client, judge=RuleBasedJudge(), checker=RuleBasedChecker(),
                              artifacts_dir=Path("/tmp/judgecheck"))
    trial = runner.evaluate(env.spec, cp, env.hold, save_outputs=True)
    rows = [json.loads(l) for l in
            Path(trial.meta["outputs_file"]).read_text(encoding="utf-8").splitlines() if l]

    llm_judge = LLMJudge(create_client(args.model))
    pairs_acc, pairs_cf, disagreements = [], [], []
    for i, row in enumerate(rows):
        sample = env.hold.samples[i]
        doc = runner._document(sample)
        lj = llm_judge.judge(doc, sample.get("expected", {}), row["output"], env.spec)
        ra, la = float(row["judge"].get("accuracy", 0.0)), float(lj.get("accuracy", 0.0))
        rc, lc = float(row["judge"].get("constraint_following", 0.0)), float(lj.get("constraint_following", 0.0))
        pairs_acc += [ra, la]
        pairs_cf += [rc, lc]
        if abs(ra - la) >= 0.25:
            disagreements.append({"sample_id": row["sample_id"], "rule": ra, "llm": la})
    a_r, a_l = pairs_acc[0::2], pairs_acc[1::2]
    c_r, c_l = pairs_cf[0::2], pairs_cf[1::2]
    out = {
        "task": args.task, "model": args.model, "genome": args.genome, "n": len(rows),
        "independence": "self-judge(非独立,弱效度对照)",
        "accuracy": {"pearson": corr(a_r, a_l), "spearman": spearman(a_r, a_l),
                     "mean_rule": round(statistics.fmean(a_r), 4), "mean_llm": round(statistics.fmean(a_l), 4)},
        "constraint_following": {"pearson": corr(c_r, c_l), "spearman": spearman(c_r, c_l),
                                 "mean_rule": round(statistics.fmean(c_r), 4), "mean_llm": round(statistics.fmean(c_l), 4)},
        "disagreements": disagreements,
    }
    dest = REPO / "experiments" / "apcbench" / f"real_judge_agreement_{args.task}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
