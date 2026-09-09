"""ESPO 基线（arXiv 2609.04197, EMNLP 2026 main）— 与 APC/GEPA 同协议对垒。

ESPO 三步核心（论文 §3 忠实复现，评测货币与预算对齐 GEPA 臂）：
  Diagnose: 一轮聚类**全部**失败案例 → 结构化错误模式（对比 GEPA 的 3-例 minibatch）
  Propose : 4 个独立偏置策略各出一个候选（抽象根因/简化/范例化/约束硬化）
  Select  : bootstrap 稳定选择 — val 案例分数重采样 B 次，候选须以 ≥75% 频率
            稳定优于现任冠军才接受（防"单点均值噪声晋升"）
预算口径：task rollout 48（r0 全 val 8 + 一轮 [4 候选×8 + 收编确认 8] = 48）；
diagnose/propose 是反思类调用，不计 rollouts（与 APC 编译开销、GEPA reflect 同逻辑）。
bootstrap 复用已评案例分数，零额外调用 — ESPO 的稳定性货币是重采样，与 APC F8 的
外生漂移带货币互补（论文 F9 划界段）。

用法：.venv/bin/python scripts/bench_real_espo.py --task financial --seeds 42,43
输出并入 experiments/apcbench/real_gepa.json，method="espo"。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from apc.core.task_spec import TaskSpec
from apc.models.factory import create_client
from apc.models.mock_client import MockClient

from bench_real_full import TASK_CFG, BASE_GENOME, real_profile  # noqa: E402
from bench_real_gepa import (RolloutBudget, document, eval_cases,  # noqa: E402
                             REFLECT_SYS)
from bench_real_gepa import OUT  # 共用 json 库: real_gepa.json
from bench_real_full import REPO as _R

DIAGNOSE_SYS = """You are an error-analysis module for a prompt optimizer (ESPO Diagnose).
You receive a prompt and ALL its failing validation cases. Cluster the failures into
at most 4 distinct error PATTERNS. For each pattern: name, the cases it covers, and
the root cause in the prompt (missing rule / ambiguous instruction / wrong format
contract / overload). Be specific and quote the offending prompt fragment.
Do NOT rewrite the prompt yet."""

BIASES = [
    ("abstraction", "Fix the ROOT CAUSE at the level of the offending instruction: "
     "rewrite that fragment precisely; change nothing else."),
    ("simplification", "Remove or relax prompt rules that plausibly distract the model; "
     "shorter is better; never drop the {{input}} slot or output format contract."),
    ("exemplification", "Add one minimal input→output mini-example that directly demonstrates "
     "the fix for the dominant error pattern; keep everything else intact."),
    ("constraint-hardening", "Add explicit MUST/NEVER clauses targeting the dominant error "
     "pattern; place them near the top; keep the prompt otherwise intact."),
]


def _clean(t: str) -> str:
    for fence in ("```text\n", "```\n", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _call(client, sys_prompt, user):
    try:
        r = client.complete(sys_prompt + "\n\n" + user, temperature=0.3)
        t = _clean(r.text.strip())
        return t if t else None
    except Exception:
        return None


def _call_plain(client, user):
    try:
        return _clean(client.complete(user, temperature=0.3).text.strip())
    except Exception:
        return None


def diagnose(client, parent: str, fails: list[dict]) -> str | None:
    ex = "\n\n".join(
        f"--- CASE (acc={c['accuracy']:.2f}) ---\nINPUT: {c['doc'][:700]}\n"
        f"GOLD: {c['expected'][:400]}\nOUTPUT: {c['output'][:500]}"
        for c in fails[:12])
    return _call(client, DIAGNOSE_SYS,
                 f"PROMPT:\n<<<\n{parent}\n>>>\n\nALL FAILING CASES:\n{ex}\n\n"
                 "Output the error-pattern clusters.")


def propose(client, parent: str, patterns: str, bias_name: str, bias_instr: str,
            spec: TaskSpec) -> str | None:
    cand = _call(client, REFLECT_SYS,
                 f"CURRENT PROMPT:\n<<<\n{parent}\n>>>\n\nDIAGNOSED ERROR PATTERNS:\n"
                 f"{patterns[:4000]}\n\nYOUR ASSIGNMENT ({bias_name}):\n{bias_instr}\n\n"
                 "Output the improved complete prompt.")
    if cand and "{{input}}" in cand:
        return cand
    if cand:  # 一次修复重试:把 input 槽加回去
        fixed = _call_plain(client, f"The following prompt lost its {{{{input}}}} slot. "
                                    f"Insert {{{{input}}}} where the document content belongs "
                                    f"and output the full prompt only:\n\n{cand}")
        if fixed and "{{input}}" in fixed:
            return fixed
    return None


def bootstrap_select(cand_scores: list[list[float]], champ_scores: list[float],
                     n_boot: int = 200, thresh: float = 0.75, rng=None) -> int:
    """返回胜出候选 idx；无人稳定过阈则 -1。分数为同序 val 案例数组。"""
    rng = rng or random.Random(42)
    K = len(cand_scores)
    wins = [0] * K
    m = len(champ_scores)
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        cm = float(np.mean([champ_scores[i] for i in idx]))
        means = [float(np.mean([cs[i] for i in idx])) for cs in cand_scores]
        best = max(means)
        for k, mu in enumerate(means):
            if mu >= best - 1e-12 and mu > cm:
                wins[k] += 1
    cands = [k for k in range(K) if wins[k] / n_boot >= thresh]
    if not cands:
        return -1
    return max(cands, key=lambda k: (float(np.mean(cand_scores[k])), wins[k]))


def espo_run(client, spec, val, z0, budget: RolloutBudget, seed: int, starve=None):
    rng = random.Random(seed)
    sc0, cases0 = eval_cases(client, spec, val, z0, budget, starve=starve)
    s0 = [c["accuracy"] for c in cases0]
    fails = [c for c in cases0 if c["accuracy"] < 0.99]
    iters = 0
    if not fails or budget.left < 40:
        return z0, sc0, iters, budget.used, []
    patterns = diagnose(client, z0, fails) or "(diagnosis failed)"
    cands, cscores = [], []
    for bname, binstr in BIASES:
        c = propose(client, z0, patterns, bname, binstr, spec)
        if not c or c == z0:
            continue
        if budget.left < 8:
            break
        cs, ccases = eval_cases(client, spec, val, c, budget, starve=starve)
        cands.append((bname, c))
        cscores.append([x["accuracy"] for x in ccases])
        iters += 1
    k = bootstrap_select(cscores, s0, rng=rng) if cscores else -1
    if k >= 0:
        champ, champ_val = cands[k][1], float(np.mean(cscores[k]))
    else:
        champ, champ_val = z0, sc0
    return champ, champ_val, iters, budget.used, [b for b, _ in cands]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--hold-n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    client = create_client(args.model, prefer_mock=False)
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(420.0)
    task_yaml, ds_name, judge_id = TASK_CFG[args.task]
    spec = TaskSpec.from_yaml(REPO / task_yaml)
    from apc.core.model_profile import ModelProfile
    from apc.models.mock_client import MockClient as _MC
    profile = (ModelProfile.model_validate_json(
                   (REPO / "artifacts/profiles/qwen_probe.json").read_text(encoding="utf-8"))
               if isinstance(client, _MC) else real_profile(client))
    from apc.evaluation.dataset import load_dataset
    val = load_dataset(REPO / "datasets" / ds_name / "validation.jsonl").samples[:8]
    hold = load_dataset(REPO / "datasets" / ds_name / "holdout.jsonl").samples[:args.hold_n]

    from apc.core.genome import PromptGenome
    from apc.compiler.renderer import DefaultPromptCompiler
    z0 = DefaultPromptCompiler().compile(
        PromptGenome.from_json(str(REPO / BASE_GENOME)), spec, profile,
        apply_rules=False).prompt_text

    global OUT
    if args.max_tokens:
        OUT = REPO / "experiments" / "apcbench" / f"real_gepa_mt{args.max_tokens}.json"
    doc = {"protocol": "real-gepa", "judge": judge_id, "rows": []}
    if OUT.exists():
        doc = json.loads(OUT.read_text(encoding="utf-8"))

    for seed in [int(x) for x in args.seeds.split(",")]:
        t0 = time.time()
        budget = RolloutBudget(args.rollouts)
        champ, cval, iters, used, biases = espo_run(client, spec, val, z0, budget, seed, starve=args.max_tokens)
        sc, cases = eval_cases(client, spec, hold, champ, RolloutBudget(10_000), starve=args.max_tokens)
        (Path("/tmp") / f"espo_{args.task}_{seed}_champ.txt").write_text(champ, encoding="utf-8")
        row = {"task": args.task, "seed": seed, "method": "espo",
               "holdout_score": round(sc, 4), "validation_score": round(cval, 4),
               "pool_size": len(biases), "iterations": iters,
               "rollouts_used": used, "elapsed_s": round(time.time() - t0, 1),
               "model_version": client.model_version, "prompt_chars": len(champ),
               "fail_sample": [c for c in cases if c["accuracy"] < 0.99][:8],
               "biases_proposed": biases}
        doc["rows"] = [r for r in doc["rows"]
                       if not (r["task"] == args.task and r.get("seed") == seed
                               and r["method"] == "espo")] + [row]
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"espo {args.task} seed{seed}: hold={sc:.4f} val={cval:.4f} "
              f"adopt={'yes' if champ != z0 else 'no'} biases={biases} "
              f"ro={used} ({row['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
