"""GEPA 官方基线（arXiv 2507.19457, arXiv 2507.19457 reflective prompt evolution +
instance-wise Pareto selection）——真实 qwen 上对 APC 的直接对比（论文 §6 标注的
"投稿必含最强基线"，本实验补上）。

与 APC 严格同口径：同一任务数据(dev/val/hold 8/20 切分)、同一 rule-judge + checker +
TrialScorer 权重、temp=0、z0 编译文本为共同起点、holdout 20 终测。预算按 **task
rollout 数** 对齐（APC b8 ≈ 42–52 rollouts；GEPA --rollouts 48 = 8 次亲子迭代×(3+3)
+ champion 池全 val 复评）。反思调用与 APC 的编译/探针开销同理不计。

用法（仓库根，需 key；长跑请 hub start）：
  .venv/bin/python scripts/bench_real_gepa.py --task financial --rollouts 48
输出：experiments/apcbench/real_gepa.json（按 task×seed 合并）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.compiler.renderer import DefaultPromptCompiler
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import Dataset, load_dataset
from apc.evaluation.judge import RuleBasedJudge
from apc.evaluation.scorer import TrialScorer
from apc.models.factory import create_client
from apc.models.mock_client import MockClient

from bench_real_full import TASK_CFG, BASE_GENOME, real_profile  # noqa: E402

OUT = REPO / "experiments" / "apcbench" / "real_gepa.json"
JUDGE = RuleBasedJudge()
CHECKER = RuleBasedChecker()

REFLECT_SYS = """You are an expert prompt optimizer (GEPA-style reflective evolution).
You will see the CURRENT prompt and up to 3 evaluation cases where the model underperformed
(input, gold expected JSON, model output, per-dimension scores).
Diagnose the root causes of the score losses, then rewrite the prompt to fix them while
preserving everything that worked. Rules:
- Keep the placeholder {{input}} exactly once.
- Keep the required output JSON schema instructions.
- Change only what the failures justify.
- Output ONLY the complete new prompt text, nothing else."""


def document(sample: dict) -> str:
    raw = sample.get("input", sample)
    return json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)


class RolloutBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    @property
    def left(self) -> int:
        return self.limit - self.used


def eval_cases(client, spec: TaskSpec, samples: list[dict], prompt_text: str,
               budget: RolloutBudget, cap: int | None = None) -> tuple[float, list[dict]]:
    """与 EvaluationRunner.evaluate 完全同构的逐样本评分；rollout 计数。"""
    cases = []
    todo = samples if cap is None else samples[:cap]
    for smp in todo:
        if budget.used >= budget.limit:
            break
        doc = document(smp)
        budget.used += 1
        try:
            call = client.complete(prompt_text.replace("{{input}}", doc), temperature=0.0)
        except Exception:
            cases.append({"sample_id": str(hash(doc))[:8], "doc": doc, "format_score": 0.0,
                          "constraint_score": 0.0, "accuracy": 0.0, "instruction_following": 0.0,
                          "output": "<call-error>", "expected": json.dumps(smp.get("expected", {}), ensure_ascii=False)})
            continue
        rule = CHECKER.check(call.text, spec)
        j = JUDGE.judge(doc, smp.get("expected", {}), call.text, spec)
        cases.append({
            "sample_id": str(hash(doc))[:8], "doc": doc,
            "format_score": rule.get("format_score", 0.0),
            "constraint_score": rule.get("constraint_score", 0.0),
            "accuracy": round(float(j.get("accuracy", 0.0)), 4),
            "instruction_following": round(float(j.get("constraint_following", 0.0)), 4),
            "output": call.text, "expected": json.dumps(smp.get("expected", {}), ensure_ascii=False),
        })
    trial = TrialScorer().score(spec, cases, trial_id=uuid.uuid4().hex[:8],
                                model_id=client.model_id, genome_id="gepa", prompt_id="gepa",
                                dataset_id="gepa", dataset_version="real",
                                judge_id="gepa_rule", temperature=0.0)
    return float(trial.score), cases


def reflect(client, parent: str, fails: list[dict]) -> str:
    ex = "\n\n".join(
        f"--- CASE (accuracy={c['accuracy']}, instruction={c['instruction_following']}, "
        f"format={c['format_score']:.2f}) ---\nINPUT: {c['doc'][:900]}\n"
        f"GOLD: {c['expected'][:500]}\nMODEL_OUTPUT: {c['output'][:700]}"
        for c in fails[:3])
    msg = (f"CURRENT PROMPT:\n<<<\n{parent}\n>>>\n\nFAILING CASES:\n{ex}\n\n"
           "Now output the improved complete prompt.")
    try:
        call = client.complete(msg, temperature=0.3)
    except Exception:
        return parent
    t = call.text.strip()
    for fence in ("```text\n", "```\n", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()
    if "{{input}}" not in t:  # 反思产物违约:回退父代(GEPA 合法性校验)
        return parent
    return t


def gepa_search(client, spec, val_samples, z0, budget, rng):
    """GEPA Algorithm 1 核心：Pareto 加权采样候选 → minibatch 反思 → 改进入池。"""
    pool: list[dict] = [{"text": z0, "scores": {}}]  # scores: doc-hash → score
    best_inst: dict[str, int] = {}                    # doc-hash → 拥有最高分的 cand idx

    def record(cand_idx: int, cases: list[dict], base: float):
        c = pool[cand_idx]
        for case in cases:
            # 实例级分:四维均值(与聚合分单调一致);同实例跨候选同 key(temp=0 确定性),重复取 max
            sc = (case["accuracy"] + case["instruction_following"]
                  + case["format_score"] + case["constraint_score"]) / 4
            k = case["sample_id"]
            c["scores"][k] = max(c["scores"].get(k, 0.0), sc)
        del base

    def best_counts() -> list[int]:
        counts = [0] * len(pool)
        tmp: dict[str, tuple[float, int]] = {}
        for idx, c in enumerate(pool):
            for k, sc in c["scores"].items():
                if k not in tmp or sc > tmp[k][0]:
                    tmp[k] = (sc, idx)
        for _, idx in tmp.values():
            counts[idx] += 1
        return counts

    iters = 0
    while budget.left >= 6:  # 一次亲子迭代需 3+3 rollouts
        counts = best_counts()
        tot = sum(counts) or 1
        parent_idx = rng.choices(range(len(pool)), weights=[w + 1 for w in counts])[0]  # 平滑
        parent = pool[parent_idx]["text"]
        mb = rng.sample(val_samples, 3)
        p_score, p_cases = eval_cases(client, spec, mb, parent, budget)
        fails = [c for c in p_cases if c["accuracy"] + c["instruction_following"] < 1.8]
        iters += 1
        if budget.left < 3:
            break
        child = reflect(client, parent, fails) if fails else parent
        if child == parent:
            continue
        c_score, c_cases = eval_cases(client, spec, mb, child, budget)
        record(parent_idx, p_cases, 0.0)
        if any(nc["accuracy"] + nc["instruction_following"] >
               oc["accuracy"] + oc["instruction_following"]
               for oc, nc in zip(p_cases, c_cases)):
            pool.append({"text": child, "scores": {}})
            record(len(pool) - 1, c_cases, 0.0)
    # champion：预算余量足够则 val 全量复评（GEPA 官方 eval_full）；否则用池内
    # minibatch 历史的实例均分选择（零成本降级，选择语义保持 per-instance feedback）
    champ, champ_val = 0, -1.0
    if budget.left >= len(val_samples) * min(3, len(pool)):
        for idx, c in enumerate(pool):
            if budget.left < len(val_samples):
                break
            sc, _ = eval_cases(client, spec, val_samples, c["text"], budget)
            if sc > champ_val:
                champ, champ_val = idx, sc
    else:
        scored = [(idx, sum(c["scores"].values()) / max(1, len(c["scores"])))
                  for idx, c in enumerate(pool) if c["scores"]]
        if scored:
            champ, champ_val = max(scored, key=lambda x: (x[1], len(pool[x[0]]["scores"])))
    return pool, champ, champ_val, iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--timeout", type=float, default=420.0)
    args = ap.parse_args()

    client = create_client(args.model, prefer_mock=False)
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(args.timeout)
    task_yaml, ds_name, judge_id = TASK_CFG[args.task]
    spec = TaskSpec.from_yaml(REPO / task_yaml)
    compiler = DefaultPromptCompiler()
    profile = real_profile(client)
    z0 = compiler.compile(PromptGenome.from_json(str(REPO / BASE_GENOME)), spec, profile,
                          apply_rules=False).prompt_text
    val = load_dataset(REPO / "datasets" / ds_name / "validation.jsonl").samples[:8]
    hold = load_dataset(REPO / "datasets" / ds_name / "holdout.jsonl").samples[:20]

    doc = {"protocol": "real-gepa", "judge": judge_id,
           "note": "GEPA=官方 reflective-Pareto 基线;预算=task rollouts;holdout 同口径", "rows": []}
    if OUT.exists():
        doc = json.loads(OUT.read_text(encoding="utf-8"))

    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = random.Random(seed)
        budget = RolloutBudget(args.rollouts)
        t0 = time.time()
        pool, champ_i, champ_val, iters = gepa_search(client, spec, val, z0, budget, rng)
        champ = pool[champ_i]["text"]
        h_score, h_cases = eval_cases(client, spec, hold, champ,
                                      RolloutBudget(10_000))  # 终测不计搜索预算(同 APC)
        row = {"task": args.task, "seed": seed, "method": "gepa",
               "holdout_score": round(h_score, 4), "validation_score": round(champ_val, 4),
               "pool_size": len(pool), "iterations": iters,
               "rollouts_used": budget.used, "elapsed_s": round(time.time() - t0, 1),
               "model_version": client.model_version,
               "prompt_chars": len(champ),
               "fail_sample": [c for c in h_cases if c["accuracy"] < 0.99][:8]}
        doc["rows"] = [r for r in doc["rows"] if not (r["task"] == args.task and r["seed"] == seed)] + [row]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        Path(f"/tmp/gepa_{args.task}_{seed}_champ.txt").write_text(champ, encoding="utf-8")
        print(f"{args.task} seed{seed}: hold={row['holdout_score']:.4f} val={champ_val:.4f} "
              f"pool={len(pool)} iters={iters} rollouts={budget.used} ({row['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
