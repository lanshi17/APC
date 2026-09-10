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


def _one_call(client, prompt: str, hard: float | None = None):
    """单次 fresh 连接 + 硬超时。

    教训链:① tenacity 在被服务端静默丢弃首包的 keep-alive 连接上连环重试
    (4 次全发不出去,零响应头) → 每 call 新建 OpenAIClient(全新连接池);
    ② httpx read-timeout 对半开连接不可靠(实测 145 分钟无事件) →
    用 socket 层 SO_RCVTIMEO 做内核级兜底(extract 阶段即触发)。
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor
    if getattr(client, "base_url", None):
        from apc.models.openai_client import OpenAIClient, resolve_api_key
        from apc.models.factory import load_model_config
        cfg = load_model_config(getattr(client, "_model_id", "") or "qwen")
        fresh = OpenAIClient(model_id=cfg["model_id"], model=cfg["model"], base_url=cfg["api_base"],
                             api_key=resolve_api_key(cfg), max_tokens=client.max_tokens, timeout=640.0)

        import socket as _sk, struct as _st
        import httpcore._backends.sync as _hs
        if not getattr(_hs.SyncStream, "_hle_patched", False):
            _orig_read = _hs.SyncStream.read

            def _read(self, max_bytes, timeout=None):
                try:
                    _raw = getattr(self._sock, "_sock", self._sock)
                    _raw.setsockopt(_sk.SOL_SOCKET, _sk.SO_RCVTIMEO, _st.pack("#l", 700, 0))
                except Exception:
                    pass
                return _orig_read(self, max_bytes, timeout)
            _hs.SyncStream.read = _read
            _hs.SyncStream._hle_patched = True
            _hs.SyncStream._hle_patched = True
        target = fresh
    else:
        target = client
    if not hard:
        return target.complete(prompt, 0.0)
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(target.complete, prompt, 0.0).result(timeout=hard)
    finally:
        ex.shutdown(wait=False)


def eval_cases(client, spec: TaskSpec, samples: list[dict], prompt_text: str,
               budget: RolloutBudget, cap: int | None = None,
               starve: int | None = None, stream_path: str | None = None,
               hard: float | None = None) -> tuple[float, list[dict]]:
    """与 EvaluationRunner.evaluate 完全同构的逐样本评分；rollout 计数。
    starve=任务评测的 token 预算(仅评测调用受限;优化器元调用不在此通路)。"""
    if starve:
        client.max_tokens = starve
    cases = []
    todo = samples if cap is None else samples[:cap]
    for smp in todo:
        if budget.used >= budget.limit:
            break
        doc = document(smp)
        budget.used += 1
        try:
            call = _one_call(client, prompt_text.replace("{{input}}", doc), hard)
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
        if stream_path:
            with open(stream_path, "a", encoding="utf-8") as _sf:
                _sf.write(json.dumps(cases[-1], ensure_ascii=False) + "\n")
    if starve:
        client.max_tokens = 2000
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


def gepa_search(client, spec, val_samples, z0, budget, rng, starve=None, hard=None):
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
    while budget.left >= 18:  # 亲子迭代 3+3;预留 2 候选 × val-full(8) 给 eval_full 相位
        counts = best_counts()
        tot = sum(counts) or 1
        parent_idx = rng.choices(range(len(pool)), weights=[w + 1 for w in counts])[0]  # 平滑
        parent = pool[parent_idx]["text"]
        mb = rng.sample(val_samples, 3)
        p_score, p_cases = eval_cases(client, spec, mb, parent, budget, starve=starve, hard=hard)
        fails = [c for c in p_cases if c["accuracy"] + c["instruction_following"] < 1.8
                 and c["output"] != "<call-error>"]  # 网络错误非提示词缺陷
        iters += 1
        if budget.left < 9:
            break
        child = reflect(client, parent, fails) if fails else parent
        if child == parent:
            continue
        c_score, c_cases = eval_cases(client, spec, mb, child, budget, starve=starve, hard=hard)
        record(parent_idx, p_cases, 0.0)
        if any(nc["accuracy"] + nc["instruction_following"] >
               oc["accuracy"] + oc["instruction_following"]
               for oc, nc in zip(p_cases, c_cases)):
            pool.append({"text": child, "scores": {}})
            record(len(pool) - 1, c_cases, 0.0)
    # champion：预算余量足够则 val 全量复评（GEPA 官方 eval_full）；否则用池内
    # minibatch 历史的实例均分选择（零成本降级，选择语义保持 per-instance feedback）
    champ, champ_val = 0, -1.0
    if budget.left >= len(val_samples):
        for idx, c in enumerate(pool):
            if budget.left < len(val_samples):
                break
            sc, _ = eval_cases(client, spec, val_samples, c["text"], budget, starve=starve, hard=hard)
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
    ap.add_argument("--rollouts", type=int, default=48, help="0 = 纯 z0 同日复测对照(无搜索)")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--max-tokens", type=int, default=None, help="token-starved(输出隔离 _mtN)")
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

    global OUT
    if args.max_tokens:
        OUT = REPO / "experiments" / "apcbench" / f"real_gepa_mt{args.max_tokens}.json"
    doc = {"protocol": "real-gepa", "judge": judge_id,
           "note": "GEPA=官方 reflective-Pareto 基线;预算=task rollouts;holdout 同口径", "rows": []}
    if OUT.exists():
        doc = json.loads(OUT.read_text(encoding="utf-8"))

    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = random.Random(seed)
        budget = RolloutBudget(args.rollouts)
        t0 = time.time()
        pool, champ_i, champ_val, iters = (
            ([{"text": z0, "scores": {}}], 0, 0.0, 0) if args.rollouts == 0
            else gepa_search(client, spec, val, z0, budget, rng))
        if args.rollouts == 0 and budget.left >= len(val):
            champ_val, _ = eval_cases(client, spec, val, z0, budget, starve=args.max_tokens)
        champ = pool[champ_i]["text"]
        h_score, h_cases = eval_cases(client, spec, hold, champ,
                                      RolloutBudget(10_000), starve=args.max_tokens)  # 终测不计搜索预算(同 APC)
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
