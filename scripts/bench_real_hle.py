"""HLE-exact 高难对垒(F11):真实 headroom regime 下 APC genome vs GEPA vs ESPO。

动机:F1-F10 的真实 API 任务全部饱和或物理不可达;F10 regime map 指出判别力需要
"可达 headroom"。HLE exact-match 子集(90 题 30/30/30,数学/物理/化学分层)同时具备:
  (a) 答案正确率有余量(题目难度 → acc ~.2-.6)
  (b) **协议余量**:external_math 严格 JSON 输出 {answer:...};reasoning 模型常在
      截断/全文混排中违反协议 → format_score=0 → 这是 genome 格式槽的机制强项
判分:hardened exact-match judge v3.2(norm_answer/answers_match/extract_pred 逐字复用
bench_real_external)+ TrialScorer 权重(accuracy .60 / if .15 / format .15 / 其余 .10)。

臂:base(z0) | transfer champs(math/contract/financial) | gepa | espo | apc 双臂搜索。
用法:
  .venv/bin/python scripts/bench_real_hle.py --arms z0 [--hold-n 30]           # pilot
  .venv/bin/python scripts/bench_real_hle.py --arms z0,champs,gepa,espo,search  # 全矩阵
结果:experiments/apcbench/real_hle.json(seed 921+,与主表隔离)。
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
from apc.models.factory import create_client
from apc.models.mock_client import MockClient
from apc.evaluation.runner import TrialScorer

from bench_real_full import real_profile, CHAMPS_DIR  # noqa: E402
from bench_real_external import (answers_match, extract_pred, JUDGE_VERSION)  # noqa: E402
from bench_real_gepa import RolloutBudget, eval_cases, gepa_search  # noqa: E402
import bench_real_gepa as G  # 用于 monkeypatch JUDGE/CHECKER 到 hle 语义

import os
OUT = Path(os.environ.get("HLE_OUT", str(REPO / "experiments" / "apcbench" / "real_hle.json")))
SPEC_PATH = REPO / "apc-pipeline" / "configs" / "tasks" / "external_math.yaml"
DS = REPO / "datasets" / "hle_exact"


class HLEJudge:
    """bench_gepa 通路 JUDGE 适配:accuracy=answers_match(gold, extracted)。"""

    def judge(self, doc: str, expected: dict, output: str, spec: TaskSpec) -> dict:
        pred = extract_pred(output)
        acc = 1.0 if pred and answers_match(pred, expected.get("answer", "")) else 0.0
        return {"accuracy": acc, "constraint_following": 1.0 if output.strip() else 0.0}


class HLEChecker:
    """CHECKER 适配:format = 输出可解析出 JSON 对象且含 answer 键。"""

    def check(self, text: str, spec: TaskSpec) -> dict:
        t = text.strip()
        ok = False
        if "{" in t and "}" in t:
            try:
                d = json.loads(t[t.find("{"): t.rfind("}") + 1])
                ok = isinstance(d, dict) and "answer" in d and str(d["answer"]).strip() != ""
            except Exception:
                ok = False
        return {"format_score": 1.0 if ok else 0.0, "constraint_score": 1.0 if ok else 0.0}


def load(part: str, n: int) -> list[dict]:
    rows = [json.loads(l) for l in (DS / f"{part}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [{"doc": r["input"], "expected": {"answer": r["expected"]["answer"]},
             "question": r["input"]} for r in rows[:n]]


def compile_for(genome: PromptGenome, spec, profile, compiler) -> str:
    return compiler.compile(genome, spec, profile, apply_rules=False).prompt_text


def make_client(model: str, timeout: float):
    """每次评测新建客户端:全新连接池 + 1-attempt(错误即落 <call-error>,由外层重跑续)。"""
    import tenacity as _tc
    from apc.models.openai_client import OpenAIClient, resolve_api_key
    from apc.models.factory import load_model_config
    cfg = load_model_config(model)
    c = OpenAIClient(model_id=cfg["model_id"], model=cfg["model"], base_url=cfg["api_base"],
                     api_key=resolve_api_key(cfg), max_tokens=cfg.get("max_tokens", 2000),
                     timeout=timeout)
    c.client = type(c.client)(timeout=__import__("httpx").Timeout(timeout, connect=30.0, read=timeout))
    _orig = c.complete
    c.complete = _tc.retry(stop=_tc.stop_after_attempt(1))(
        lambda prompt, temperature=0.0: _orig.__wrapped__(prompt, temperature)
        if hasattr(_orig, "__wrapped__") else _orig(prompt, temperature))
    return c


def run_eval(client, spec, samples, prompt_text, budget, stream_path=None,
               hard=330.0, timeout=300.0) -> tuple[float, list]:
    """带 per-sample 流式落盘 + 断点续跑的评测(代理挂起/进程死后可重入)。"""
    if not stream_path:
        return eval_cases(client, spec, samples, prompt_text, budget)
    done = {}
    sp = Path(stream_path)
    if sp.exists():
        for line in sp.read_text(encoding="utf-8").splitlines():
            try:
                c = json.loads(line)
                done[c["sample_id"]] = c
            except Exception:
                pass
    from apc.evaluation.runner import TrialScorer
    from concurrent.futures import ThreadPoolExecutor
    import uuid

    def _hard_call(prompt, hard):
        """每次调用独立 executor:future 超时后 shutdown(wait=False) 不阻塞主循环。"""
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(client.complete, prompt, 0.0).result(timeout=hard)
        finally:
            ex.shutdown(wait=False)

    cases = []
    for smp in samples:
        sid = str(hash(smp["doc"]))[:8]
        if sid in done:
            cases.append(done[sid]); continue
        prompt = prompt_text.replace("{{input}}", smp["doc"])
        try:
            call = _hard_call(prompt, hard)
        except Exception as e:
            print(f"  hard-fail {sid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            call = None
        if call is None:
            # 换独立新连接重试一次(绕开可能被污染的 keep-alive 连接)
            try:
                import apc.models.openai_client as OC
                fresh = OC.OpenAIClient(model_id=client.model_id, model=client.model,
                                        base_url=client.base_url, api_key=client.api_key,
                                        max_tokens=client.max_tokens, timeout=timeout)
                call = fresh.client.post(f"{fresh.base_url}/chat/completions",
                                         headers={"Authorization": f"Bearer {fresh.api_key}"} if fresh.api_key else {},
                                         json={"model": fresh.model,
                                               "messages": [{"role": "user", "content": prompt}],
                                               "temperature": 0.0,
                                               "max_tokens": fresh.max_tokens},
                                         timeout=(timeout + 40, 30.0))
                j = call.json()
                class _C:  # 统一返回形状
                    text = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                call = _C()
            except Exception as e2:
                print(f"  retry-fail {sid}: {type(e2).__name__}: {str(e2)[:120]}", flush=True)
                call = None
        if call is None:
            c = {"sample_id": sid, "doc": smp["doc"], "format_score": 0.0, "constraint_score": 0.0,
                 "accuracy": 0.0, "instruction_following": 0.0, "output": "<call-error>",
                 "expected": json.dumps(smp.get("expected", {}), ensure_ascii=False)}
        else:
            rule = G.CHECKER.check(call.text, spec)
            j = G.JUDGE.judge(smp["doc"], smp.get("expected", {}), call.text, spec)
            c = {"sample_id": sid, "doc": smp["doc"], "format_score": rule.get("format_score", 0.0),
                 "constraint_score": rule.get("constraint_score", 0.0),
                 "accuracy": round(float(j.get("accuracy", 0.0)), 4),
                 "instruction_following": round(float(j.get("constraint_following", 0.0)), 4),
                 "output": call.text, "expected": json.dumps(smp.get("expected", {}), ensure_ascii=False)}
        cases.append(c)
        with open(sp, "a", encoding="utf-8") as f:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    trial = TrialScorer().score(spec, cases, trial_id=uuid.uuid4().hex[:8],
                                model_id=client.model_id, genome_id="hle", prompt_id="hle",
                                dataset_id="hle", dataset_version="real",
                                judge_id="hle-v32", temperature=0.0)
    return float(trial.score), cases


def row_of(method, seed, sc, cases, t0, chars, budget, extra=None):
    r = {"task": "hle_exact", "seed": seed, "method": method,
         "holdout_score": round(float(sc), 4),
         "n_acc": sum(1 for c in cases if c["accuracy"] > 0.5),
         "n_format": sum(1 for c in cases if c["format_score"] > 0.5),
         "n": len(cases),
         "rollouts_used": budget.used if budget else None,
         "elapsed_s": round(time.time() - t0, 1), "prompt_chars": chars,
         "fail_sample": [c["sample_id"] for c in cases
                         if c["accuracy"] < 0.5 or c["format_score"] < 0.5][:10]}
    if extra:
        r.update(extra)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="z0", help="逗号分隔: z0,champs,gepa,espo,search")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--hold-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=921)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--hard", type=float, default=330.0, help="thread 级硬超时(read timeout 对半开连接可失效)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="关推理模型思考通路:题目仍难但单题成本 x10(协议余量主信号)")
    args = ap.parse_args()

    client = make_client(args.model, args.timeout)
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(args.timeout, connect=30.0, read=args.timeout)
    if args.no_thinking:
        _orig_complete = client.complete
        def _nt(messages, max_tokens=2000, temperature=0.0):
            return _orig_complete(messages, max_tokens=max_tokens, temperature=temperature,
                                  extra_body={"enable_thinking": False})
        client.complete = _nt
    # 判分适配:monkeypatch gepa 全局 JUDGE/CHECKER(hle 语义)
    G.JUDGE = HLEJudge()
    G.CHECKER = HLEChecker()
    G.document = lambda smp: smp["doc"]  # hle 样本无 document 字段映射

    spec = TaskSpec.from_yaml(str(SPEC_PATH))
    compiler = DefaultPromptCompiler()
    profile = real_profile(client)
    hold = load("holdout", args.hold_n)
    val = load("validation", 30)
    base = PromptGenome.from_json(str(REPO / "configs/genomes/base.json"))
    z0_text = compile_for(base, spec, profile, compiler)

    doc = {"protocol": "real-hle", "judge": JUDGE_VERSION, "rows": []}
    if OUT.exists():
        doc = json.loads(OUT.read_text(encoding="utf-8"))
    arms = {m.strip() for m in args.arms.split(",")}

    def persist(r):
        doc["rows"] = [x for x in doc["rows"]
                       if not (x["method"] == r["method"] and x.get("seed") == r["seed"])] + [r]
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{r['method']:16s} hold={r['holdout_score']:.4f} acc={r['n_acc']}/{r['n']} "
              f"fmt={r['n_format']}/{r['n']} ({r['elapsed_s']}s)", flush=True)

    if "z0" in arms:
        t0 = time.time(); b = RolloutBudget(10_000)
        sc, cases = run_eval(client, spec, hold, z0_text, b, f"/tmp/hle_z0_{args.seed}.jsonl",
                           hard=args.hard, timeout=args.timeout)
        persist(row_of("z0", args.seed, sc, cases, t0, len(z0_text), b))

    if "champs" in arms:
        for nm, fn in (("math-champ", "real_math_champ.json"),
                       ("contract-champ", "real_contract_champ.json"),
                       ("financial-champ", "real_financial_champ.json")):
            t0 = time.time(); b = RolloutBudget(10_000)
            g = PromptGenome.model_validate_json((CHAMPS_DIR / fn).read_text(encoding="utf-8"))
            text = compile_for(g, spec, profile, compiler)
            sc, cases = run_eval(client, spec, hold, text, b, f"/tmp/hle_{nm}_{args.seed}.jsonl",
                               hard=args.hard, timeout=args.timeout)
            persist(row_of(nm, args.seed, sc, cases, t0, len(text), b))

    if "gepa" in arms:
        t0 = time.time(); b = RolloutBudget(args.rollouts)
        champ, cval, iters, used = gepa_search(client, spec, val[:8], z0_text, b,
                                               random.Random(args.seed))
        (Path("/tmp") / f"hle_gepa_{args.seed}_champ.txt").write_text(champ, encoding="utf-8")
        b2 = RolloutBudget(10_000)
        sc, cases = run_eval(client, spec, hold, champ, b2, f"/tmp/hle_gepa_{args.seed}.jsonl",
                           hard=args.hard, timeout=args.timeout)
        persist(row_of("gepa", args.seed, sc, cases, t0, len(champ), b,
                       {"validation_score": round(cval, 4), "iterations": iters}))

    if "espo" in arms:
        from bench_real_espo import espo_run
        t0 = time.time(); b = RolloutBudget(args.rollouts)
        champ, cval, iters, used, biases = espo_run(client, spec, val[:8], z0_text, b, args.seed)
        (Path("/tmp") / f"hle_espo_{args.seed}_champ.txt").write_text(champ, encoding="utf-8")
        b2 = RolloutBudget(10_000)
        sc, cases = run_eval(client, spec, hold, champ, b2, f"/tmp/hle_espo_{args.seed}.jsonl",
                           hard=args.hard, timeout=args.timeout)
        persist(row_of("espo", args.seed, sc, cases, t0, len(champ), b,
                       {"validation_score": round(cval, 4), "biases": biases}))

    if "search" in arms:
        # 在线 genome 搜索臂留待多模型轮;F11 判别由 champs(预编译结构先验) vs
        # gepa/espo(反思发现)承担 — 两条通路对协议余量的利用方式正是要对比的对象。
        print("search arm skipped by design (champs+gepa+espo 足以判别)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
