"""外部真实推理基准（Limitation 2 回应）：GSM8K / MATH Level-5 / AIME24+25 上的
zero-adapt genome 迁移验证。三臂 = base genome / math 冠军(transfer-0) / contract 冠军，
同一外部 TaskSpec 编译，exact-match accuracy 判分（normalize 数值/latex 结构）。

结论方向：若 transfer-0 臂 ≥ base ⇒ F3 零损耗迁移在外部基准（作者未造的任务）
上成立；若各臂都满 ⇒ F1 饱和的外部复现；若冠军 < base ⇒ 先验负债的外部证据。

用法（仓库根，需 key）：
  .venv/bin/python scripts/bench_real_external.py --dataset aime --arms base,math-champ,contract-champ
输出：experiments/apcbench/real_external.json（按 dataset×arm checkpoint 合并）
数据：datasets/gsm8k/test.jsonl (OpenAI grade-school-math, MIT),
     datasets/hendrycks_math/test.jsonl (HuggingFaceH4/MATH, MIT),
     datasets/aime/test.jsonl (AIME 2024+2025 via HF4/math-ai)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apc-pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.compiler.renderer import DefaultPromptCompiler
from apc.models.factory import create_client
from apc.models.mock_client import MockClient

from bench_real_full import real_profile  # noqa: E402

EXT_SPEC = REPO / "apc-pipeline" / "configs" / "tasks" / "external_math.yaml"
CHAMPS = REPO / "artifacts" / "optimizations"
OUT = REPO / "experiments" / "apcbench" / "real_external.json"  # 默认;--out 覆盖

_NUMPAT = re.compile(r"[\d+\-*/.()eE]+")
_MATRIX = (r"\\begin\{([pbvB]?matrix|cases|array)(?:\[[^\]]*\])?\}(?:\{[^{}]*\})?"
           r"(.*?)\\end\{\1\}")


def _scan_from(s: str, i: int) -> str | None:
    depth, out = 0, []
    for ch in s[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def last_boxed_full(t: str) -> str | None:
    """整个字符串恰为 \boxed{…} 时返回其内容(支持嵌套)，否则 None。"""
    if not re.match(r"^\\boxed\s*\{", t):
        return None
    return _scan_from(t, t.find("{"))


def last_boxed_any(s: str) -> str | None:
    """取最后一个 \boxed{…} 内容，支持嵌套花括号。"""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    return _scan_from(s, i) if i >= 0 else None


def _split_top(body: str) -> list[str]:
    """按 depth-0 的 , 与行分隔 \\\\ 切分（嵌套括号内的逗号不切）。"""
    parts, cur, depth, i = [], [], 0, 0
    while i < len(body):
        if body.startswith("\\\\", i):
            if depth == 0:
                parts.append("".join(cur))
                cur = []
            i += 2
            continue
        ch = body[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p for p in parts if p.strip()]


_UNICODE = {"\u221a": "\\sqrt", "\u03c0": "\\pi", "\u00b0": "^\\circ",
            "\u2212": "-", "\u00d7": "*", "\u2264": "<=", "\u2265": ">="}
_PLACE_RE = re.compile(r"\x01(\d+)\x02")


def norm_answer(s: object) -> str:
    r"""答案规约 → 可比对规范串；判不准时按错（保守）。

    步骤：控制符还原 → boxed 剥壳 → unicode → text/left-right/空白/$/% → 千分位
    → sqrt{}→sqrt() → frac→()/() → 矩阵/cases 括号 stash(有序) → 方括号/裸表 set 排序
    stash → 命令去反斜杠 + circ/花括号/圆括号/乘号剥离 → 占位还原 → 数值 float 规范。
    圆括号结构保持顺序；方括号与裸逗号表按集合（排序）⇒ 坐标 (1,2)≠(2,1) 不误判。
    """
    t = str(s).strip().replace("\x08", "\\b").replace("\x0c", "\\f")
    for _ in range(3):
        inner = last_boxed_full(t)
        if inner is None:
            break
        t = inner.strip()
    for k, v in _UNICODE.items():
        t = t.replace(k, v)
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    for frag in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\quad"):
        t = t.replace(frag, "")
    t = t.replace(" ", "").replace("$", "").replace("\\%", "").replace("%", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
    for _ in range(3):
        t2 = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", t)
        if t2 == t:
            break
        t = t2
    store: list[str] = []

    def stash(parts: list[str], ordered: bool) -> str:
        elems = sorted(norm_answer(x) for x in parts)  # multiset: 一切括号列表无序比对
        del ordered
        store.append("(" + ",".join(elems) + ")")
        return f"\x01{len(store) - 1}\x02"

    t = re.sub(_MATRIX, lambda m: stash(_split_top(m.group(2)), False), t, flags=re.S)
    if not _PLACE_RE.search(t):
        m = re.fullmatch(r"\[([^\[\]]+)\]", t)
        if m and len(ps := _split_top(m.group(1))) >= 2:
            t = stash(ps, False)
        elif m := re.fullmatch(r"\((.+)\)", t):
            if len(ps := _split_top(m.group(1))) >= 2:
                t = stash(ps, False)
        elif len(ps := _split_top(t)) >= 2 and t[:1] not in "([" and max(map(len, ps)) <= 40:
            t = stash(ps, False)  # 裸逗号表 = 解集(无序)
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)
    t = re.sub(r"\^?circ", "", t).replace("\\", "")
    t = re.sub(r"[{}]", "", t)
    t = t.replace("*", "").replace("(", "").replace(")", "")
    t = re.sub(r"\^+$", "", t).rstrip(".")
    for _ in range(3):
        t = _PLACE_RE.sub(lambda m: store[int(m.group(1))], t)
    if _NUMPAT.fullmatch(t):
        try:
            return f"{float(eval(t)):.6g}"
        except Exception:
            pass
    return t.lower()


def answers_match(pred: object, gold: object) -> bool:
    return norm_answer(pred) == norm_answer(gold)


def extract_pred(text: str) -> str:
    t = text.strip()
    if "{" in t and "}" in t:
        try:
            d = json.loads(t[t.find("{"): t.rfind("}") + 1])
            if isinstance(d, dict) and "answer" in d:
                ans = str(d["answer"])
                # 模型常把 \boxed 直写进 JSON 字符串：\b/\f 被解成控制字符，需还原
                return ans.replace("\x08", "\\b").replace("\x0c", "\\f")
        except Exception:
            pass
    b = last_boxed_any(text)
    if b is not None:
        return b
    m = re.findall(r"(?:final answer|答案|answer)[:：]\s*(.+)", text, re.I)
    return m[-1].strip() if m else ""


def load_problems(dataset: str, n: int) -> list[dict]:
    if dataset == "gsm8k":
        raw = (REPO / "datasets/gsm8k/test.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        random.Random(42).shuffle(rows)
        return [{"problem": r["question"], "gold": r["answer"].split("####")[-1].strip()}
                for r in rows[:n]]
    if dataset == "math5":
        raw = (REPO / "datasets/hendrycks_math/test.jsonl").read_text(encoding="utf-8")
        rows = [r for r in map(json.loads, raw.splitlines()) if r.get("level") == "Level 5"]
        return [{"problem": r["problem"],
                 "gold": last_boxed_any(r["solution"]) or r["solution"].strip()[-60:]}
                for r in rows[:n]]
    if dataset == "aime":
        raw = (REPO / "datasets/aime/test.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        return [{"problem": r["problem"] + "\n(Answer with an integer from 0 to 999.)",
                 "gold": r["answer"]} for r in rows[:n]]
    raise SystemExit(f"未知数据集 {dataset}")


def arm_genome(name: str) -> PromptGenome:
    if name == "base":
        return PromptGenome.from_json(str(REPO / "configs/genomes/base.json"))
    fn = {"math-champ": "real_math_champ.json", "contract-champ": "real_contract_champ.json",
          "financial-champ": "real_financial_champ.json"}[name]
    return PromptGenome.model_validate_json((CHAMPS / fn).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math5", "aime"])
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--arms", default="base,math-champ,contract-champ")
    ap.add_argument("--n", type=int, default=135)
    ap.add_argument("--out", default=None, help="独立输出文件(并行分数据集防写竞态)")
    ap.add_argument("--timeout", type=float, default=420.0, help="单请求读超时(AIME 长 CoT+并发)")
    args = ap.parse_args()

    client = create_client(args.model, prefer_mock=False)
    if hasattr(client, "client"):
        import httpx
        client.client.timeout = httpx.Timeout(args.timeout)
    if isinstance(client, MockClient):
        raise SystemExit("需要真实凭证")
    spec = TaskSpec.from_yaml(str(EXT_SPEC))
    compiler = DefaultPromptCompiler()
    profile = real_profile(client)
    problems = load_problems(args.dataset, args.n)

    out = Path(args.out) if args.out else OUT
    doc = {"protocol": "real-external", "judge": "exact-match-normalized-v2", "rows": []}
    if out.exists():
        doc = json.loads(out.read_text(encoding="utf-8"))

    for arm in [a.strip() for a in args.arms.split(",")]:
        g = arm_genome(arm)
        cp = compiler.compile(g, spec, profile, apply_rules=False)
        t0, acc, json_ok, err, fails = time.time(), 0, 0, 0, []
        for i, p in enumerate(problems):
            try:
                call = client.complete(cp.prompt_text.replace("{{input}}", p["problem"]), temperature=0.0)
            except Exception:
                err += 1
                fails.append({"i": i, "pred": "<call-error>", "gold": str(p["gold"])[:60]})
                continue
            try:
                dd = json.loads(call.text[call.text.find("{"): call.text.rfind("}") + 1])
                json_ok += int(isinstance(dd, dict) and "answer" in dd)
            except Exception:
                pass
            pred = extract_pred(call.text)
            ok = answers_match(pred, p["gold"]) if pred != "" else False
            acc += ok
            if not ok:
                fails.append({"i": i, "pred": pred[:60], "gold": str(p["gold"])[:60]})
        row = {"dataset": args.dataset, "arm": arm, "n": len(problems),
               "accuracy": round(acc / len(problems), 4),
               "answer_field_rate": round(json_ok / len(problems), 4),
               "call_errors": err,
               "budget_used": 0, "elapsed_s": round(time.time() - t0, 1),
               "model_version": client.model_version, "fail_sample": fails[:12]}
        doc["rows"] = [r for r in doc["rows"] if not (r["dataset"] == args.dataset and r["arm"] == arm)] + [row]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{args.dataset:6s} {arm:15s} acc={row['accuracy']:.4f} json_ok={row['answer_field_rate']:.4f} "
              f"({row['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
