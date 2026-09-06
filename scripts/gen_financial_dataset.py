# -*- coding: utf-8 -*-
"""APCBench 生成器：合成财务报告样本(确定种子,离线可复现)。

每样本 5-11 个数值指标 + 2-4 个风险叙事 + 多余干扰句(robustness_to_distraction 信号源)。
四分割: dev 40 / validation 30 / holdout 30 / perturbation 30。
金标准由生成器直接给出(规则可判),保证 Mock 模式下评分有区分度。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RISK_POOL = [
    ("应收账款回收风险", "应收账款{a}亿元，同比增加{p}%"),
    ("现金流压力风险", "经营性现金流净额{a}亿元，同比下降{p}%"),
    ("存货积压风险", "存货周转天数增至{a}天"),
    ("毛利率下滑风险", "毛利率{a}%，较上年同期下降{p}个百分点"),
    ("需求走弱风险", "在手订单{a}亿元，同比下降{p}%"),
    ("债务集中偿付风险", "短期借款{a}亿元，货币资金{b}亿元"),
    ("汇率波动风险", "海外收入占比{a}%"),
    ("客户集中度风险", "前五大客户收入占比{a}%"),
]
UNITS = ["亿元", "亿元", "亿元", "万元", "%"]
CHANGE_VERBS = [("同比增长", "+"), ("同比上升", "+"), ("同比增长", "+"),
                ("同比下降", "-"), ("同比减少", "-"), ("环比增长", "+"), ("环比下降", "-")]
DISTRACTORS = [
    "公司管理层对未来经营保持谨慎乐观态度。",
    "本报告已经过董事会审议通过。",
    "审计机构对财务报表出具了标准无保留意见。",
    "公司将继续加大研发投入以保持技术领先。",
    "报告期内公司完成了办公楼装修升级工程。",
]
IND_NAMES = ["营收", "营业收入", "净利润", "归母净利润", "毛利率", "研发费用",
             "经营性现金流净额", "应收账款", "存货", "合同负债", "销售费用"]


def gen_sample(rng: random.Random, idx: int) -> dict:
    year, quarter = rng.choice([2025, 2026]), rng.choice(["Q1", "Q2", "Q3", "Q4"])
    n_metrics = rng.randint(4, 7)
    ind_names = rng.sample(IND_NAMES, n_metrics)
    lines = [f"{year}年{quarter}财报："]
    metrics, risk_ctx = [], {}
    for name in ind_names:
        unit = rng.choice(UNITS)
        base = rng.randint(8, 450) if unit == "亿元" else rng.randint(90, 99999)
        if unit == "%":
            base = rng.randint(3, 89)
        value = base + (rng.randint(1, 9) / 10 if rng.random() < 0.5 else 0)
        verb, sign = rng.choice(CHANGE_VERBS)
        pct = rng.randint(2, 48)
        if name == "毛利率":
            unit = "%"
            value = rng.randint(18, 82)
        lines.append(f"{name}{value}{unit}，{verb}{pct}%。")
        metrics.append({"name": name, "value": f"{value}{unit}",
                        "change": f"{sign}{pct}%", "comment": verb})
    n_risks = rng.randint(2, 4)
    risk_names, risk_lines = [], []
    for rname, tpl in rng.sample(RISK_POOL, n_risks):
        a = rng.randint(2, 60)
        p = rng.randint(5, 70)
        b = rng.randint(2, 60)
        risk_lines.append(tpl.format(a=a, p=p, b=b))
        risk_names.append(rname)
    lines += risk_lines
    lines.append(rng.choice(DISTRACTORS))
    doc = "\n".join(lines)
    up = sum(1 for m in metrics if m["change"].startswith("+"))
    trend = "整体向好" if up >= n_metrics - up else "整体承压"
    return {
        "input": {"document": doc},
        "expected": {
            "summary": f"{quarter}营收与利润{trend}",
            "metrics": metrics,
            "risks": risk_names,
            "confidence": round(rng.uniform(0.6, 0.9), 2),
        },
    }


def main():
    out_dir = REPO / "datasets" / "financial_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260907)
    splits = {"dev": 40, "validation": 30, "holdout": 30, "perturbation": 30}
    seen_docs, all_samples = set(), []
    for split, n in splits.items():
        samples = []
        while len(samples) < n:
            s = gen_sample(rng, len(all_samples))
            if s["input"]["document"] in seen_docs:
                continue
            seen_docs.add(s["input"]["document"])
            all_samples.append(s)
            if split == "perturbation":
                # 扰动集（金标准保持清洁版答案）：
                # (a) 追加 2 条干扰句；(b) 将首个指标值替换为不同数字。
                # 模型若照搬被扰动数值则与金标准失配 → 可测量的稳健性下降。
                extra = "\n".join(rng.sample(DISTRACTORS, 2))
                s["input"]["document"] += "\n" + extra
                gm = s["expected"]["metrics"][0]
                old_val = gm["value"]
                m = re.match(r"^([0-9.]+)(.*)$", old_val)
                if m:
                    num = float(m.group(1))
                    new_num = round(num * 1.5 + 1.7, 1)
                    new_val = (str(int(new_num)) if float(new_num).is_integer()
                               else str(new_num)) + m.group(2)
                    s["input"]["document"] = s["input"]["document"].replace(
                        old_val, new_val, 1)
            samples.append(s)
        path = out_dir / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in samples) + "\n",
                        encoding="utf-8")
        print(f"{split}: {n} samples -> {path}")


if __name__ == "__main__":
    main()
