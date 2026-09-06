# -*- coding: utf-8 -*-
"""合同抽取数据集生成器（确定种子 20260908，离线可复现）。

文档模板与 Mock 抽取正则严格对齐：
- 甲乙方：`甲方{name}[公司]` / `乙方{name}[公司]`
- 金额：`合同金额{num}{unit}`
- 日期：`签订日期：YYYY-MM-DD`
- 义务：`；{义务句}`（每句含 义务/责任/提供/支付/交付/配合/保密 之一）
四分割：dev 40 / validation 30 / holdout 30（无 perturbation）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PARTY_A = ["华信科技", "恒达实业", "明州电子", "长风物流", "安泰制药", "博雅文化"]
PARTY_B = ["云帆贸易", "瑞丰制造", "星野软件", "通达建材", "康宁食品", "卓越咨询"]
SUFFIX = ["公司", "公司", "集团", ""]
UNITS = ["万元", "万元", "亿元", "万元", "元"]
OBL_TEMPLATES = [
    "甲方应按约定提供技术资料并配合验收",
    "乙方应承担设备安装调试的全部责任",
    "甲方应在验收后30日内支付合同款项",
    "乙方应按期交付全部货物并承担运费",
    "双方应对合作内容承担保密义务",
    "乙方应提供一年期免费售后维护义务",
    "甲方应配合乙方完成现场勘测义务",
    "乙方对产品质量问题承担更换责任",
]
DISTRACT = "本合同已经过双方法务审核确认。"


def gen_sample(rng: random.Random) -> dict:
    a = rng.choice(PARTY_A) + rng.choice(SUFFIX)
    b = rng.choice(PARTY_B) + rng.choice(SUFFIX)
    unit = rng.choice(UNITS)
    num = rng.randint(5, 900) if unit != "元" else rng.randint(1000, 99999)
    if rng.random() < 0.4:
        num = round(num + rng.randint(1, 9) / 10, 1)
    amount = f"{num}{unit}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    n_obl = rng.randint(2, 4)
    obls = rng.sample(OBL_TEMPLATES, n_obl)
    doc = (f"甲方{a}与乙方{b}签订本合同。合同金额{amount}。"
           f"签订日期：{date}。" + "；".join([""] + obls) + f"。{DISTRACT}")
    return {
        "input": {"document": doc},
        "expected": {
            "parties": [a.replace("公司", "").replace("集团", ""), b.replace("公司", "").replace("集团", "")],
            "amount": amount,
            "date": date,
            "obligations": obls,
            "confidence": round(rng.uniform(0.6, 0.9), 2),
        },
    }


def main():
    out_dir = REPO / "datasets" / "contract_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260908)
    splits = {"dev": 40, "validation": 30, "holdout": 30}
    seen = set()
    for split, n in splits.items():
        samples = []
        while len(samples) < n:
            s = gen_sample(rng)
            if s["input"]["document"] in seen:
                continue
            seen.add(s["input"]["document"])
            samples.append(s)
        path = out_dir / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in samples) + "\n",
                        encoding="utf-8")
        print(f"{split}: {n} samples -> {path}")


if __name__ == "__main__":
    main()
