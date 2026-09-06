# -*- coding: utf-8 -*-
"""数学应用题数据集生成器（确定种子 20260909，离线可复现）。

三类整数安全题型（答案精确整数；文档中只出现解题相关数字）：
- sum：甲购入A件单价P，乙购入B件单价Q，共花费 = A*P+B*Q
- discount：标价M，打D折，售价 = M*D//10（M 取整十）
- average：三天产量 a/b/c，平均 = sum//3（和被 3 整除）
噪声句均为定性句（不含阿拉伯数字，避免干扰抽取）。
四分割：dev 40 / validation 30 / holdout 30。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOISE = [
    "本题为小学数学应用题，请分步计算。",
    "请注意单位统一后再作答。",
    "计算过程保留整数结果即可。",
]


def gen_sample(rng: random.Random) -> dict:
    kind = rng.choice(["sum", "discount", "average"])
    if kind == "sum":
        a, p = rng.randint(2, 12), rng.randint(5, 60)
        b, q = rng.randint(2, 12), rng.randint(5, 60)
        ans = a * p + b * q
        doc = (f"甲购入{a}件单价为{p}元的货物，"
               f"乙购入{b}件单价为{q}元的同款货物，两人共花费多少元？"
               f"{rng.choice(NOISE)}")
    elif kind == "discount":
        m, d = rng.randint(1, 9) * 100, rng.randint(1, 9)
        ans = m * d // 10
        doc = (f"某商品标价{m}元，现打{d}折出售，打折后售价多少元？"
               f"{rng.choice(NOISE)}")
    else:
        vals = [rng.randint(10, 99) for _ in range(3)]
        vals[2] += (-sum(vals)) % 3  # 和被 3 整除
        ans = sum(vals) // 3
        doc = (f"某车间三天产量分别为{vals[0]}件、{vals[1]}件、{vals[2]}件，"
               f"平均每天产量多少件？{rng.choice(NOISE)}")
    steps = ["提取题干中的数值", "列式并分步计算", "核对结果与单位"]
    return {
        "input": {"document": doc},
        "expected": {"answer": str(ans), "steps": steps,
                     "confidence": round(rng.uniform(0.6, 0.9), 2)},
    }


def main():
    out_dir = REPO / "datasets" / "math_reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260909)
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
