# -*- coding: utf-8 -*-
"""可验证约束遵循数据集生成器（确定种子 20260910，离线可复现）。

样本空间：20 主题 × 8 背景 × 3 口吻 × 2 句数 = 960 ≫ 100（防去重死锁，
另有尝试上限守卫）。
每样本：主题 + 背景 + 口吻 + 最大句数（2~3）+ 关键词 + 禁数字。
文档只含主题与要求，不含答案文本（模型须自己组织语言）。
三分割：dev 40 / validation 30 / holdout 30。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TOPICS = [
    ("城市绿化", "绿荫"), ("秋日丰收", "金黄"), ("海滨度假", "浪花"),
    ("山间晨雾", "缭绕"), ("夜市小吃", "飘香"), ("春日踏青", "新绿"),
    ("冬日暖阳", "融融"), ("乡村夜晚", "蛙鸣"), ("湖畔垂柳", "依依"),
    ("草原驰骋", "辽阔"), ("雪后故宫", "皑皑"), ("江南烟雨", "朦胧"),
    ("沙漠驼铃", "悠远"), ("竹林听雨", "潇潇"), ("枫叶正红", "似火"),
    ("麦浪滚滚", "丰登"), ("荷塘月色", "清香"), ("溪流潺潺", "叮咚"),
    ("云海日出", "壮阔"), ("古镇石桥", "斑驳"),
]
CTXS = [
    "街道两旁种满了行道树，夏日出行有了阴凉。",
    "田野里的稻穗已经成熟，风吹过泛起波浪。",
    "沙滩细软洁净，海水清澈见底。",
    "清晨的山谷被雾气笼罩，远山若隐若现。",
    "整条街摆满了摊位，空气中弥漫着食物香气。",
    "公园里的草地返青，人们纷纷出门游玩。",
    "午后的阳光洒在身上，驱散了寒意。",
    "池塘边的叫声此起彼伏，星空格外明亮。",
]
TONES = ["请以写景口吻作答", "请以抒情口吻作答", "请客观描述作答"]


def gen_sample(rng: random.Random) -> dict:
    topic, keyword = rng.choice(TOPICS)
    ctx = rng.choice(CTXS)
    tone = rng.choice(TONES)
    max_sent = rng.choice([2, 2, 3])
    doc = (f"主题：{topic}。背景：{ctx}要求：{tone}，用至多{max_sent}句话描述该场景，"
           f"必须包含关键词“{keyword}”，不得出现阿拉伯数字。")
    return {
        "input": {"document": doc},
        "expected": {"keyword": keyword, "max_sentences": max_sent,
                     "forbid_digits": True,
                     "confidence": round(rng.uniform(0.6, 0.9), 2)},
    }


def main():
    out_dir = REPO / "datasets" / "constraint_following"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260910)
    splits = {"dev": 40, "validation": 30, "holdout": 30}
    seen = set()
    for split, n in splits.items():
        samples = []
        attempts = 0
        while len(samples) < n:
            attempts += 1
            if attempts > n * 100:
                raise RuntimeError(f"去重死锁：{split} 仅生成 {len(samples)}/{n}（样本空间不足）")
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
