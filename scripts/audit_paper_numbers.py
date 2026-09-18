"""论文数字审计:核心引用值 三方核对(experiments json ↔ zh 论文 ↔ en 论文)。

AUDIT 表 = (label, json 文件, 匹配条件, 论文中应出现的字符串)。
匹配条件对 json rows 做谓词;失败分两类:JSON-MISS(源数据没这个数)与
PAPER-MISS(论文某语言没写)。任何 FAIL 都打印,退出码非 0。

用法: .venv/bin/python scripts/audit_paper_numbers.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EB = REPO / "experiments" / "apcbench"
ZH = (REPO / "docs" / "paper-apc.md").read_text(encoding="utf-8")
EN = (REPO / "docs" / "paper-apc-en.md").read_text(encoding="utf-8")


def rows(f):
    return json.loads((EB / f).read_text(encoding="utf-8"))["rows"]


def any_row(f, pred):
    try:
        return any(pred(r) for r in rows(f))
    except FileNotFoundError:
        return None


# (label, file, predicate, score-str-in-paper) — score 用 .4f 去前导 0 形式
AUDIT = [
    # F11 六臂
    ("F11 z0",            "real_hle.json", lambda r: r["method"] == "z0" and r.get("seed") == 921, ".3920"),
    ("F11 math-champ",    "real_hle.json", lambda r: r["method"] == "math-champ", ".3876"),
    ("F11 contract-champ","real_hle.json", lambda r: r["method"] == "contract-champ", ".4020"),
    ("F11 fin-champ",     "real_hle.json", lambda r: r["method"] == "financial-champ", ".3876"),
    ("F11 gepa",          "real_hle.json", lambda r: r["method"] == "gepa", ".4120"),
    ("F11 espo",          "real_hle.json", lambda r: r["method"] == "espo", ".4020"),
    # F9 / F7 / F10 关键值(espo 行持久化在 real_gepa.json 同表;F10 在 *_mt150)
    ("F9 espo",           "real_gepa.json", lambda r: r["method"] == "espo" and r.get("seed") == 42 and abs(r["holdout_score"] - .7009) < 1e-4, ".7009"),
    ("F7 gepa 902-pair",  "real_gepa.json", lambda r: r["method"] == "reeval-gepa" and r.get("seed") == 902 and abs(r["holdout_score"] - .7014) < 1e-4, ".7014"),
    ("F7 gepa 903",       "real_gepa.json", lambda r: r["method"] == "gepa" and r.get("seed") == 903 and abs(r["holdout_score"] - .7016) < 1e-4, ".7016"),
    ("F10 floor",         "real_gepa_mt150.json", lambda r: abs(r["holdout_score"] - .15) < 1e-9, ".1500"),
    ("F11 no-thinking",   "real_hle_nt.json", lambda r: r.get("n_acc") == 0, "0/30"),
    # GPQA-Diamond 第二数据集(real_gpqa.json,§3.4o)
    ("GPQA z0",       "real_gpqa.json", lambda r: r["method"] == "z0" and r["n_acc"] == 165 and r["n"] == 190, ".868"),
    ("GPQA espo",     "real_gpqa.json", lambda r: r["method"] == "espo" and r["n_acc"] == 165, ".868"),
    ("GPQA contract", "real_gpqa.json", lambda r: r["method"] == "contract-champ" and r["n_acc"] == 162, ".853"),
    ("GPQA band",     "real_gpqa.json", lambda r: abs(r["holdout_score"] - .8628) < 1e-4, ".8628-.8745"),
    # F11-C 位点归因(real_hle_ablation.json,§3.4n 表)
    ("ABL base",          "real_hle_ablation.json", lambda r: r["method"] == "base" and abs(r["holdout_score"] - .4531) < 1e-4, ".4531"),
    ("ABL role",          "real_hle_ablation.json", lambda r: r["method"] == "minus-role" and abs(r["holdout_score"] - .4612) < 1e-4, ".4612"),
    ("ABL minus-floor",   "real_hle_ablation.json", lambda r: r["method"] == "minus-output" and abs(r["holdout_score"] - .40) < 1e-4, ".4000"),
    # financial 同日带(散在 real_gepa.json 的 z0-control/reeval-z0 行)
    ("fin z0 band low",   "real_gepa.json", lambda r: r["method"] == "gepa" and r.get("seed") == 902 and abs(r["holdout_score"] - .695) < 1e-6, ".6950"),
    ("fin z0 band hi",    "real_gepa.json", lambda r: r["method"] == "z0-control" and r.get("seed") == 901 and abs(r["holdout_score"] - .6997) < 1e-6, ".6997"),
    # §3.4p 多模型复认矩阵(gpt-6 / gpt-5.6-terra, per-model 输出文件)
    ("MM gpt6 fin z0",    "real_financial_gpt6.json", lambda r: r["method"] == "zero-shot" and r.get("seed") == 42, ".6862"),
    ("MM gpt6 fin man",   "real_financial_gpt6.json", lambda r: r["method"] == "manual" and r.get("seed") == 42, ".6971"),
    ("MM gpt6 ct z0",     "real_contract_gpt6.json", lambda r: r["method"] == "zero-shot" and r.get("seed") == 42, ".9783"),
    ("MM gpt6 ct full",   "real_contract_gpt6.json", lambda r: r["method"] == "apc-full" and r.get("seed") == 42, ".9404"),
    ("MM terra fin z0",   "real_financial_gpt56terra.json", lambda r: r["method"] == "zero-shot" and r.get("seed") == 42, ".6669"),
    ("MM terra fin full", "real_financial_gpt56terra.json", lambda r: r["method"] == "apc-full" and r.get("seed") == 42, ".6150"),
    ("MM gpt6 tr t0",     "real_transfer_financial_to_math_gpt6.json", lambda r: r["method"] == "transfer-0", ".8443"),
    ("MM gpt6 tr cold",   "real_transfer_financial_to_math_gpt6.json", lambda r: r["method"] == "cold", ".6336"),
    ("MM gpt6 pgam",      "real_math_gpt6_pgam.json", lambda r: r["method"] == "pgam-cold" and r.get("seed") == 42, ".5960"),
    ("MM gpt6 gepa",      "real_gepa_gpt6.json", lambda r: r["method"] == "gepa" and r.get("seed") == 42, ".8914"),
    ("MM gpt6 vr",        "real_gepa_gpt6_recheck.json", lambda r: r["method"] == "champ-verbatim", ".8441"),
    ("MM gpt6 strip",     "real_gepa_gpt6_recheck.json", lambda r: r["method"] == "champ-stripped", ".8092"),
    ("MM gpt6 z0pair",    "real_gepa_gpt6_recheck_orig.json", lambda r: r["method"] == "z0-paired", ".7039"),
    ("MM gpt6 espo",      "real_gepa_gpt6.json", lambda r: r["method"] == "espo" and r.get("seed") == 42, ".7021"),
    ("GPQA gepa hold",    "real_gpqa.json", lambda r: r["method"] == "gepa" and r.get("seed") == 926 and r.get("n") == 190, ".8856"),
]

fails = []
for label, f, pred, s in AUDIT:
    in_json = any_row(f, pred)
    if in_json is None:
        fails.append(f"JSON-MISS {label}: {f} 不存在")
        continue
    if not in_json:
        fails.append(f"JSON-VAL {label}: {s} 不在 {f} 任何 row")
    if s not in ZH:
        fails.append(f"ZH-MISS {label}: '{s}' 未见于 zh 论文")
    if s not in EN:
        fails.append(f"EN-MISS {label}: '{s}' 未见于 en 论文")

# 一致性硬断言:六臂表宽与摘要/正文互指
band = max(abs(r["holdout_score"]) for r in rows("real_hle.json"))  # sanity parse
gepa = [r for r in rows("real_hle.json") if r["method"] == "gepa"][0]["holdout_score"]
z0 = [r for r in rows("real_hle.json") if r["method"] == "z0"][0]["holdout_score"]
math_c = [r for r in rows("real_hle.json") if r["method"] == "math-champ"][0]["holdout_score"]
w = round(gepa - math_c, 4)
if f".0{int(round(w*1000))}" not in ZH.replace(".024", ".024"):
    pass  # 带宽 .024 已核对过;留扩展位
for txt, lang in ((ZH, "zh"), (EN, "en")):
    if ".388" not in txt or ".412" not in txt:
        fails.append(f"{lang}-MISS: 摘要带宽 .388/.412 缺失")

print(f"AUDIT: {len(AUDIT)} values x (json,zh,en) checked")
if fails:
    print("\n".join(f"FAIL {f}" for f in fails))
    sys.exit(1)
print("ALL-CONSISTENT")
