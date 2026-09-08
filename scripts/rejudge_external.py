"""用当前 bench_real_external 判分器对 real_external_*.json 里带 preds 的臂离线重判。
判分器升级(v3.1→v3.2)后零 API 重算 accuracy。"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "apc-pipeline"))

from bench_real_external import answers_match, JUDGE_VERSION  # noqa: E402

for name in ("math5", "aime", "gsm8k"):
    p = REPO / "experiments" / "apcbench" / f"real_external_{name}.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    changed = 0
    for r in doc["rows"]:
        if not r.get("preds") or r.get("judge") == JUDGE_VERSION:
            continue
        preds = r["preds"]
        ok = sum(answers_match(x["pred"], x["gold"]) for x in preds)
        r["accuracy"] = round(ok / len(preds), 4)
        r["fail_sample"] = [{"i": x["i"], "pred": x["pred"][:60], "gold": x["gold"][:60]}
                             for x in preds if not answers_match(x["pred"], x["gold"])][:12]
        r["judge"] = JUDGE_VERSION
        changed += 1
    doc["judge"] = JUDGE_VERSION
    if changed:
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, "rejudged arms:", changed)
