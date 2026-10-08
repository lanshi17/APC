"""Fetch the *real* source material for the P3 extended corpus.

Nothing is synthesised here: articles come from the CNN/DailyMail validation
split (via the HuggingFace datasets-server JSON API, so no multi-GB download)
and images come from Flickr30k (config ``TEST``, 5 human captions each).

Usage
-----
    python -m jev_rsi.corpus_extended.fetch_sources cnn_dailymail --n 10
    python -m jev_rsi.corpus_extended.fetch_sources flickr30k --n 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parent
DS_SERVER = "https://datasets-server.huggingface.co"
SCENARIO_C = ROOT / "scenario_c"
SCENARIO_B = ROOT / "scenario_b"


def _fetch_json(url: str, retries: int = 4) -> Dict[str, Any]:
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            r = httpx.get(url, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def _rows(dataset: str, config: str, split: str, offset: int, length: int) -> List[dict]:
    d = _fetch_json(f"{DS_SERVER}/rows?dataset={dataset}&config={config}"
                    f"&split={split}&offset={offset}&length={length}")
    return [row["row"] for row in d.get("rows", [])]


# ---------------------------------------------------------------------------
# CNN / DailyMail
# ---------------------------------------------------------------------------
def fetch_cnn_dailymail(n: int = 10, min_words: int = 300, max_words: int = 520,
                        pool: int = 400, out_dir: Path = SCENARIO_C) -> List[dict]:
    """Pick ``n`` validation articles inside the plan's 300-500 word band."""
    cands: List[dict] = []
    got = 0
    while got < pool:
        batch = _rows("abisee/cnn_dailymail", "3.0.0", "validation", got, 100)
        if not batch:
            break
        for r in batch:
            w = len(r["article"].split())
            if min_words <= w <= max_words:
                cands.append({"id": r["id"], "article": r["article"],
                              "highlights": r["highlights"], "n_words": w})
        got += 100
        if len(cands) >= pool:
            break

    # Deterministic spread across the (id-sorted) candidate list.
    cands.sort(key=lambda r: r["id"])
    if len(cands) < n:
        raise RuntimeError(f"only {len(cands)} candidates in the word band")
    step = len(cands) / n
    picked = [cands[min(len(cands) - 1, int(i * step))] for i in range(n)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "articles").mkdir(exist_ok=True)
    manifest = []
    for i, rec in enumerate(picked, 1):
        aid = f"article_{i:02d}"
        (out_dir / "articles" / f"{aid}.txt").write_text(rec["article"] + "\n", encoding="utf-8")
        manifest.append({"article_id": aid, "source_id": rec["id"], "n_words": rec["n_words"],
                         "article_path": f"articles/{aid}.txt",
                         "article": rec["article"], "reference": rec["highlights"]})
    (out_dir / "source_articles.json").write_text(
        json.dumps({"dataset": "abisee/cnn_dailymail", "config": "3.0.0",
                    "split": "validation", "word_band": [min_words, max_words],
                    "n_candidates": len(cands), "articles": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir/'source_articles.json'}  ({n} articles, "
          f"{len(cands)} candidates in [{min_words},{max_words}] words)")
    return manifest


# ---------------------------------------------------------------------------
# Flickr30k
# ---------------------------------------------------------------------------
def fetch_flickr30k(n: int = 15, out_dir: Path = SCENARIO_B,
                    pool: int = 120, per_page: int = 100) -> List[dict]:
    """Download ``n`` images + their 5 human captions.

    The datasets-server hands out signed, expiring image URLs, so the images are
    copied into the repo and referenced locally from then on.
    """
    rows: List[dict] = []
    for off in range(0, pool, per_page):
        rows.extend(_rows("nlphuji/flickr30k", "TEST", "test", off, per_page))
        if len(rows) >= pool:
            break
    rows.sort(key=lambda r: str(r.get("filename")))
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} flickr30k rows fetched")
    step = len(rows) / n
    picked = [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, rec in enumerate(picked, 1):
        iid = f"img_{i:03d}"
        src = rec["image"]["src"]
        dest = img_dir / f"{iid}.jpg"
        if not dest.exists():
            for attempt in range(4):
                try:
                    r = httpx.get(src, timeout=120, follow_redirects=True)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"could not download {src}")
        manifest.append({
            "image_id": iid, "filename": rec.get("filename"),
            "path": f"images/{iid}.jpg",
            "width": rec["image"].get("width"), "height": rec["image"].get("height"),
            "references": list(rec["caption"]),
        })
    (out_dir / "source_images.json").write_text(
        json.dumps({"dataset": "nlphuji/flickr30k", "config": "TEST", "split": "test",
                    "n_pool": len(rows), "images": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir/'source_images.json'}  ({n} images, "
          f"{sum(len(m['references']) for m in manifest)} human captions)")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cnn_dailymail")
    c.add_argument("--n", type=int, default=10)
    c.add_argument("--min-words", type=int, default=300)
    c.add_argument("--max-words", type=int, default=520)
    f = sub.add_parser("flickr30k")
    f.add_argument("--n", type=int, default=15)
    args = ap.parse_args(argv)
    if args.cmd == "cnn_dailymail":
        fetch_cnn_dailymail(args.n, args.min_words, args.max_words)
    else:
        fetch_flickr30k(args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
