"""Scenario B -- multi-modal alignment groups (P3 plan, Week 3).

15 Flickr30k images (5 human reference captions each) x 6 caption styles:

    factual, verbose, concise, technical, poetic, child_like

Val     = text-only judge: the candidate caption plus the *human reference
          captions* are shown, but the image is not.  This is the kind of
          cheap proxy a val set would contain.
Holdout = vision judge: the image plus the candidate caption are shown, but the
          human references are not.  This is the thing we actually care about
          (does the caption describe *this* image?).

Both judges return 0-10 and are normalised to 0-1 for the gate; the raw numbers
stay in ``judge_raw``.

Usage
-----
    python -m jev_rsi.corpus_extended.build_scenario_b --stage arms
    python -m jev_rsi.corpus_extended.build_scenario_b --stage scores
    python -m jev_rsi.corpus_extended.build_scenario_b            # both
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from ..llm import (PRIMARY_MODEL, LLMClient, extract_score, image_part,
                   text_part, write_json)
from ..metrics import spearman

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "scenario_b"

NO_THINK = {"enable_thinking": False}
GEN_TEMPERATURE = 0.8
JUDGE_TEMPERATURE = 0.3

STYLES = ["factual", "verbose", "concise", "technical", "poetic", "child_like"]
STYLE_PROMPT = {
    "factual": "Describe this image factually and objectively in one sentence.",
    "verbose": ("Describe this image in rich detail - subjects, setting, colours, "
                "lighting and mood. About 60 words."),
    "concise": "Describe this image in as few words as possible (3-6 words).",
    "technical": ("Describe this image in a technical cataloguing style: subjects, "
                  "setting, lighting, composition. No emotion words."),
    "poetic": ("Describe this image poetically, using metaphor and lyricism. "
               "About 40 words."),
    "child_like": "Describe this image the way an excited five-year-old would.",
}


def gid_of(image_id: str) -> str:
    """Harness group id (the plan's Scenario B groups are keyed ``caption_*``)."""
    return f"caption_{image_id}"


def load_sources() -> dict:
    data = json.loads((SCENARIO / "source_images.json").read_text(encoding="utf-8"))
    return {m["image_id"]: m for m in data["images"]}


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def generate_arms(model: str = PRIMARY_MODEL, workers: int = 4) -> dict:
    src = load_sources()
    client = LLMClient()
    jobs = [(iid, style) for iid in sorted(src) for style in STYLES]

    def run(job):
        iid, style = job
        img = SCENARIO / src[iid]["path"]
        content = [image_part(img), text_part(STYLE_PROMPT[style])]
        rec = client.chat([{"role": "user", "content": content}], model=model,
                          temperature=GEN_TEMPERATURE, max_tokens=300,
                          tag=f"scenario_b/gen/{style}", extra=NO_THINK)
        return iid, style, rec["text"].strip(), rec["usage"], rec["model_returned"]

    groups: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (iid, style, caption, usage, backend) in enumerate(ex.map(run, jobs), 1):
            groups.setdefault(gid_of(iid), []).append({
                "id": style, "name": style, "style": style, "caption": caption,
                "prompt": STYLE_PROMPT[style], "temperature": GEN_TEMPERATURE,
                "usage": usage, "model_returned": backend,
            })
            print(f"  [{i:>3}/{len(jobs)}] {iid} {style:<11} {len(caption.split()):>3}w  "
                  f"{caption[:60]!r}")

    for iid in groups:
        groups[iid].sort(key=lambda a: STYLES.index(a["id"]))
    write_json(SCENARIO / "arms.json", {
        "provenance": {"generator_model": model, "temperature": GEN_TEMPERATURE,
                       "enable_thinking": False, "styles": STYLES,
                       "n_generations": len(jobs), "n_llm_calls": client.n_llm_calls,
                       "source": "nlphuji/flickr30k TEST (15 images, 75 human captions)"},
        "groups": groups,
    })
    print(f"wrote {SCENARIO/'arms.json'} ({len(jobs)} captions)")
    return groups


def load_arms() -> dict:
    data = json.loads((SCENARIO / "arms.json").read_text(encoding="utf-8"))
    return data.get("groups", data)


# ---------------------------------------------------------------------------
# judges
# ---------------------------------------------------------------------------
VAL_PROMPT = """A human wrote these reference descriptions of an image:

{references}

A system produced this description:

"{caption}"

Rate the system description from 0 to 10 on how well it captures the content
covered by the human references. 0 = unrelated or wrong, 5 = partially aligned,
10 = fully captures the same content. Reply with JSON only: {{"score": <number>}}"""

HOLDOUT_PROMPT = """Look at the image and read this description of it:

"{caption}"

Rate how well the description matches the image from 0 to 10. 0 = the
description contradicts or is unrelated to the image, 5 = partly right with
errors or invented detail, 10 = accurate and well supported by the image.
Reply with JSON only: {{"score": <number>}}"""


def _score(obj) -> float:
    v = extract_score(obj)
    if v is None:
        raise ValueError(f"bad judge payload: {obj!r}")
    return max(0.0, min(10.0, v))


def score_arms(model: str = PRIMARY_MODEL, workers: int = 4) -> dict:
    src = load_sources()
    arms = load_arms()
    client = LLMClient()

    def run(item):
        gid, iid, arm = item
        refs = "\n".join(f"- {r}" for r in src[iid]["references"])
        val_prompt = VAL_PROMPT.format(references=refs, caption=arm["caption"])
        obj_v, _ = client.chat_json(val_prompt, model=model,
                                    temperature=JUDGE_TEMPERATURE, max_tokens=200,
                                    seed=2000, tag="scenario_b/judge/val")
        img = SCENARIO / src[iid]["path"]
        content = [image_part(img),
                   text_part(HOLDOUT_PROMPT.format(caption=arm["caption"]))]
        obj_h, _ = client.chat_json_content(content, model=model,
                                            temperature=JUDGE_TEMPERATURE,
                                            max_tokens=200, seed=2001,
                                            tag="scenario_b/judge/holdout")
        return gid, iid, arm["id"], _score(obj_v), _score(obj_h)

    jobs = [(gid, gid[len("caption_"):], arm)
            for gid, g in arms.items() for arm in g]
    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (gid, iid, sid, val, hold) in enumerate(ex.map(run, jobs), 1):
            rec = out.setdefault(gid, {
                "scenario": "B", "task": "image_captioning",
                "image_path": src[iid]["path"],
                "image_source": {"dataset": "nlphuji/flickr30k", "split": "test",
                                 "filename": src[iid]["filename"]},
                "references": src[iid]["references"],
                "arms": [],
            })
            rec["arms"].append({
                "id": sid, "name": sid, "style": sid,
                "caption": next(a["caption"] for a in arms[gid] if a["id"] == sid),
                "val_score": round(val / 10.0, 6),
                "holdout_score": round(hold / 10.0, 6),
                "judge_raw": {"val_10": val, "holdout_10": hold},
            })
            print(f"  [{i:>3}/{len(jobs)}] {iid} {sid:<11} "
                  f"val={val:4.1f} holdout={hold:4.1f}")

    for iid, g in out.items():
        g["arms"].sort(key=lambda a: STYLES.index(a["id"]))
    payload = {
        "provenance": {
            "judge_model": model, "judge_temperature": JUDGE_TEMPERATURE,
            "val_judge": "text-only: caption + human references, no image",
            "holdout_judge": "vision: image + caption, no human references",
            "scale_note": ("val_score/holdout_score normalised to 0-1 for the gate; "
                           "judge_raw keeps the 0-10 numbers"),
            "n_llm_calls": client.n_llm_calls,
        },
        "groups": out,
    }
    write_json(SCENARIO / "scores.json", payload)
    report(payload)
    return payload


def report(payload: dict) -> dict:
    per: Dict[str, List[tuple]] = {}
    for g in payload["groups"].values():
        for a in g["arms"]:
            per.setdefault(a["id"], []).append((a["val_score"], a["holdout_score"]))
    print("\n=== Scenario B: per-style mean (val text-judge | holdout vision-judge) ===")
    for s in STYLES:
        rows = per[s]
        mv = statistics.fmean(x for x, _ in rows)
        mh = statistics.fmean(y for _, y in rows)
        print(f"  {s:11s} val={mv:.3f}  holdout={mh:.3f}")
    return {}


# ---------------------------------------------------------------------------
# secondary holdout: within-image ranking
# ---------------------------------------------------------------------------
RANK_PROMPT = """Here is an image, followed by its {n} candidate descriptions.

{captions}

Rank the descriptions from best to worst on how accurately each one describes
THIS image (ignore style preferences; judge factual support only). Use every id
exactly once. Reply with JSON only:
{{"ranking": [<best id>, ..., <worst id>]}}"""


def _parse_ranking(obj, ids: List[str]) -> List[str]:
    seq = None
    if isinstance(obj, dict):
        for k in ("ranking", "order", "ids", "ranking_ids"):
            if k in obj and isinstance(obj[k], list):
                seq = obj[k]
                break
        if seq is None:
            for v in obj.values():
                if isinstance(v, list):
                    seq = v
                    break
    elif isinstance(obj, list):
        seq = obj
    if seq is None:
        raise ValueError(f"bad ranking payload: {obj!r}")
    flat: List[str] = []
    for item in seq:
        if isinstance(item, dict):
            item = item.get("id") or next(iter(item.values()), "")
        token = str(item).strip().strip('"').lower()
        for i in ids:
            if token == i.lower() or token.startswith(i.lower()):
                if i not in flat:
                    flat.append(i)
                break
    # append anything the judge dropped, preserving declaration order
    flat += [i for i in ids if i not in flat]
    return flat


def rank_arms(model: str = PRIMARY_MODEL, workers: int = 4) -> dict:
    """Within-image ranking holdout.

    The 0-10 vision judge saturates (most captions get 9-10), so it cannot
    discriminate styles.  Asking the same model to *rank* the six captions of one
    image keeps the comparison inside a single image and removes the ceiling.
    """
    src = load_sources()
    payload = json.loads((SCENARIO / "scores.json").read_text(encoding="utf-8"))
    client = LLMClient()
    ids = list(STYLES)

    def run(gid: str):
        iid = gid[len("caption_"):]
        arms = payload["groups"][gid]["arms"]
        listing = "\n".join(f'{a["id"]}: "{a["caption"]}"' for a in arms)
        content = [image_part(SCENARIO / src[iid]["path"]),
                   text_part(RANK_PROMPT.format(n=len(arms), captions=listing))]
        obj, _rec = client.chat_json_content(content, model=model,
                                             temperature=0.0, max_tokens=600,
                                             tag="scenario_b/rank")
        return gid, _parse_ranking(obj, ids)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gid, order in ex.map(run, sorted(payload["groups"])):
            arms = payload["groups"][gid]["arms"]
            n = len(arms)
            for a in arms:
                pos = order.index(a["id"])
                a["holdout_rank_score"] = round((n - 1 - pos) / (n - 1), 6)
                a["holdout_rank_position"] = pos + 1
            vs = [a["val_score"] for a in arms]
            rs = [a["holdout_rank_score"] for a in arms]
            payload["groups"][gid]["val_vs_rank_spearman"] = round(spearman(vs, rs), 4)
            print(f"  {gid}: ranking={order}  spearman(val,rank)="
                  f"{payload['groups'][gid]['val_vs_rank_spearman']:+.3f}")

    payload.setdefault("provenance", {})["ranking_holdout"] = (
        "same model, image + all six captions, asked for a strict order; "
        "holdout_rank_score = (n-1-pos)/(n-1)")
    payload["provenance"]["n_llm_calls_rank"] = client.n_llm_calls
    write_json(SCENARIO / "scores.json", payload)

    per: Dict[str, List[float]] = {}
    for g in payload["groups"].values():
        for a in g["arms"]:
            per.setdefault(a["id"], []).append(a["holdout_rank_score"])
    print("\n=== Scenario B: mean ranking-holdout per style (1.0 = always best) ===")
    for s in STYLES:
        print(f"  {s:11s} {statistics.fmean(per[s]):.3f}")
    sp = [g["val_vs_rank_spearman"] for g in payload["groups"].values()]
    print(f"mean Spearman(val, vision rank) = {statistics.fmean(sp):+.3f}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["arms", "scores", "rank", "all"], default="all")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.stage in ("arms", "all"):
        generate_arms(args.model, args.workers)
    if args.stage in ("scores", "all"):
        score_arms(args.model, args.workers)
    if args.stage in ("rank", "all"):
        rank_arms(args.model, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
