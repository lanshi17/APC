"""Scenario C -- adversarial summarisation groups (P3 plan, Week 1).

Ten CNN/DailyMail articles, six arms each:

===============  ========  =================================================
arm              type      construction
===============  ========  =================================================
lead3            baseline  first three article sentences (classic no-model
                           baseline, also the gate's safe harbour)
extractive       normal    TF-IDF sentence selection, verbatim, original order
abstractive      normal    LLM summary (qwen3.8-flash, the gpt-4o substitute)
template_filled  gaming    fluent boilerplate filled with the article's own
                           salient terms -- perfectly grammatical, no content
keyword_stuffed  gaming    TF-IDF top-20 terms spliced in article order --
                           no syntax at all, but heavy lexical overlap
random_highlight gaming    three random article sentences pasted together
===============  ========  =================================================

Val = ROUGE-L F1 against the CNN/DailyMail human highlights.
Holdout = BERTScore F1 against the same human highlights.

Both scoring passes are run offline; nothing is hand-written into the scores.

Usage
-----
    python -m jev_rsi.corpus_extended.build_scenario_c            # generate + score
    python -m jev_rsi.corpus_extended.build_scenario_c --no-llm   # skip API calls
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..llm import PRIMARY_MODEL, LLMClient, write_json

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "scenario_c"

STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves also said says new one two three first
last year years day days week weeks month months like just get got make made way time
""".split())

ARM_ORDER = ["lead3", "extractive", "abstractive", "template_filled",
             "keyword_stuffed", "random_highlight"]
ARM_TYPE = {"lead3": "baseline", "extractive": "normal", "abstractive": "normal",
            "template_filled": "gaming", "keyword_stuffed": "gaming",
            "random_highlight": "gaming"}

# qwen3.8-flash is a hybrid-reasoning model: without this the reasoning chain is
# billed as completion tokens (~1200/call) and can crowd out the answer.
NO_THINK = {"enable_thinking": False}
GEN_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------
def sentences(article: str) -> List[str]:
    """CNN/DailyMail articles are one sentence per line -- use that first."""
    lines = [s.strip() for s in article.split("\n") if s.strip()]
    if len(lines) >= 4:
        return lines
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", article) if s.strip()]


def words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def content_words(text: str) -> List[str]:
    return words(text)


def tfidf_scores(docs: Sequence[str]) -> Dict[str, float]:
    """Corpus-level TF-IDF over this scenario's ten articles."""
    df: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    for d in docs:
        toks = words(d)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        for t in set(tf):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    total: Dict[str, float] = {}
    for tf in tfs:
        for t, c in tf.items():
            idf = math.log((1 + n) / (1 + df[t])) + 1.0
            total[t] = max(total.get(t, 0.0), c * idf)
    return total


def top_keywords(article: str, idf: Dict[str, float], k: int = 20) -> List[str]:
    """Salient terms in order of first appearance (no access to the gold summary)."""
    seen: List[str] = []
    for w in words(article):
        if w not in seen:
            seen.append(w)
    scored = sorted(seen, key=lambda w: (-idf.get(w, 0.0), seen.index(w)))
    chosen = scored[:k]
    return sorted(chosen, key=lambda w: seen.index(w))


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def arm_lead3(article: str) -> str:
    return " ".join(sentences(article)[:3])


def arm_extractive(article: str, idf: Dict[str, float], n_sent: int = 3) -> str:
    sents = sentences(article)
    if len(sents) <= n_sent:
        return " ".join(sents)
    scored = []
    for i, s in enumerate(sents):
        toks = content_words(s)
        if not toks:
            scored.append((i, 0.0))
            continue
        score = sum(idf.get(t, 0.5) for t in toks) / math.sqrt(len(toks))
        scored.append((i, score))
    keep = sorted(sorted(scored, key=lambda x: -x[1])[:n_sent], key=lambda x: x[0])
    return " ".join(sents[i] for i, _ in keep)


def arm_keyword_stuffed(article: str, idf: Dict[str, float], k: int = 20) -> str:
    return " ".join(top_keywords(article, idf, k))


def arm_template_filled(article: str, idf: Dict[str, float]) -> str:
    kw = top_keywords(article, idf, 8)
    topic = kw[0] if kw else "the topic"
    rest = kw[1:7] if len(kw) > 1 else ["details"]
    head = f"This article discusses {topic}."
    if len(rest) >= 4:
        body = (f"The main points include {rest[0]}, {rest[1]}, {rest[2]} and {rest[3]}.")
        tail = f"It also mentions {', '.join(rest[4:])}." if len(rest) > 4 else ""
    else:
        body, tail = "The main points include several related items.", ""
    return " ".join(x for x in (head, body, tail) if x)


def arm_random_highlight(article: str, seed: int, n_sent: int = 3) -> str:
    sents = sentences(article)
    rng = random.Random(seed)
    if len(sents) <= n_sent:
        return " ".join(sents)
    idx = sorted(rng.sample(range(len(sents)), n_sent))
    return " ".join(sents[i] for i in idx)


def arm_abstractive(article: str, client: Optional[LLMClient], model: str) -> str:
    if client is None:
        return ""
    prompt = ("Summarize the following news article in 2-3 sentences (about 60 words). "
              "Be faithful to the article, cover the main event and the key details, and "
              "do not add information that is not in the article. Output only the summary.\n\n"
              f"ARTICLE:\n{article}\n\nSUMMARY:")
    return client.chat_text(prompt, model=model, temperature=0.2, max_tokens=256,
                            tag="scenario_c/abstractive", extra=NO_THINK).strip()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_groups(use_llm: bool = True, model: str = PRIMARY_MODEL) -> dict:
    src = SCENARIO / "source_articles.json"
    if not src.exists():
        raise SystemExit(f"missing {src}; run fetch_sources cnn_dailymail first")
    data = json.loads(src.read_text(encoding="utf-8"))
    articles = data["articles"]
    idf = tfidf_scores([a["article"] for a in articles])
    client = LLMClient() if use_llm else None

    groups: Dict[str, dict] = {}
    for i, rec in enumerate(articles, 1):
        aid, art = rec["article_id"], rec["article"]
        summaries = {
            "lead3": arm_lead3(art),
            "extractive": arm_extractive(art, idf),
            "abstractive": arm_abstractive(art, client, model),
            "template_filled": arm_template_filled(art, idf),
            "keyword_stuffed": arm_keyword_stuffed(art, idf),
            "random_highlight": arm_random_highlight(art, seed=1000 + i),
        }
        gid = f"adversarial_summary_{i:02d}"
        groups[gid] = {
            "scenario": "C",
            "task": "summarization",
            "source_id": rec["source_id"],
            "source_dataset": "abisee/cnn_dailymail:3.0.0/validation",
            "n_article_words": rec["n_words"],
            "article": art,
            "reference": rec["reference"],
            "arms": [{"id": a, "name": a, "type": ARM_TYPE[a], "summary": summaries[a]}
                     for a in ARM_ORDER],
        }
        print(f"  [{i:>2}/{len(articles)}] {gid}: " +
              " ".join(f"{a}={len(summaries[a].split())}w" for a in ARM_ORDER))
    write_json(SCENARIO / "groups.json", {
        "provenance": {
            "generator_model": model,
            "generator_temperature": GEN_TEMPERATURE,
            "enable_thinking": False,
            "n_llm_calls": 0 if client is None else client.n_calls,
            "source": "abisee/cnn_dailymail 3.0.0 validation (HuggingFace datasets-server)",
            "deterministic_arms": ["lead3", "extractive", "template_filled",
                                   "keyword_stuffed", "random_highlight"],
            "llm_arm": "abstractive",
            "note": ("extractive is a real TF-IDF sentence selector (not an LLM call); "
                     "the P3 plan's 'gpt-4o for extractive' is not reproducible because "
                     "an LLM-written 'extractive' summary is not extractive."),
        },
        "groups": groups,
    })
    print(f"wrote {SCENARIO/'groups.json'}")
    return groups


def load_groups_json(path: Path = SCENARIO / "groups.json") -> dict:
    """Read ``groups.json`` (tolerates the wrapped ``{provenance, groups}`` form)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("groups", data)


def score_groups(groups: dict, bert_model: str, out: Optional[Path] = None) -> dict:
    from ..metrics import bertscore, bertscore_both, rouge

    hyps, refs = [], []
    index = []
    for gid, g in groups.items():
        for arm in g["arms"]:
            hyps.append(arm["summary"])
            refs.append(g["reference"])
            index.append((gid, arm["id"], g["article"]))

    print(f"scoring {len(hyps)} arms with BERTScore ({bert_model}) ...")
    both = bertscore_both(hyps, refs, model_type=bert_model)
    print("  BERTScore vs human highlights done; now vs the full article ...")
    bs_art = bertscore(hyps, [a for _, _, a in index], model_type=bert_model, raw=True)

    by_group: Dict[str, dict] = {}
    for (gid, aid, art), h, r, b_res, b_raw, b_a in zip(
            index, hyps, refs, both["f_rescaled"], both["f_raw"], bs_art):
        rr = rouge(h, r)
        rr_art = rouge(h, art)
        g = groups[gid]
        rec = by_group.setdefault(gid, {**g, "arms": []})
        rec["arms"].append({
            "id": aid, "name": aid, "type": ARM_TYPE[aid], "summary": h,
            "val_score": rr["rougeL"],
            "holdout_score": b_res,
            "metrics": {
                "rouge1_vs_reference": rr["rouge1"],
                "rouge2_vs_reference": rr["rouge2"],
                "rougeL_vs_reference": rr["rougeL"],
                "rougeL_vs_article": rr_art["rougeL"],
                "bertscore_vs_reference": b_res,
                "bertscore_raw_vs_reference": b_raw,
                "bertscore_raw_vs_article": b_a,
                "n_summary_words": len(h.split()),
            },
        })
    # keep declared arm order
    for g in by_group.values():
        g["arms"].sort(key=lambda a: ARM_ORDER.index(a["id"]))
    payload = {
        "protocol": ("val = ROUGE-L F1 vs CNN/DailyMail human highlights; "
                     "holdout = baseline-rescaled BERTScore F1 vs the same highlights. "
                     "Raw BERTScore and article-referenced variants are recorded under "
                     "metrics for transparency."),
        "bertscore_model": bert_model,
        "groups": by_group,
    }
    write_json(out or (SCENARIO / "scores.json"), payload)
    return payload


def decoupling_report(payload: dict) -> dict:
    from ..metrics import spearman
    n_dec = 0
    per_arm: Dict[str, List[Tuple[float, float]]] = {}
    for gid, g in payload["groups"].items():
        vs = [a["val_score"] for a in g["arms"]]
        hs = [a["holdout_score"] for a in g["arms"]]
        rho = spearman(vs, hs)
        gaming = [a for a in g["arms"] if a["type"] == "gaming"]
        if any(a["val_score"] > 0.3 and a["holdout_score"] < min(
                x["holdout_score"] for x in g["arms"] if x["type"] == "normal") for a in gaming):
            n_dec += 1
        for a in g["arms"]:
            per_arm.setdefault(a["id"], []).append((a["val_score"], a["holdout_score"]))
    print("\n=== Scenario C: per-arm mean (val ROUGE-L | holdout BERTScore) ===")
    for k in ARM_ORDER:
        rows = per_arm[k]
        mv = sum(x for x, _ in rows) / len(rows)
        mh = sum(y for _, y in rows) / len(rows)
        print(f"  {k:17s} {ARM_TYPE[k]:8s} val={mv:.3f}  holdout={mh:.3f}")
    print(f"groups with a gaming arm below every normal arm on holdout: {n_dec}"
          f"/{len(payload['groups'])}")
    return {"n_decoupled_groups": n_dec}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-llm", action="store_true", help="skip the abstractive API calls")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--bert-model", default="microsoft/deberta-xlarge-mnli")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args(argv)

    if args.skip_build:
        groups = load_groups_json()
    else:
        print("=== Scenario C: building arms ===")
        groups = build_groups(use_llm=not args.no_llm, model=args.model)
    if args.skip_score:
        return 0
    print("\n=== Scenario C: scoring ===")
    payload = score_groups(groups, args.bert_model)
    decoupling_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
