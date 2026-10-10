#!/usr/bin/env python3
"""Apply the P3 (extended corpus) additions to the negative-result manuscript.

The manuscript is authored in ``negative-result.md`` and converted to
``.tex``/``.pdf`` by ``manuscript_negative/build.py``, so this patch edits the
markdown source only.  Every insertion is anchored on an existing sentence and
is idempotent: re-running after a successful patch reports "already applied".

    python docs/paper_patch/patch_manuscript.py [--dry-run]

Numbers come from ``code/jev_rsi/results/jev_rsi_extended_{results,analysis}.json``
and were cross-checked with ``docs/paper_patch/verify_numbers.py``.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
CODE = pathlib.Path(__file__).resolve().parents[2]      # <repo>/docs/paper_patch
PAPER_REPO = pathlib.Path(os.environ.get("APC_PAPER_DIR", CODE.parent / "02_apc_paper"))
PAPER = PAPER_REPO
MD = PAPER / "manuscript_negative" / "negative-result.md"

MARK = "<!-- P3-EXTENDED-CORPUS -->"


# ---------------------------------------------------------------------------
# 1. Abstract
# ---------------------------------------------------------------------------
ABSTRACT_ANCHOR = "so no method can win on the rest."
ABSTRACT_ADD = """
We then built three extension corpora whose entire purpose was to give structure
an advantage, and it lost all three. The 45 new groups (485 arms; 77 deployment
decisions in total) come from adversarial summarisation whose gaming arms should
inflate the validation metric, a creative-writing judge whose per-arm spread
should dominate the signal, and a text-versus-image captioning split. Each
premise failed to hold on real runs — ROUGE-L is not gameable by keyword
stuffing, the reasoning judges are too consistent (0 of 100 arms reach σ > 2.0),
and the vision judge saturates at 10/10 — and the gate's deficit grew rather
than shrank: **37/77** against **47/77** (paired p = .0125). The losses have
legible causes: thresholds calibrated on one corpus are off by up to 14× in
another, and the gate retreats to a fallback arm that is not always safe.
"""

# ---------------------------------------------------------------------------
# 2. Introduction item 6
# ---------------------------------------------------------------------------
INTRO_ANCHOR = "only because we went looking for it, which suggests others should look too."
INTRO_ADD = """
6. **A designed-for-the-method extension that still loses.** We built the three
   regimes a structured gate is supposed to need — a gameable validation metric,
   a noisy judge, and a modality gap between validation and deployment — with 45
   new groups and 485 real model runs. All three premises turned out to be
   unreachable in practice, the gate fell further behind (37/77 vs 47/77,
   p = .0125), and the residual losses are attributable to *threshold
   miscalibration* and an *unsafe fallback arm*, not to the absence of the
   intended regime.
"""

# ---------------------------------------------------------------------------
# 3. Experimental setup -- extended corpus paragraph
# ---------------------------------------------------------------------------
SETUP_ANCHOR = """**Reproducibility.** `python -m jev_rsi.experiments` regenerates every artifact
byte-for-byte in 11 seconds. Determinism is a design constraint, not an
observation: all Monte Carlo is seeded, and the offline replay reads only
validation scores."""
SETUP_ADD = """

**Extended corpus.** Section 5.6 adds 45 groups (350 arms) from three scenarios
built to favour structured gating: adversarial summarisation (10 groups, 60
arms), creative writing with a sampled judge (20 groups, 200 arms), and
image captioning with a text-only validation judge (15 groups, 90 arms). Every
score in that section comes from a real model call; the prompts, completions,
judge draws and a per-call usage ledger are released under
`code/jev_rsi/corpus_extended/`. The frozen corpus above is unchanged: the
original artifact reproduces byte-for-byte (`--corpus frozen`)."""

# ---------------------------------------------------------------------------
# 4. New section 5.6
# ---------------------------------------------------------------------------
S56 = r"""
## 5.6 Extended corpus: three scenarios designed to favour structure

The result above can be read as a statement about the corpus rather than about
the gate: 43 groups from one project, a quarter of them carrying no validation
signal at all. To test that reading we built three corpora whose *stated purpose
was to give the structured gate an advantage*, and pre-registered what each
premise would have to look like. Table 8 gives the deployment outcomes, Table 9
the diagnostics that the premises were supposed to move.

**Table 8. Extended corpus: deployment outcomes by scenario** (decision groups;
`jev` is the static gate, `jev+RSI` the leave-one-out self-improvement variant).

| Corpus | n | `greedy` | `heuristic` | `jev` | `jev+RSI` | `greedy` − `jev` |
|---|---|---|---|---|---|---|
| Frozen corpus (§5.1) | 32 | **18/32** (+.01253) | 13/32 (+.01300) | 11/32 (+.01383) | 11/32 (+.01384) | W/L/T 3/12/17, p = .035 |
| Scenario C: adversarial metric | 10 | **6/10** (+.02394) | 6/10 (+.02394) | 6/10 (+.02394) | 6/10 (+.02394) | 0/0/10, p = 1.00 |
| Scenario A: noisy judge | 20 | **8/20** (+.03550) | 8/20 (+.03550) | 5/20 (+.16550) | 5/20 (+.16550) | 4/9/7, p = .267 |
| Scenario B: modality gap | 15 | **15/15** (+.00000) | 15/15 (+.00000) | 15/15 (+.00000) | 15/15 (+.00000) | 0/0/15, p = 1.00 |
| **Total** | **77** | **47/77** (+.01754) | 42/77 (+.01773) | 37/77 (+.05184) | — | **7/21/49, p = .0125** |

Parentheses give average holdout regret. Across the 77 decision groups the
baseline's advantage more than doubles relative to the frozen corpus (mean
paired difference −.0343, 95% CI [−.0658, −.0082] against −.0013, [−.0029,
−.0001]), and it is no longer confined to a thin margin: the gate is worse on 21
groups and better on 7.

**Table 9. What each scenario was supposed to break, and what it did.**

| Scenario | Pre-registered premise | Measured | Verdict |
|---|---|---|---|
| C | gaming arms reach val > 0.4 with holdout < 0.3 | max val 0.380; gaming arms average val .139 / holdout −.026 | premise unreachable |
| C | validation argmax is captured by a gaming arm | 0/10 groups (also 0/10 under every other recorded val metric) | not captured |
| A | judge σ > 2.0 (0–10) in ≥ 5 groups | mean per-arm σ = .47; 0/20 groups; 0/100 probed arms | premise unreachable |
| B | text-only judge disagrees with vision judge | Spearman(val, vision) = **+.346** (ranking holdout **+.527**) | agrees |
| B | vision judge discriminates styles | 5 of 6 styles score ≥ .86; factual 1.000 | ceiling |

### 5.6.1 Scenario C: the validation metric is not gameable

Ten CNN/DailyMail articles (300–520 words, human highlights) each get six
summaries: lead-3, a deterministic TF-IDF extractive selector, a
model-written abstractive summary, and three cheap gaming arms (slot template,
keyword stuffing, random highlight). Validation is ROUGE-L F1 against the human
highlights — the metric the paper's deployment pipeline actually uses — and
holdout is baseline-rescaled BERTScore F1 against the same highlights.

The gaming arms do collapse on holdout: −0.135 (keyword-stuffed), −0.029
(template), +0.086 (random) against +0.276 for the honest abstractive summary.
But they collapse on validation too (.171, .126, .121 against .252). ROUGE-L F1
is bounded by precision as well as recall, so padding a summary with article
keywords lowers it; the pre-registered capture band (`val > 0.4`) is unreachable
because *no* arm reaches 0.38. Validation argmax is captured by a gaming arm in
**0 of 10** groups, and remains uncaptured under all six validation metrics we
recorded (including ROUGE-L against the full article, where it captures 1 of
10). Mean Spearman between validation and holdout is **+.50**.

This is the paper's central point restated in the adversarial direction: it is
not that structure fails to detect gaming, it is that a *faithful* validation
metric leaves nothing to detect. The gate tied the baseline on all ten groups
(0 wins, 0 losses, 10 ties).

### 5.6.2 Scenario A: the judge refuses to be noisy

Twenty creative-writing tasks (story continuation, sonnet, dialogue, scene
description, fable), each attempted by ten prompt strategies, 200 arms in total.
Validation is a **single** judge draw; holdout is the **mean of five further
independent draws** of the same judge prompt at temperature 1.0 — the regime in
which a single-sample validation score should be a bad guide.

The premise that a modern judge is noisy enough for this to matter does not
survive contact. Mean per-arm standard deviation is **0.47** on the 0–10 scale
and no group reaches 2.0. Because one model could be a lucky choice, we probed
every available judge on 5 groups × 10 arms × 6 draws: `qwen3.8-flash`
mean σ = .556, `qwen3.8-27b` mean σ = .675, and **0 of 100 arms** exceed σ = 2.0
(maximum 1.37). Reasoning judges at temperature 1.0 are simply not that
variable.

The gate nonetheless loses here more than anywhere else: 5/20 against 8/20, with
average regret rising from +.0355 to +.1655. The reason is visible in the rule
firings. It retreats from validation argmax on 15 of 20 groups, almost always
through the alignment veto, and the arm it retreats to is `zero_shot` — the
designated safe arm of this family. In creative writing `zero_shot` is not safe:
several groups have strategies that score 1–3 points (0-10) above it, and one
retreat costs 0.24 holdout (the largest single loss in either corpus). The
gate's failure mode is thus not that it detects a phantom collapse, but that it
trusts a fallback whose safety was never established on this task.

### 5.6.3 Scenario B: the two judges agree

Fifteen Flickr30k images, each captioned in six styles (factual, verbose,
concise, technical, poetic, child-like), 90 arms. Validation is a text-only
judge that sees the caption and the five human reference captions but *not* the
image; holdout is a vision judge that sees the image and the caption but no
references. This is the cross-modal analogue of a validation/deployment gap.

Both metrics rank the styles the same way. The vision judge's scores are
compressed near the ceiling (factual 1.000, concise .993, technical .990,
verbose .953, child-like .920, poetic .860), so we added a second holdout that
cannot saturate: the same model sees the image plus all six captions and returns
a strict order. On that ranking, technical (.893) and factual (.880) still lead,
and mean Spearman with validation is **+.527** (negative in 1 of 15 groups).
Every policy is exactly right on all 15 groups, and the gate never fired.

### 5.6.4 Why the gate loses when it does: thresholds and fallbacks

The gate's three thresholds are constants fitted to the frozen corpus. The
extended scenarios measure validation on different scales, and the constants do
not transfer. Table 10 reports the median top-1/top-2 validation margin in each
corpus as a multiple of `NOISE_BAND` (0.007), the band the thresholds were
calibrated against.

**Table 10. Validation scale drift and its effect.**

| Corpus | median val margin | × `NOISE_BAND` | alignment rule saturated | effect of refitting the scale |
|---|---|---|---|---|
| Frozen | 0.0006 | 0.1× | 28% of groups | — |
| Scenario C | 0.0257 | 3.7× | 60% of groups | static gate worsens (6/10 → 4/10); LOO-RSI returns to parity (6/10) |
| Scenario A | 0.0000 | 0.0× | 35% of groups | fallback arm dominates the loss, not the scale |
| Scenario B | 0.1000 | 14.3× | 67% of groups | unchanged (all policies exact) |

When we re-estimate the noise band per scenario (median absolute gap between
adjacent arms) and refit only through the gate's own self-improvement loop,
Scenario C recovers to parity with validation argmax (6/10, +.0239) — the
deficit there is a calibration artifact, not evidence that the primitives cannot
work. Scenario A does not recover, because no threshold choice makes an unsafe
fallback safe. That distinction is the practically useful one: a structured gate
ships as a *policy plus constants*, and the constants are as much a part of the
deployed object as the rules are.

**Boundary conditions, extended.** Across both corpora the gate beats validation
argmax on 7 groups and loses on 21 (ties 49). Every win has the same shape: the
arm the gate retreats to happens to be the arm that was actually best
(§6.1's alignment diagnostic, by another route). Every large loss has one of the
two shapes above — a fallback that is not safe, or a threshold whose units came
from another task. We did not find a regime in which structure wins for a reason
that a one-line baseline could not have found first.
"""

# ---------------------------------------------------------------------------
# 5. Discussion
# ---------------------------------------------------------------------------
DISC_ANCHOR = """in the regime our corpus almost entirely lacks: groups where validation is
informative but *insufficient*, i.e. where the ranking is neither trustworthy
nor silent."""
DISC_ADD = """

**Extended-corpus boundary conditions.** We tried to manufacture that regime
three times, in the three ways the literature suggests it arises — a gameable
validation metric, an under-powered judge, and a validation/deployment modality
gap — and failed to manufacture it at all. The metrics were faithful
(ROUGE-L punishes keyword stuffing rather than rewarding it), the judges were
consistent (σ ≈ 0.5 on a 0–10 scale, with none of 100 probed arms above 2.0),
and the modalities agreed (validation and vision rankings correlate at +.53).
The gate accordingly lost on 21 groups and won on 7, and the loss was an order
of magnitude larger than on the original corpus.

The two mechanisms behind the losses are more useful than the headline. First,
**thresholds are task-local**: the gate's constants encode a noise band fitted
on a benchmark-fraction scale, and the same constants are off by 0.1× to 14.3×
in the extensions; refitting them locally through the gate's own learner
recovers parity in Scenario C. Second, **a fallback arm is only safe relative to
a task**: Scenario A's largest deficits come from retreating to `zero_shot`,
which the frozen corpus's Occam order assumes is the conservative choice, but
which is 1–3 judge points below the best strategy on creative writing. Any
deployment gate that carries a "safe arm" should have to demonstrate that
safety on the target task, and our results give a concrete estimate of what it
costs when it does not (regret +.166 versus +.036)."""
DISC_WHATIF_ANCHOR = """**What would change our conclusion.** (i) A corpus with groups where four or
more arms separate beyond the noise band *and* the baseline is wrong — our
corpus never separates more than three, so this regime is untested rather than
tested and rejected."""
DISC_WHATIF_ADD = """

The extended corpus supplies the separation we lacked — Scenario A and B groups
separate six to ten arms — but the baseline is not wrong there either (§5.6), so
criterion (i) remains open with one half satisfied. What we now know is that the
other half is hard to arrange: of the three constructions designed to make
validation argmax wrong, none did."""

# ---------------------------------------------------------------------------
# 6. Conclusion
# ---------------------------------------------------------------------------
CONC_ANCHOR = """We release the frozen corpus, all artifacts, and an 11-second deterministic
reproduction, in the hope that the next structured-deployment paper starts by
beating this baseline rather than assuming it is too weak to matter."""
CONC_ADD = """

We then gave the hypothesis three more chances, in 45 purpose-built groups and
485 real model runs, and it did worse rather than better: 37/77 against 47/77,
with the whole of the additional loss explained by thresholds that do not
transfer between tasks and by a fallback arm that was never safe on the new
ones. The scenarios that are supposed to favour structure — a gameable metric,
a noisy judge, a modality gap — did not exist in any of the three regimes we
constructed, which is itself the finding we would most want a practitioner to
take away: check that the pathology you are defending against is present before
paying for the defence."""

# ---------------------------------------------------------------------------
# 7. Reproducibility statement
# ---------------------------------------------------------------------------
# A new paragraph appended to the *existing* statement.  The first attempt
# anchored on the heading plus the statement's opening sentence, which split that
# sentence in half and inserted a second "# Reproducibility statement" heading
# mid-paragraph; the anchor is now the end of the statement instead.
REPRO_ANCHOR = "programmatically against those artifacts."
REPRO_ADD = """The extended-corpus results of §5.6 are regenerated by
`cd code && python -m jev_rsi.corpus_extended.build_scenario_{a,b,c}` (all model
calls are response-cached on disk, so a re-run is free and byte-identical) and
analysed by `python -m jev_rsi.analysis_extended`; `python scripts/p3_acceptance.py`
re-checks the pre-registered acceptance criteria. The corpus, the model
completions, the raw judge draws and a per-call token ledger live under
`code/jev_rsi/corpus_extended/`. Two protocol deviations are material and are
recorded with the artifacts: `gpt-4o`/`gpt-4o-vision` were unavailable, so
`qwen3.8-flash` (with `qwen3.8-27b` as a variance contrast) generated and judged
every text, and BERTScore is reported baseline-rescaled against the dataset's
human summaries rather than the raw article."""

REPRO_PARA = REPRO_ADD
# what the broken first attempt injected, verbatim, so it can be removed
REPRO_BROKEN = ("\n# Reproducibility statement\n\n" + REPRO_PARA
                + "\n\nAll results are regenerated by "
                  "`cd code && python -m jev_rsi.experiments`\n\n")


PATCHES = [
    ("abstract", ABSTRACT_ANCHOR, ABSTRACT_ADD),
    ("intro item 6", INTRO_ANCHOR, INTRO_ADD),
    ("setup: extended corpus", SETUP_ANCHOR, SETUP_ADD),
    ("results 5.6", "\n# 6. Analysis\n", None),           # special: insert before
    ("discussion", DISC_ANCHOR, DISC_ADD),
    ("what-if criterion (i)", DISC_WHATIF_ANCHOR, DISC_WHATIF_ADD),
    ("conclusion", CONC_ANCHOR, CONC_ADD),
    ("reproducibility", REPRO_ANCHOR, REPRO_ADD),
]

# Independent, idempotent edits (each checked by its own marker) so they can be
# applied to a manuscript that already carries the main patch above.
TABLE1_OLD = """**Table 1. Deployment outcomes over 32 decision groups.**
Regret is (oracle \u2212 deployed) on the holdout slice; lower is better."""
TABLE1_NEW = """**Table 1. Deployment outcomes over the frozen corpus (32 decision groups).**
Regret is (oracle \u2212 deployed) on the holdout slice; lower is better.
Full-corpus totals over all 77 deployment decisions, including the 45 groups of
\u00a75.6, are in **Table 8**."""

POST_EDITS = [
    ("table 1 -> table 8 pointer", TABLE1_OLD, TABLE1_NEW,
     "Full-corpus totals over all 77 deployment decisions"),
    # repair the reproducibility statement: (a) drop the heading + paragraph the
    # first attempt injected mid-sentence, (b) re-attach the paragraph at the end
    ("repro: drop injected block", REPRO_BROKEN, "\n", None),
    ("repro: append paragraph",
     "programmatically against those artifacts.\n\n# References",
     ("programmatically against those artifacts.\n\n" + REPRO_PARA
      + "\n\n# References"), REPRO_PARA[:60]),
]


def check_structure(text: str) -> None:
    """Catch the duplicate-heading / split-sentence class of patching bug."""
    heads = re.findall(r"^# .*$", text, flags=re.M)
    dupes = sorted({h for h in heads if heads.count(h) > 1})
    if dupes:
        raise SystemExit("duplicate headings in the patched manuscript: "
                         + "; ".join(dupes))
    n = text.count("programmatically against those artifacts.")
    if n != 1:
        raise SystemExit(f"reproducibility statement sentence appears {n}x (want 1)")


def apply_post(text: str) -> tuple[str, list[str]]:
    log: list[str] = []
    for name, old, new, marker in POST_EDITS:
        done = (old not in text) if marker is None else (marker in text)
        if done:
            log.append(f"already applied: {name}")
            continue
        if old not in text:
            log.append(f"skipped (anchor missing): {name}")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"anchor for {name!r} is not unique ({text.count(old)}x)")
        text = text.replace(old, new, 1)
        log.append(f"applied: {name}")
    return text, log


def apply(text: str, dry_run: bool = False) -> tuple[str, list[str]]:
    log: list[str] = []
    if MARK in text:
        return text, ["already applied -- nothing to do"]

    # 5.6 goes immediately before "# 6. Analysis"
    anchor = "\n# 6. Analysis\n"
    if anchor not in text:
        raise SystemExit("anchor for section 5.6 not found: '# 6. Analysis'")
    text = text.replace(anchor, MARK + "\n" + S56.rstrip() + "\n" + anchor, 1)
    log.append("inserted section 5.6 before '# 6. Analysis'")

    for name, a, add in PATCHES:
        if add is None:
            continue
        if a not in text:
            raise SystemExit(f"anchor not found for {name!r}:\n{a[:120]}")
        if text.count(a) != 1:
            raise SystemExit(f"anchor for {name!r} is not unique ({text.count(a)}x)")
        text = text.replace(a, a.rstrip() + "\n" + add.strip("\n") + "\n", 1)
        log.append(f"inserted: {name}")
    return text, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--md", default=str(MD), help="manuscript markdown to patch")
    args = ap.parse_args()
    md = pathlib.Path(args.md)
    if not md.exists():
        print(f"manuscript not found: {md}", file=sys.stderr)
        return 1
    src = md.read_text(encoding="utf-8")
    out, log = apply(src, args.dry_run)
    out, log2 = apply_post(out)
    log += log2
    check_structure(out)
    for line in log:
        print(" -", line)
    if args.dry_run or out == src:
        return 0
    md.write_text(out, encoding="utf-8")
    print(f"patched {md} ({len(src)} -> {len(out)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
