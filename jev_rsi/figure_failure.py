"""Figure 1 for the negative-result manuscript: failure analysis by group type.

Four panels over the 32 decision groups:

(a) exact-oracle rate by val/holdout alignment bucket, per policy;
(b) exact-oracle rate by effective signal count, per policy;
(c) exact-oracle rate by corpus family, per policy;
(d) where the greedy headroom actually sits -- its concentration in the two
    quarantined out-of-distribution groups.

Reads only the frozen artifacts; writes a vector PDF for LaTeX.

Usage
-----
    python -m jev_rsi.figure_failure --out /path/to/manuscript_negative/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import NOISE_BAND, JevGate, MetaParams, get_policy
from .diagnosis import _effective_signal, _rho
from .experiments import Harness

RESULTS = Path(__file__).resolve().parent / "results"
POLICIES = [("greedy", "#2b6cb0"), ("heuristic", "#d69e2e"), ("jev_static", "#c53030")]
GROUPS_FOR_LABEL = 8

# Fixed PDF timestamp so figures are byte-reproducible (see build()).
# 22 chars, exactly like matplotlib's "D:YYYYMMDDHHMMSS+HH'MM'", so the
# byte-level substitution in _strip_pdf_dates never shifts xref offsets.
_EPOCH = "D:19700101000000+00'00'"


def _strip_pdf_dates(path: Path) -> None:
    """Remove any remaining wall-clock /CreationDate|/ModDate from a PDF.

    ``savefig(metadata=...)`` suppresses matplotlib's own stamp on current
    versions, but the key can also arrive via a third-party backend, so scrub
    the bytes as a belt-and-braces pass.  Both the regexp and its replacement
    are fixed-width, so the file length never changes.
    """
    import re
    data = path.read_bytes()
    fixed = re.sub(rb"/CreationDate\s*\(D:\d{14}[^)]*\)",
                   b"/CreationDate (" + _EPOCH.encode() + b")", data)
    fixed = re.sub(rb"/ModDate\s*\(D:\d{14}[^)]*\)",
                   b"/ModDate (" + _EPOCH.encode() + b")", fixed)
    if fixed != data:
        path.write_bytes(fixed)


def _collect() -> List[Dict[str, Any]]:
    h = Harness(corpus="frozen")
    gate = JevGate(meta=MetaParams())
    heuristic = get_policy("heuristic")
    rows = []
    for g in h.decision_groups:
        rho, _ = _rho(g)
        # _rho returns NaN for a degenerate ranking; keep it as an explicit
        # "no val spread" bucket rather than letting NaN fall into the middle.
        # Rounding to 4dp matches the stored artifact, so the bucket a group
        # lands in here is the same one it lands in in diagnosis_results.json
        # (contract-s47 has raw rho 0.7999999999999999 == exactly 4/5).
        rho = None if (rho is None or np.isnan(rho)) else round(float(rho), 4)
        d_jev = gate.act(g, h.feats[g.gid])
        d_heur = heuristic(g)
        rows.append({
            "gid": g.gid,
            "family": g.family,
            "rho": rho,
            "effective": _effective_signal(g),
            "is_informative": g.is_informative,
            "greedy_correct": g.greedy.key == g.oracle.key,
            "heuristic_correct": d_heur.regret < 1e-9,
            "jev_static_correct": d_jev.regret < 1e-9,
            "greedy_regret": g.oracle.hold - g.greedy.hold,
        })
    return rows


def _rate(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return float(np.mean([r[f"{key}_correct"] for r in rows])) if rows else 0.0


def _grouped_bars(ax, labels: Sequence[str], slices: Sequence[Sequence[Dict[str, Any]]],
                  title: str, counts: bool = True) -> None:
    n = len(labels)
    x = np.arange(n)
    width = 0.26
    for i, (name, colour) in enumerate(POLICIES):
        vals = [_rate(s, name) for s in slices]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=colour, alpha=.9)
        if counts:
            for b, s in zip(bars, slices):
                if not s:
                    continue
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .02,
                        f"{sum(r[f'{name}_correct'] for r in s)}/{len(s)}",
                        ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel("exact-oracle rate", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=.25, linewidth=.5)
    ax.set_axisbelow(True)


def build(out_dir: Path) -> Path:
    rows = _collect()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axes[0][0]
    buckets = [
        # same typography as the manuscript's Table 5 ("\u03c1 \u2265 0.8", ...)
        ("\u03c1 \u2265 .8", [r for r in rows if r["rho"] is not None and r["rho"] >= .8]),
        (".3 \u2264 \u03c1 < .8", [r for r in rows if r["rho"] is not None and .3 <= r["rho"] < .8]),
        ("\u03c1 < .3", [r for r in rows if r["rho"] is not None and r["rho"] < .3]),
        ("\u03c1 undefined", [r for r in rows if r["rho"] is None]),
    ]
    _grouped_bars(ax, [b[0] for b in buckets], [b[1] for b in buckets],
                  "(a) by val/holdout rank alignment")

    ax = axes[0][1]
    eff = [("1", [r for r in rows if r["effective"] == 1]),
           ("2", [r for r in rows if r["effective"] == 2]),
           ("3", [r for r in rows if r["effective"] == 3])]
    _grouped_bars(ax, [f"effective = {e[0]}" for e in eff], [e[1] for e in eff],
                  "(b) by effective signal count")

    ax = axes[1][0]
    fams: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fams.setdefault(r["family"] or "other", []).append(r)
    order = sorted(fams, key=lambda k: (-len(fams[k]), k))
    _grouped_bars(ax, [f"{k}\n(n={len(fams[k])})" for k in order], [fams[k] for k in order],
                  "(c) by corpus family")

    ax = axes[1][1]
    top = sorted(rows, key=lambda r: -r["greedy_regret"])[:GROUPS_FOR_LABEL]
    total = sum(r["greedy_regret"] for r in rows)
    names = [r["gid"][:22] for r in top][::-1]
    vals = [r["greedy_regret"] for r in top][::-1]
    colours = ["#c53030" if r["is_informative"] else "#a0aec0" for r in top][::-1]
    ax.barh(np.arange(len(top)), vals, color=colours, alpha=.9)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("greedy regret (oracle - greedy)", fontsize=8)
    ax.set_title("(d) headroom concentration", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=.25, linewidth=.5)
    ax.set_axisbelow(True)
    share = sum(vals[-2:]) / total
    ax.text(.97, .06, f"top 2 groups = {share:.0%} of all headroom",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#c53030")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, .965))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig1_failure_by_group.pdf"
    # Determinism: matplotlib stamps the PDF /CreationDate with the wall clock,
    # which makes an otherwise identical figure differ byte-for-byte between
    # runs.  The whole artifact set for this project is meant to be reproducible
    # byte-for-byte, so pin the metadata to a fixed epoch instead.
    fig.savefig(path, bbox_inches="tight",
                metadata={"CreationDate": _EPOCH, "ModDate": _EPOCH})
    _strip_pdf_dates(path)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    path = build(Path(args.out))
    print(f"wrote {path}")
    rows = _collect()
    total = sum(r["greedy_regret"] for r in rows)
    top2 = sorted(rows, key=lambda r: -r["greedy_regret"])[:2]
    print(f"  total greedy headroom {total:.4f}; top 2 = "
          f"{sum(r['greedy_regret'] for r in top2) / total:.1%} "
          f"({', '.join(r['gid'] for r in top2)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
