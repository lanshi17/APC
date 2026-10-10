#!/usr/bin/env python3
"""Re-run every documented module and assert the artifact tree stays clean.

The manuscript claims that re-running the suite leaves ``git status`` clean in
both repositories.  That claim is only true while each module reproduces its
committed artifacts byte-for-byte, and it silently stopped being true when
``Harness`` began defaulting to the extended corpus: ``repair``, ``bayes_meta``
and ``figure_failure`` build a ``Harness()`` with no arguments and write to fixed
paths, so a plain re-run replaced the frozen 32-group artifacts (and Figure 1)
with 77-group versions -- while the paper still quoted the frozen numbers.

    python scripts/check_reproducibility.py            # full suite (~5-8 min)
    python scripts/check_reproducibility.py --quick    # frozen corpus only

Exit status is non-zero if any tracked artifact changed.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parents[1]
PAPER = pathlib.Path(os.environ.get("APC_PAPER_DIR", CODE.parent / "02_apc_paper"))
FIGDIR = PAPER / "manuscript_negative/figures"

FROZEN: list[list[str]] = [
    ["-m", "jev_rsi.experiments", "--corpus", "frozen"],
    ["-m", "jev_rsi.diagnosis"],
    ["-m", "jev_rsi.repair"],
    ["-m", "jev_rsi.bayes_meta"],
    ["-m", "jev_rsi.cross_task_rank"],
    ["-m", "jev_rsi.figure_failure", "--out", str(FIGDIR)],
]
# The scenario builders are response-cached, so the extended replay needs no API.
EXTENDED: list[list[str]] = [
    ["-m", "jev_rsi.experiments", "--corpus", "extended"],
    ["-m", "jev_rsi.analysis_extended"],
]


def run(cmd: list[str]) -> None:
    r = subprocess.run([sys.executable, *cmd], cwd=CODE,
                       capture_output=True, text=True, errors="replace")
    if r.returncode:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"FAILED: python {' '.join(cmd)}")
    print(f"  ran: python {' '.join(cmd)}")


def dirty(root: pathlib.Path, *paths: str) -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                       cwd=root, capture_output=True, text=True, errors="replace")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="frozen corpus and its five modules only")
    args = ap.parse_args()

    print(f"re-running the suite from {CODE}")
    for cmd in FROZEN + ([] if args.quick else EXTENDED):
        run(cmd)

    changed = ([("code", ln) for ln in dirty(CODE, "jev_rsi/results")]
               + [("paper", ln) for ln in dirty(PAPER, "manuscript_negative/figures")])
    if changed:
        print("\nFAIL: re-running the suite changed committed artifacts:")
        for repo, ln in changed:
            print(f"  [{repo}] {ln}")
        return 1
    print(f"\nOK: re-running the suite reproduced every artifact byte-for-byte"
          f"{' (frozen only)' if args.quick else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
