#!/usr/bin/env bash
# Install the P3/Week-4 paper deliverables into the paper repository.
#
#   bash docs/paper_patch/install_week4.sh [--dry-run]
#
# The paper repository is a sibling of this code repository
# (../02_apc_paper); override with APC_PAPER_DIR=/path/to/repo.
# Idempotent: the manuscript patch refuses to run twice, and every other step
# simply overwrites the artifacts it generated.  Originals of the touched
# manuscript files are kept in $PAPER/.p3_backup/.
set -euo pipefail

CODE="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER="${APC_PAPER_DIR:-$(dirname "$CODE")/02_apc_paper}"
PY="${PYTHON:-$CODE/.venv/bin/python}"
PP="$CODE/docs/paper_patch"
MANU="$PAPER/manuscript_negative"
SUBMIT="$PAPER/submission_tmlr"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
say() { echo; echo "=== $* ==="; }

[ -d "$PAPER" ] || { echo "paper repo not found: $PAPER" >&2; exit 1; }
[ -f "$MANU/negative-result.md" ] || { echo "manuscript not found" >&2; exit 1; }
export TMPDIR=/tmp          # sandbox default TMPDIR is read-only (pandoc/latexmk)

say "0. backup"
run mkdir -p "$PAPER/.p3_backup"
for f in negative-result.md negative-result.tex negative-result.pdf P3_TASK_TRACKER.md; do
  [ -f "$MANU/$f" ] && run cp -n "$MANU/$f" "$PAPER/.p3_backup/$f" || true
  [ -f "$PAPER/$f" ] && run cp -n "$PAPER/$f" "$PAPER/.p3_backup/$f" || true
done

say "1. patch the manuscript markdown (Results 5.6, Abstract, Intro, Discussion, Conclusion)"
run "$PY" "$PP/patch_manuscript.py" --md "$MANU/negative-result.md"

say "2. rebuild negative-result.tex / .pdf"
run "$PY" "$MANU/build.py"

say "3. supplementary material"
run "$PY" "$PP/make_supplementary.py" --out "$MANU"

say "4. TMLR submission build"
run mkdir -p "$MANU/tmlr"
[ -e "$MANU/tmlr/figures" ] || run ln -sfn ../figures "$MANU/tmlr/figures"
run "$PY" "$PP/make_tmlr.py"

say "5. task tracker + paper-side docs"
run cp "$PP/p3_task_tracker_updated.md" "$PAPER/P3_TASK_TRACKER.md"
run "$PY" "$PP/update_paper_docs.py" --paper "$PAPER"

say "6. submission package"
run mkdir -p "$SUBMIT"
run cp "$MANU/tmlr/negative-result-tmlr.pdf" "$SUBMIT/"
run cp "$MANU/tmlr/negative-result-tmlr.tex" "$SUBMIT/"
# the bibliography trio, so the submitted .tex compiles standalone
run cp "$MANU/tmlr/references.bib" "$SUBMIT/"
run cp "$MANU/tmlr/tmlr.bst" "$SUBMIT/"
run cp "$MANU/tmlr/negative-result-tmlr.bbl" "$SUBMIT/"
run cp "$MANU/tmlr/tmlr.sty" "$SUBMIT/"
run cp "$MANU/supplementary.pdf" "$SUBMIT/"
run cp "$PP/README_reproduce.md" "$SUBMIT/"
# the .tex pulls figures/fig1_failure_by_group.pdf, so the figure ships too
run cp -r "$MANU/figures" "$SUBMIT/"
if git -C "$CODE" diff --quiet && git -C "$CODE" diff --cached --quiet; then
  run bash "$PAPER/make_anon_bundle.sh" "$SUBMIT/code.zip"
  run cp "$SUBMIT/code.zip" "$MANU/code.zip"
else
  echo "!! code repo has uncommitted changes -- code.zip would miss them;" \
       "commit first, then re-run step 6" >&2
fi

say "7. verify every quoted number against the artifacts"
run "$PY" "$PP/verify_numbers.py" --md "$MANU/negative-result.md"

say "done"
echo "manuscript : $MANU/negative-result.pdf ($(pdfinfo "$MANU/negative-result.pdf" 2>/dev/null | awk '/^Pages/{print $2}') pages)"
echo "tmlr       : $MANU/tmlr/negative-result-tmlr.pdf"
echo "supplement : $MANU/supplementary.pdf"
echo "submission : $SUBMIT"
