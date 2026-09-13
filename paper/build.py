#!/usr/bin/env python3
"""docs/paper-apc-en.md → paper/apc-en.tex (ACL 样式)可复跑构建。

流程:unicode/mermaid 预处理(代码 span 保护)→ pandoc → longtable→tabular
(宽表 ≥5 列升级 table* 双栏)→ \texttt 断点注入 → 标题/摘要抽取 → TikZ
regime map 嵌入 → apc-en.tex。
"""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper-apc-en.md"
OUT = ROOT / "paper" / "apc-en.tex"

REP = {"≤": "$\\leq$", "≥": "$\\geq$", "→": "$\\rightarrow$", "⇒": "$\\Rightarrow$",
       "×": "$\\times$", "·": "$\\cdot$", "—": "---", "–": "--", "−": "-",
       "≠": "$\\neq$", "±": "$\\pm$", "∈": "$\\in$", "∩": "$\\cap$", "∪": "$\\cup$",
       "…": "$\\ldots$", "“": "``", "”": "''", "‘": "`", "’": "'",
       "α": "$\\alpha$", "β": "$\\beta$", "δ": "$\\delta$", "ε": "$\\epsilon$",
       "λ": "$\\lambda$", "μ": "$\\mu$", "π": "$\\pi$", "ρ": "$\\rho$", "σ": "$\\sigma$",
       "τ": "$\\tau$", "φ": "$\\phi$", "ω": "$\\omega$", "Δ": "$\\Delta$", "Θ": "$\\Theta$",
       "≈": "$\\approx$", "≡": "$\\equiv$", "↑": "$\\uparrow$", "↓": "$\\downarrow$",
       "§": "\\S{}", "₀": "$_0$", "≫": "$\\gg$"}


def preprocess(md: str) -> str:
    md = re.sub(r"```mermaid.*?```", "[[REGIME-MAP-TIKZ]]", md, flags=re.S)
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00SP{len(spans) - 1}\x00"

    md = re.sub(r"`[^`\n]+`", stash, md)
    for k, v in REP.items():
        md = md.replace(k, v)
    for i, sp in enumerate(spans):
        md = md.replace(f"\x00SP{i}\x00", sp)
    md = re.sub(r"`[^`\n]+`", lambda m: m.group(0).replace("−", "-").replace("₀", "0"), md)
    return md


def break_tt(tex: str) -> str:
    """\texttt{...} 内在 / 与 _ 后插 \allowbreak,消灭不可断长路径。"""

    def one(m: re.Match) -> str:
        t = m.group(1).replace("/", "/\\allowbreak{}").replace("_", "_\\allowbreak{}")
        return "\\texttt{" + t + "}"

    return re.sub(r"\\texttt\{([^{}]*)\}", one, tex)


def lt2tab(tex: str) -> str:
    out, pos = [], 0
    while True:
        k = tex.find(r"\begin{longtable}[]{", pos)
        if k < 0:
            out.append(tex[pos:])
            break
        out.append(tex[pos:k])
        i = k + len(r"\begin{longtable}[]{")
        depth, j = 1, i
        while depth:
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            j += 1
        raw_spec = tex[i:j - 1]
        spec = raw_spec.replace("\\columnwidth", "\\linewidth")
        e = tex.index(r"\end{longtable}", j) + len(r"\end{longtable}")
        ncols = len(re.findall(r"[lrc]", raw_spec.replace("@{}", "").replace("|", "")))
        inner = tex[j:e - len(r"\end{longtable}")]
        inner = re.sub(r"\\toprule\\noalign\{\}\s*", "", inner)
        inner = re.sub(r"\\midrule\\noalign\{\}\s*\\endhead\s*", lambda _: "\\midrule\n", inner)
        inner = re.sub(r"\\bottomrule\\noalign\{\}\s*\\endlastfoot\s*", "", inner)
        inner = inner.replace("\\midrule\\noalign{}", "\\midrule")
        inner = re.sub(r"^\s*\\midrule\s*", "", inner, count=1)
        inner = inner.replace("\\toprule", "\\midrule").replace("\\bottomrule", "\\midrule")
        tab = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + inner.strip()
               + "\n\\bottomrule\n\\end{tabular}")
        if ncols >= 5:  # 宽表跨双栏浮动
            out.append("\\begin{table*}[t]\n\\centering\n"
                       "\\resizebox{0.96\\textwidth}{!}{\n" + tab + "}\n\\end{table*}\n")
        else:  # 窄表:p 列均分栏宽,允许换行(消灭自然宽溢出)
            n = max(1, ncols)
            pspec = "@{}" + "".join(
                ">{\\raggedright\\arraybackslash}p{(\\linewidth - " + str(2 * n)
                + "\\tabcolsep) * \\real{" + f"{1.0 / n:.2f}" + "}}"
                for _ in range(n)) + "@{}"
            tabp = tab.replace("\\begin{tabular}{" + spec + "}", "\\begin{tabular}{" + pspec + "}", 1)
            out.append("{\\small\n\\noindent" + tabp + "}\n")
        pos = e
    return "".join(out)


TIKZ = r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  every node/.style={font=\small, align=center},
  box/.style={draw, rounded corners, fill=blue!4, inner sep=4pt},
  act/.style={draw, rounded corners, fill=green!6, inner sep=4pt},
  lab/.style={font=\scriptsize\itshape},
  edge/.style={->, thick}, dedge/.style={->, thick, dashed}]
\node[box] (A) {Before deployment: headroom-type diagnostic\\ per-problem z0 triage};
\node[draw, rounded corners, inner sep=3pt, below=4mm of A] (B) {Which dimension is missing?};
\node[box, below left=9mm and 1mm of B] (K) {Knowledge-type (F11, HLE, acc .10)\\ all six arms tie; reflection learned\\ content rules, zero transfer};
\node[box, below=9mm of B] (T) {Physical truncation (F10, max\_tokens 150)\\ six pathways pinned at judge floor .15};
\node[box, below right=9mm and 1mm of B] (P) {Protocol/format (F1--F9, financial)\\ saturated $\rightarrow$ all tie; unsaturated\\ weak model $\rightarrow$ structured $+0.05$};
\node[act, below=6mm of K] (X1) {Action: change model / add retrieval / fine-tune\\ prompt optimization = zero payoff};
\node[act, below=6mm of T] (X2) {Action: raise budget / shrink input\\ unreachable by any prompt pathway};
\node[act, below=6mm of P] (X3) {Action: the APO-valid domain\\ AutoAPC-Select gate (F8) picks the arm};
\node[box, below=52mm of T, dashed] (S) {Budget-truncation stacked cell (F11)\\ search/spatial/combinatorial types; all 6 arms fail;\\ short answer, long chain; a priori: type $\rightarrow$ timeout risk};
\draw[edge] (A) -- (B);
\draw[edge] (B) -- node[lab, left, pos=0.35] {acc gap = cannot} (K);
\draw[edge] (B) -- node[lab, right, pos=0.35] {acc gap = truncation} (T);
\draw[edge] (B) -- node[lab, right, pos=0.3] {acc fine, protocol} (P);
\draw[edge] (K) -- (X1); \draw[edge] (T) -- (X2); \draw[edge] (P) -- (X3);
\draw[dedge] (K) -- node[lab] {stacked in-problem} (S);
\end{tikzpicture}
\caption{Headroom typology decision map. The stacked budget cell (\S3.4m, GPQA \S3.4o)
is recognizable a priori from problem type.}
\label{fig:regime}
\end{figure*}"""

PREAMBLE = r"""\documentclass[11pt]{article}
% ACL-style preprint build of docs/paper-apc-en.md — DO NOT EDIT; run paper/build.py.
% Mechanical conversion: inline arXiv-ID citations stay as text; camera-ready switches to .bib+natbib.
\usepackage[final]{acl}
\usepackage{times,latexsym,amsmath,amssymb}
\usepackage{booktabs,longtable,array,multirow,calc}
\usepackage{graphicx,microtype,url,xcolor,textcomp}
\usepackage{tikz}
\usetikzlibrary{positioning}
\setcounter{secnumdepth}{0} % md headings carry their own 3.4m-style numbers
\sloppy
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\title{Adaptive Prompt Compiler: A Headroom Typology for Prompt Optimization\\ on Real Reasoning Models (Anonymous Preprint v0.3)}
\author{Anonymous submission}
\begin{document}
\maketitle
\begin{abstract}
"""


def main() -> int:
    body_md = preprocess(SRC.read_text(encoding="utf-8"))
    tmp = OUT.with_suffix(".body.md")
    tmp.write_text(body_md, encoding="utf-8")
    subprocess.run(["pandoc", "-f", "gfm+tex_math_dollars", "-t", "latex", "--wrap=none",
                    str(tmp), "-o", str(OUT.with_suffix(".body.tex"))], check=True)
    body = OUT.with_suffix(".body.tex").read_text(encoding="utf-8")
    body = body.replace("-\\/\\/-\\/-", "---").replace("-\\/-", "--")
    body = break_tt(body)
    i = body.index(r"\textbf{Abstract.}") + len(r"\textbf{Abstract.}")
    j = body.index(r"\subsection{1.")
    abs_text = body[i:j].strip()
    tex = (PREAMBLE + abs_text + "\n\\end{abstract}\n"
           + lt2tab(body[j:]) + "\n\\end{document}\n")
    while "-\\/-" in tex:  # pandoc 连字转义归一
        tex = tex.replace("-\\/-", "--")
    tex = tex.replace("{[}{[}REGIME-MAP-TIKZ{]}{]}", TIKZ)
    tex = tex.replace("[[REGIME-MAP-TIKZ]]", TIKZ)
    OUT.write_text(tex, encoding="utf-8")
    OUT.with_suffix(".body.tex").unlink()
    print(f"wrote {OUT} ({len(tex)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
