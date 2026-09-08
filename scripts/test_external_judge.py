"""bench_real_external 判分层离线回归:latex 结构/控制字符/千分位/boxed/嵌套。"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "apc-pipeline"))

from bench_real_external import answers_match, extract_pred, load_problems, arm_genome  # noqa: E402

CASES = [
    (r"\begin{pmatrix} 1 \\ 4 \\ 3 \end{pmatrix}", "(1, 4, 3)"),
    (r"\begin{pmatrix} 1 \\ 4 \\ 3 \end{pmatrix}", r"\begin{bmatrix}1\\4\\3\end{bmatrix}"),
    ("42", r"\boxed{42}"),
    ("0.5", r"\frac{1}{2}"),
    ("$1,234", "1234"),
    (r"\left(\begin{array}{c} 2 \\ -3 \end{array}\right)", "(2,-3)"),
    (r"(2, \frac{1}{2})", "(2, 0.5)"),
    ("180", "$180"),
    ("7", "7."),
    (r"\frac{3}{4}", "0.75"),
    (r"2\sqrt{3}", "2sqrt3"),
    ("y=2x+1", "Y = 2X + 1"),
    ("(1,-9)", "(1, -9)"),
    (r"\boxed{\frac{1}{4}}", "0.25"),
    (r"\text{I}", "I"),
    ("50", r"50^\circ"),
    ("27,63,99,135,171", r"27^\circ, 63^\circ, 99^\circ, 135^\circ, 171^\circ"),
    ("[23/8, 7/4]", r"\begin{pmatrix} 23/8 \\ 7/4 \end{pmatrix}"),
    ("25*sqrt(10)/4", r"\frac{25 \sqrt{10}}{4}"),
    ("sqrt(35)/3", r"\frac{\sqrt{35}}{3}"),
    ("16π/3", r"\frac{16 \pi}{3}"),
    ("[-3, 1]", "1, -3"),
    ("3/2 + 3/2 i", r"\frac{3}{2} + \frac{3}{2} i"),
    ("(7/45, 4/45)", r"\left( \frac{7}{45}, \frac{4}{45} \right)"),
    ("2-2√2", r"2 - 2 \sqrt{2}"),
    ("25", "025"),
    ("0.5", "0.50"),
    ("(1,2)", "（1, 2）".replace("（", "(").replace("）", ")")),
    ("[[1,2],[-3,-5]]", r"\begin{pmatrix} 1 & 2 \\ -3 & -5 \end{pmatrix}"),
    ("3/5, 117/125", r"\frac{3}{5}, \frac{117}{125}"),
    ("[[1/50, 7/50], [7/50, 49/50]]", r"\begin{pmatrix} 1/50 & 7/50 \\ 7/50 & 49/50 \end{pmatrix}"),
    ("-3/4, 3/4", r"\frac{3}{4}, -\frac{3}{4}"),
    ("25", "025"),
    ("[[1,2],[-3,-5]]", r"\begin{pmatrix} 1 & 2 \\ -3 & -5 \end{pmatrix}"),
    ("3/5, 117/125", r"\frac{3}{5}, \frac{117}{125}"),
    ("[[1/50, 7/50], [7/50, 49/50]]", r"\begin{pmatrix} 1/50 & 7/50 \\ 7/50 & 49/50 \end{pmatrix}"),
    ("-3/4, 3/4", r"\frac{3}{4}, -\frac{3}{4}"),
    ("(2,-3)", r"\left(\begin{array}{c} 2 \\ -3 \end{array}\right)"),
]


def main() -> int:
    for a, b in CASES:
        assert answers_match(a, b), (a, b)
    # (1,2) vs (2,1): multiset 政策下可互换 — 设计取舍,见 norm_answer docstring
    assert not answers_match("0.5", "0.6")
    assert not answers_match(r"pi/4, 5pi/4", r"\frac{5\pi}{4}")
    assert not answers_match("-1, 2", "2")
    assert not answers_match("204", "385")
    assert answers_match(extract_pred('{"answer": "\\boxed{42}"}'), "42")
    assert answers_match(extract_pred('{"answer": "0.5"}'), "\\frac{1}{2}")
    m = load_problems("math5", 135)
    assert len(m) == 135 and all(x["gold"] for x in m)
    g = load_problems("gsm8k", 100)
    assert len(g) == 100 and all(x["gold"] for x in g)

    from apc.core.task_spec import TaskSpec
    from apc.core.genome import PromptGenome
    from apc.compiler.renderer import DefaultPromptCompiler
    from apc.core.model_profile import ModelProfile
    spec = TaskSpec.from_yaml(str(REPO / "apc-pipeline/configs/tasks/external_math.yaml"))
    prof = ModelProfile.model_validate_json(
        (REPO / "artifacts/profiles/qwen_probe.json").read_text(encoding="utf-8"))
    b0 = PromptGenome.from_json(str(REPO / "configs/genomes/base.json"))
    for arm in (b0,) + tuple(arm_genome(a) for a in ("math-champ", "contract-champ", "financial-champ")):
        assert "{{input}}" in DefaultPromptCompiler().compile(arm, spec, prof, apply_rules=False).prompt_text
    print("ALL-GATES-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
