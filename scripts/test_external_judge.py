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
    ("(1,2)", "（1, 2）".replace("（", "(").replace("）", ")")),
]


def main() -> int:
    for a, b in CASES:
        assert answers_match(a, b), (a, b)
    assert not answers_match("(1,2)", "(2,1)")
    assert not answers_match("0.5", "0.6")
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
