"""探针与画像测试：15 探针、模型间确定性差异、15 维能力。"""
from __future__ import annotations

from apc.models.mock_client import MockClient
from apc.profiler.profile_builder import ProfileBuilder
from apc.profiler.probe_runner import ProbeRunner


def _profile(model_id: str, repo_root):
    runner = ProbeRunner(MockClient(model_id))
    results = runner.run_suite(str(repo_root / "probes" / "v1"))
    return results, ProfileBuilder().build(model_id, results)


def test_fifteen_probes_run(repo_root):
    results, profile = _profile("glm", repo_root)
    assert len(results) == 15
    scores = [float(r["score"]) if isinstance(r, dict) else float(r.score) for r in results]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_profile_has_fifteen_capability_dims(repo_root):
    _, profile = _profile("glm", repo_root)
    caps = profile.capability.model_dump()
    assert len(caps) == 15
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in caps.values())


def test_models_show_deterministic_differences(repo_root):
    _, glm = _profile("glm", repo_root)
    _, qwen = _profile("qwen", repo_root)
    # glm wrong_answer 偏差命中 math/table 探针；qwen 不命中
    assert glm.capability.math == 0.0
    assert qwen.capability.math == 1.0
    # qwen self_verification 偏差未通过 p11
    assert qwen.capability.self_verification_benefit == 0.0
    assert glm.capability.self_verification_benefit == 1.0
