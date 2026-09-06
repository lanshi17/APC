"""TaskSpec 校验规则测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from apc.core.task_spec import TaskSpec


def _spec_dict(**overrides):
    base = dict(
        task_id="test_task_v1",
        name="测试任务",
        version="1.0",
        objective="输出 JSON 分析结果",
        output={"format": "json", "schema_": {"summary": "str", "confidence": "float"}},
        constraints=["risks 至少 3 条"],
        quality_weights={"accuracy": 0.5, "instruction_following": 0.2, "format": 0.15,
                         "robustness": 0.1, "efficiency": 0.05},
        cost_limits={"max_latency_ms": 3000, "max_output_tokens": 2000},
    )
    base.update(overrides)
    return base


def test_valid_spec_roundtrip():
    spec = TaskSpec.model_validate(_spec_dict())
    assert spec.task_id == "test_task_v1"


def test_weights_must_sum_to_one():
    bad = _spec_dict(quality_weights={"accuracy": 0.5, "instruction_following": 0.2,
                                     "format": 0.15, "robustness": 0.1, "efficiency": 0.2})
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(bad)


def test_negative_cost_limits_rejected():
    bad = _spec_dict(cost_limits={"max_latency_ms": -1, "max_output_tokens": 2000})
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(bad)


def test_bad_task_id_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(_spec_dict(task_id="bad id with spaces!"))


def test_strict_json_requires_schema():
    bad = _spec_dict(output={"type": "json", "strict": True})
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(bad)
