from __future__ import annotations

import json
import re

from apc.core.task_spec import TaskSpec


class RuleBasedChecker:
    """规则检查（先于 Judge 执行，FR-5）：JSON 可解析、字段完整、无额外字段、约束满足。"""

    def check(self, output: str, task_spec: TaskSpec) -> dict:
        result: dict = {"json_parsable": False, "format_score": 0.0, "constraint_score": 0.0}
        try:
            data = json.loads(output.strip())
            if not isinstance(data, dict):
                raise ValueError("top-level is not an object")
        except Exception as e:
            result["format_error"] = f"json_parse_failed: {type(e).__name__}"
            return result
        result["json_parsable"] = True
        schema = task_spec.output.schema_ or {}
        if not schema:
            result["format_score"] = 1.0
        else:
            missing = [k for k in schema if k not in data]
            extra = [k for k in data if k not in schema]
            result["missing_fields"] = missing
            result["extra_fields"] = extra
            penalty = 0.6 * (len(missing) / len(schema)) + 0.4 * (len(extra) / max(len(data), 1))
            result["format_score"] = round(max(0.0, 1.0 - penalty), 4)
        result["constraint_score"] = self._constraint_score(data, task_spec)
        return result

    @staticmethod
    def _constraint_score(data: dict, task_spec: TaskSpec) -> float:
        checks = []
        min_risks = 1
        for c in task_spec.constraints:
            m = re.search(r"至少\s*(\d+)\s*条", c)
            if m:
                min_risks = int(m.group(1))
        risks = data.get("risks")
        checks.append(1.0 if isinstance(risks, list) and len(risks) >= min_risks else 0.0)
        conf = data.get("confidence")
        checks.append(1.0 if isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0 else 0.0)
        summary = data.get("summary")
        checks.append(1.0 if isinstance(summary, str) and summary.strip() else 0.0)
        return round(sum(checks) / len(checks), 4)

    @staticmethod
    def format_error(check_result: dict) -> bool:
        """格式错误：JSON 不可解析，或存在缺失/额外字段（KR-7 口径）。"""
        return not check_result.get("json_parsable") or bool(check_result.get("missing_fields")) or bool(
            check_result.get("extra_fields"))
