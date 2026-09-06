from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from apc.core.task_spec import TaskSpec


def _ngrams(s: str, n: int = 2) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def _bigram_f1(a: str, b: str) -> float:
    ga, gb = _ngrams(a), _ngrams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    if inter == 0:
        return 0.0
    p, r = inter / len(gb), inter / len(ga)
    return 2 * p * r / (p + r)


def _norm_value(v: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5.]", "", str(v))


def _value_eq(a: str, b: str) -> bool:
    """指标值等价：剥离货币/百分比单位后按数值比较（12.5亿元 == 12.5亿）。"""
    strip = lambda s: re.sub(r"[亿元万美美%＄¥\s]", "", str(s))
    na, nb = strip(a), strip(b)
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except (TypeError, ValueError):
        return False


class BaseJudge(ABC):
    """独立 Judge 接口：评估 accuracy / constraint_following / faithfulness / completeness（FR-5）。"""

    judge_id: str = "base"

    @abstractmethod
    def judge(self, input_text: str, expected: dict, actual: str, task_spec: TaskSpec) -> dict: ...


class RuleBasedJudge(BaseJudge):
    """确定性规则 Judge：与金标准比对 + 输入材料忠实性检查。"""

    judge_id = "rule_based_v1"

    def judge(self, input_text: str, expected: dict, actual: str, task_spec: TaskSpec) -> dict:
        try:
            data = json.loads(actual.strip())
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            return {"accuracy": 0.0, "constraint_following": 0.0, "faithfulness": 0.0, "completeness": 0.0}

        accuracy = self._accuracy(expected, data)
        constraints = task_spec.constraints
        risks = data.get("risks")
        c_risks = 1.0 if isinstance(risks, list) and len(risks) >= self._min_risks(constraints) else 0.0
        conf = data.get("confidence")
        c_conf = 1.0 if isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0 else 0.0
        c_summary = 1.0 if isinstance(data.get("summary"), str) and data["summary"].strip() else 0.0
        constraint_following = (c_risks + c_conf + c_summary) / 3
        faithfulness = self._faithfulness(input_text, data)
        completeness = self._completeness(task_spec, data)
        return {"accuracy": accuracy, "constraint_following": constraint_following,
                "faithfulness": faithfulness, "completeness": completeness}

    @staticmethod
    def _min_risks(constraints: list[str]) -> int:
        for c in constraints:
            m = re.search(r"至少\s*(\d+)\s*条", c)
            if m:
                return int(m.group(1))
        return 1

    def _accuracy(self, expected: dict, data: dict) -> float:
        scores = []
        if "summary" in expected:
            scores.append(_bigram_f1(str(expected["summary"]), str(data.get("summary", ""))))
        gold_metrics = expected.get("metrics", [])
        if gold_metrics:
            got = data.get("metrics", [])
            got_list = got if isinstance(got, list) else []
            per = []
            for gm in gold_metrics:
                best = 0.0
                for am in got_list:
                    if not isinstance(am, dict):
                        continue
                    name_hit = 1.0 if (gm.get("name") in str(am.get("name", "")) or
                                       str(am.get("name", "")) in gm.get("name", "")) and am.get("name") else 0.0
                    value_hit = 1.0 if _value_eq(gm.get("value", ""), am.get("value", "")) else 0.0
                    comment_hit = 1.0
                    if gm.get("comment"):
                        gc, ac = str(gm["comment"]), str(am.get("comment", ""))
                        comment_hit = 1.0 if (gc in ac or ac in gc) and ac else 0.0
                    best = max(best, 0.4 * name_hit + 0.4 * value_hit + 0.2 * comment_hit)
                per.append(best)
            scores.append(sum(per) / len(per) if per else 0.0)
        if "risks" in expected:
            gold_risks = set(_norm_value(r) for r in expected["risks"])
            got_risks = data.get("risks") if isinstance(data.get("risks"), list) else []
            got_norm = set(_norm_value(r) for r in got_risks)
            scores.append(len(gold_risks & got_norm) / len(gold_risks) if gold_risks else 0.0)
        if "confidence" in expected:
            try:
                diff = abs(float(expected["confidence"]) - float(data.get("confidence", 0)))
                scores.append(max(0.0, 1.0 - diff))
            except (TypeError, ValueError):
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _faithfulness(input_text: str, data: dict) -> float:
        """指标数值必须能在输入材料中找到（防虚构，FR-5 忠实性）。"""
        metrics = data.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            return 0.0
        grounded = 0
        for m in metrics:
            if isinstance(m, dict):
                v = _norm_value(str(m.get("value", "")))
                grounded += 1.0 if v and v in re.sub(r"\s+", "", input_text) else 0.0
        return grounded / len(metrics)

    def _completeness(self, task_spec: TaskSpec, data: dict) -> float:
        schema = task_spec.output.schema_ or {}
        if not schema:
            return 1.0
        present = sum(1 for k in schema if k in data and data[k] not in (None, "", []))
        return present / len(schema)


class LLMJudge(BaseJudge):
    """独立 Judge 模型（OpenAI-compatible）。模型不得与被评测模型相同，否则显式标注非独立。"""

    def __init__(self, client, judge_id: str | None = None):
        self.client = client
        self.judge_id = judge_id or f"llm:{client.model_id}"

    def judge(self, input_text: str, expected: dict, actual: str, task_spec: TaskSpec) -> dict:
        prompt = (
            "你是独立的评测 Judge。根据输入材料、金标准与模型实际输出，输出 JSON 评分。\n"
            f"输入材料：\n{input_text[:4000]}\n\n金标准：\n{json.dumps(expected, ensure_ascii=False)[:2000]}\n\n"
            f"模型实际输出：\n{actual[:2000]}\n\n"
            '只输出 JSON：{"accuracy": 0-1, "constraint_following": 0-1, "faithfulness": 0-1, "completeness": 0-1}'
        )
        try:
            result = self.client.complete(prompt, temperature=0.0)
            scores = json.loads(result.text.strip())
            out = {}
            for k in ("accuracy", "constraint_following", "faithfulness", "completeness"):
                out[k] = max(0.0, min(1.0, float(scores.get(k, 0.0))))
            return out
        except Exception:
            return {"accuracy": 0.0, "constraint_following": 0.0, "faithfulness": 0.0, "completeness": 0.0}


def create_judge(judge_name: str | None = None, eval_model_id: str | None = None) -> BaseJudge:
    """选择 Judge：默认规则 Judge；指定 llm:<model_id> 时用独立模型。"""
    from apc.models.factory import create_client
    if judge_name and judge_name.startswith("llm:"):
        judge_model = judge_name[4:]
        if eval_model_id and judge_model == eval_model_id:
            raise ValueError("Judge 模型不得与被评测模型相同（FR-5 独立性要求）")
        client = create_client(judge_model)
        return LLMJudge(client)
    return RuleBasedJudge()
