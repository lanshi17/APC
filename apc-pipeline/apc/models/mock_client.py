from __future__ import annotations
import hashlib
import re
import json
import time
from apc.models.base import BaseModelClient, CallResult

# 每个模型 id 的确定性偏差率：让无凭证环境下不同模型的画像/迁移差异可复现。
# 未知模型使用默认中等偏差率。
MODEL_DEVIATIONS: dict[str, dict[str, float]] = {
    "glm": {"extra_field": 0.10, "missing_field": 0.05, "prefix_noise": 0.00, "wrong_answer": 0.05},
    "qwen": {"extra_field": 0.20, "missing_field": 0.10, "prefix_noise": 0.15, "wrong_answer": 0.15},
    "gpt": {"extra_field": 0.05, "missing_field": 0.00, "prefix_noise": 0.00, "wrong_answer": 0.00},
}
DEFAULT_DEVIATIONS = {"extra_field": 0.15, "missing_field": 0.10, "prefix_noise": 0.10, "wrong_answer": 0.10}

_PROBE_ANSWERS: list[tuple[str, str]] = [
    # (prompt 关键词, 确定性正确响应)
    ("翻译成英文", "The market experienced significant volatility today."),
    ("不得出现数字", "公司存在风险，需持续关注。"),
    ("合同编号", "HT-2026-0917"),
    ("上海Q2", "135"),
    ("打 8 折", "140"),
    ("所有 A 都是 B", "不能"),
    ("公司并非没有风险", "有"),
    ("是否满足约束", "满足"),
    ("抽取合同金额", "50万元"),
    ("付款条款", "甲方应在验收后30日内支付款项。"),
    ("get_weather", '{"tool": "get_weather", "arguments": {"city": "北京"}}'),
    ("- title", '{"title": "示例商品", "price": 99}'),
    ("amount", '{"amount": "358000", "date": "2026-09-01"}'),
    ("情感倾向", ""),
]

_NUM_UNIT = re.compile(r"([\u4e00-\u9fa5]{2,4})([0-9]+(?:\.[0-9]+)?)(亿元|亿|万元|万|%)")
_CHANGE = re.compile(r"(同比|环比)(?:增长|上升|提高|下降|减少|下滑)([0-9.]+%)")


def _rate(model_id: str, kind: str, prompt: str) -> bool:
    rate = MODEL_DEVIATIONS.get(model_id, DEFAULT_DEVIATIONS).get(kind, 0.0)
    h = int(hashlib.sha256(f"{model_id}:{kind}:{prompt}".encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < rate


def _financial_json(document: str, prompt: str, model_id: str) -> str:
    """财务分析应答：输出质量随 prompt 工程信号变化（示例/校验/格式守卫），
    使 genome 变异与迁移在 Mock 模式下产生可复现的分数差异。"""
    guarded = {
        "json_only": "只输出 JSON" in prompt,
        "no_extra": "不要添加 schema 之外" in prompt or "额外字段" in prompt,
        "schema": '"metrics"' in prompt and '"summary"' in prompt,
        "examples": "示例" in prompt and "输出：" in prompt,
        "verification": "输出前" in prompt,
    }
    metrics = []
    for m in _NUM_UNIT.finditer(document):
        name, value, unit = m.group(1), m.group(2), m.group(3)
        tail = document[m.end():m.end() + 14]
        ch = _CHANGE.search(tail)
        change, comment = "", ""
        if ch:
            decline = any(k in tail[:ch.start()] for k in ("下降", "减少", "下滑"))
            direction = "下降" if decline else "增长"
            change = f"{'-' if decline else '+'}{ch.group(2)}"
            comment = f"{ch.group(1)}{direction}"
        metrics.append({"name": name, "value": f"{value}{unit}", "change": change, "comment": comment})
    if not guarded["examples"] and _rate(model_id, "wrong_answer", document) and metrics:
        metrics[0]["value"] = "99"
    confidence = 0.85 if guarded["examples"] else 0.6
    risks = ["业绩波动风险", "现金流压力风险", "应收账款回收风险"]
    if not guarded["verification"]:
        risks = risks[:2]  # 缺少输出前校验 → 约束「至少 3 条」不满足
    data = {
        "summary": "营收" + (metrics[0]["value"] if metrics else "数据不足") + "，需关注风险与现金流变化",
        "metrics": metrics,
        "risks": risks,
        "confidence": confidence,
    }
    if not guarded["no_extra"] and _rate(model_id, "extra_field", document):
        data["notes"] = "模型补充说明（额外字段）"
    if not guarded["schema"] and _rate(model_id, "missing_field", document):
        for k in ("confidence", "summary", "risks"):
            if k in data:
                data.pop(k)
                break
    text = json.dumps(data, ensure_ascii=False)
    if not guarded["json_only"] and _rate(model_id, "prefix_noise", document):
        text = "好的，以下是分析结果：\n" + text
    return text


class MockClient(BaseModelClient):
    """无外部凭证时的确定性客户端：固定应答 + 按模型 id 的确定性偏差。"""

    def __init__(self, model_id: str = "mock"):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str | None:
        return f"mock-1.0[{self._model_id}]"

    def complete(self, prompt: str, temperature: float = 0.0) -> CallResult:
        text = self._respond(prompt)
        h = int(hashlib.sha256(f"{self._model_id}:{prompt}".encode()).hexdigest(), 16)
        latency = 8 + h % 40
        return CallResult(
            text=text, model_id=self._model_id, model_version=self.model_version,
            temperature=temperature, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4,
            latency_ms=latency, finish_reason="stop",
        )

    def _respond(self, prompt: str) -> str:
        for keys, answer in _PROBE_ANSWERS:
            if keys and keys in prompt:
                if not answer:  # 情感分类探针
                    return self._sentiment(prompt)
                if _rate(self._model_id, "wrong_answer", prompt):
                    return "无法确定"
                return answer
        if "summary" in prompt and ("metrics" in prompt or "risks" in prompt):
            doc = self._extract_document(prompt)
            return _financial_json(doc, prompt, self._model_id)
        return '{"answer": 42}'

    @staticmethod
    def _extract_document(prompt: str) -> str:
        m = re.search(r"<document>\n?(.*?)\n?</document>", prompt, re.S)
        return m.group(1) if m else prompt

    @staticmethod
    def _sentiment(prompt: str) -> str:
        positive = ("好", "喜欢", "满意", "优秀", "推荐")
        negative = ("差", "失望", "问题", "糟糕", "难用")
        body = prompt.rsplit("评论", 1)[-1]
        pos = any(k in body for k in positive)
        neg = any(k in body for k in negative)
        return "积极" if pos and not neg else ("消极" if neg and not pos else "积极")
