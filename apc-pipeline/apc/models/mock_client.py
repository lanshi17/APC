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
    """财务分析应答：输出质量随 prompt 工程信号变化，使 genome 变异与迁移
    在 Mock 模式下产生可复现的分数差异（受控合成评测环境 v2）。

    基因效应（全部确定性、可复现；模型相关增益模拟真实的能力差异）：
    - examples.count: 0→1→2 逐步降低数值抽取错误率（模型相关增益）；
      3 无进一步增益（收益递减），count 经渲染器以 skeleton 个数编码。
    - reasoning.strategy: structured_checklist > hidden_analysis >
      decompose_then_answer > brief_plan > none（逐步降低抽取/推理错误）。
    - verification.type: source_grounding_check 提升忠实度（数值扎根），
      constraint_check 补足风险条数语义，format_check 降低格式噪声。
    - output.strictness: 经 schema 是否完整呈现传导（high 完整 schema，
      medium 字段列表，low 无 schema），影响缺失字段率。
    """
    guarded = {
        "json_only": "只输出 JSON" in prompt,
        "no_extra": "不要添加 schema 之外" in prompt or "额外字段" in prompt,
        "schema": '"metrics"' in prompt and '"summary"' in prompt,
        "n_examples": min(3, prompt.count("<按输入填写>")),
        # 注：“在输出前”（带“在”）只出现在 verification 段；output 高严格度的
        # “输出前请自查”不含“在”，故二者信号解耦（v2 修复 v1 的信号污染）。
        "verification": "在输出前" in prompt,
        "verif_grounding": "找到依据" in prompt,
        "verif_constraint": "逐条核对" in prompt,
        "verif_format": "格式是否完全符合" in prompt,
        "reason_checklist": "检查清单" in prompt,
        "reason_hidden": "不要输出分析过程" in prompt,
        "reason_decompose": "分解为子问题" in prompt,
        "reason_brief": "一两句话规划" in prompt,
    }
    # 模型相关的示例增益与饱和点（ground-truth 异质性 v2.2）：
    # 强模型早饱和（gpt:1，多余示例引入噪声）、弱模型需更多演示（glm:3）。
    # 最优 count 因模型而异（gpt=1/qwen=2/glm=3），是迁移衰减的主要来源。
    _EX_GAIN = {"glm": 0.06, "qwen": 0.09, "gpt": 0.035}
    _EX_SAT = {"glm": 3, "qwen": 2, "gpt": 1}
    ex_gain = _EX_GAIN.get(model_id, 0.05)
    ex_sat = _EX_SAT.get(model_id, 2)
    ex_level = guarded["n_examples"]  # 0..3
    ex_bonus = min(ex_sat, ex_level) * ex_gain - max(0, ex_level - ex_sat) * 0.06
    # 推理策略增益：模型相关的策略亲和（ground-truth 异质性 v2.1，
    # 模拟不同模型对推理脚手架的偏好差异；直接迁移时产生可测量的衰减，
    # 是迁移协议（FR-7/KR-6）评测的前提）。与 verification 的上位交互保留：
    # 无校验时增益减半。未知模型回退默认排序。
    _REASON_AFFINITY: dict[str, list[tuple[str, float, float]]] = {
        # (策略键, skill bonus, comment dropout)
        "glm": [("reason_decompose", 0.090, 0.00), ("reason_checklist", 0.060, 0.02),
                ("reason_hidden", 0.050, 0.04), ("reason_brief", 0.025, 0.12)],
        "qwen": [("reason_checklist", 0.090, 0.00), ("reason_hidden", 0.060, 0.02),
                 ("reason_decompose", 0.050, 0.04), ("reason_brief", 0.025, 0.12)],
        "gpt": [("reason_hidden", 0.090, 0.00), ("reason_checklist", 0.060, 0.02),
                ("reason_decompose", 0.050, 0.04), ("reason_brief", 0.025, 0.12)],
    }
    _DEFAULT_AFFINITY = [("reason_checklist", 0.090, 0.00), ("reason_hidden", 0.060, 0.02),
                         ("reason_decompose", 0.050, 0.04), ("reason_brief", 0.025, 0.12)]
    reason_bonus, drop_rate, drop_kind = 0.0, 0.25, "reason_drop_none"
    for key, bonus, drop in _REASON_AFFINITY.get(model_id, _DEFAULT_AFFINITY):
        if guarded.get(key):
            reason_bonus, drop_rate = bonus, drop
            drop_kind = f"reason_drop_{key}"
            break
    else:
        if guarded["reason_brief"]:
            reason_bonus, drop_rate, drop_kind = 0.025, 0.12, "reason_drop_brief"
    if not guarded["verification"]:
        reason_bonus *= 0.5
    if guarded["verif_grounding"]:
        ground_bonus = 0.050
    elif guarded["verif_constraint"]:
        ground_bonus = 0.025
    else:
        ground_bonus = 0.0
    skill = ex_bonus + reason_bonus + ground_bonus  # 技能分：越高错误率越低
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
    # 推理策略第二通道：drop_rate/drop_kind 已由上面的模型亲和表给出，
    # 此处只应用逐指标确定性丢失（comment 缺失 → judge comment_hit=0）。
    if drop_rate > 0 and metrics:
        for idx, m in enumerate(metrics):
            hh = int(hashlib.sha256(f"{model_id}:{drop_kind}:{document}:{idx}".encode()).hexdigest(), 16)
            if (hh % 1000) / 1000 < drop_rate:
                m["comment"] = ""
    # 数值抽取错误：基础 wrong_answer 率按技能分线性下降（保底 1% 噪声）。
    # 系数 1.5 使推理/示例/校验的满配与零配之间拉开约 0.25 的错误率差。
    base_rate = MODEL_DEVIATIONS.get(model_id, DEFAULT_DEVIATIONS).get("wrong_answer", 0.10)
    err_rate = max(0.01, base_rate + 0.22 - 1.5 * skill)
    h = int(hashlib.sha256(f"{model_id}:skill_err:{document}".encode()).hexdigest(), 16)
    if (h % 1000) / 1000 < err_rate and metrics:
        metrics[0]["value"] = "99"
    elif not ex_level and _rate(model_id, "wrong_answer", document) and metrics:
        metrics[0]["value"] = "99"
    # 置信度校准：示例越充分校准越好（2 个示例 0.78 最接近金标准均值 0.75，
    # 形成 0→1→2 的 uphill；与 err 通道同向叠加）。
    confidence = 0.78 if ex_level >= 2 else (0.66 if ex_level == 1 else 0.60)
    risks = ["业绩波动风险", "现金流压力风险", "应收账款回收风险"]
    if not guarded["verification"]:
        risks = risks[:2]  # 缺少输出前校验 → 约束「至少 3 条」不满足
    elif guarded["verif_format"] and not guarded["verif_constraint"] and not guarded["verif_grounding"]:
        risks = risks[:2] + ["格式自检通过"]  # format_check 只保格式不补语义：数量够但语义弱
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
