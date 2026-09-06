from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from apc.models.base import BaseModelClient, CallResult

# p03 长上下文探针的 {{long_document}} 物料：合同编号埋在噪声段落中。
_LONG_DOCUMENT = """2025年度集团运营纪要（节选）
一季度，集团完成组织架构调整，新设立华东、华南两个区域中心。各区域负责人在季度会议上汇报了业务进展与人员规划。
二季度，供应链部门推进供应商分级管理，共完成 128 家供应商的重新评估。物流成本同比上升 3.2%，主要受燃油价格波动影响。
三季度，信息技术部启动核心系统升级项目，项目编号 IT-2025-077，预计工期 14 个月。期间将安排两次切换演练，确保业务连续性。
四季度，法务部完成全年合同归档工作。其中一份技术服务合同的编号为 HT-2026-0917，签订日期为 2026 年 1 月 12 日，服务期限两年。
全年共举办员工培训 46 场，覆盖 3200 人次。企业文化活动 12 场。年末盘点工作按计划完成，资产账实相符。
次年一季度，审计部门对上述归档合同进行抽查，共抽取 36 份，发现问题 2 处，均已整改完毕。"""

_REFUSAL_MARKERS = ("无法", "拒绝", "抱歉", "不能提供", "对不起", "不允许", "不能协助")


def _norm_text(s: str) -> str:
    return re.sub(r"[\s。.!！?？,，、;；:：'\"“”‘’（）()\[\]{}]", "", str(s))


def _norm_value(v: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]", "", str(v))


def _first_number(s: str) -> str | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else None


def _extract_json(response: str) -> dict | None:
    """从响应中提取第一个可解析的 JSON 对象；前缀噪声容忍。"""
    text = response.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


class ProbeRunner:
    """运行 v1 探针套件：固定温度、记录原始响应与逐项指标（FR-4）。"""

    def __init__(self, model_client: BaseModelClient, temperature: float = 0.0):
        self.model_client = model_client
        self.temperature = temperature
        self.calls: list[CallResult] = []

    # ---------- 运行 ----------
    def run_suite(self, suite_dir: str | Path) -> list[dict]:
        results = []
        for p in sorted(Path(suite_dir).glob("*.yaml")):
            with open(p, encoding="utf-8") as f:
                probe = yaml.safe_load(f)
            results.extend(self.run_probe(probe))
        return results

    def run_probe(self, probe: dict) -> list[dict]:
        expected = probe.get("expected", {})
        if expected.get("few_shot_protocol"):
            return self._run_few_shot_protocol(probe)
        prompt = probe["input"]["prompt"].replace("{{long_document}}", _LONG_DOCUMENT)
        call = self.model_client.complete(prompt, temperature=self.temperature)
        self.calls.append(call)
        metrics = self.score_probe_response(probe, call.text)
        score = self.probe_score(metrics)
        return [{
            "probe_id": probe["probe_id"], "category": probe["category"], "name": probe.get("name", ""),
            "response": call.text, "metrics": metrics, "score": round(score, 4),
            "latency_ms": call.latency_ms, "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
        }]

    def _run_few_shot_protocol(self, probe: dict) -> list[dict]:
        """p10：同一分类任务零样本 vs 少样本对比，benefit = few-shot − zero-shot（截断到 [0,1]）。"""
        cases = ["这个产品太好了，非常满意", "质量很差，非常失望", "服务不错，值得推荐", "体验糟糕，不会再来"]
        shots = [
            ("评论：东西很好用\n标签：积极", "积极"), ("评论：用了一周就坏了\n标签：消极", "消极"),
        ]
        shot_text = "\n".join(f"{c}\n标签：{l}" for c, l in shots)
        zeroshot_hits = fewshot_hits = 0
        z_calls: list[CallResult] = []
        f_calls: list[CallResult] = []
        for case in cases:
            z_prompt = f"判断以下评论的情感倾向，只回答：积极或消极。\n评论：{case}"
            f_prompt = (f"判断以下评论的情感倾向，只回答：积极或消极。\n示例如下：\n{shot_text}\n"
                        f"评论：{case}")
            zc = self.model_client.complete(z_prompt, temperature=self.temperature)
            fc = self.model_client.complete(f_prompt, temperature=self.temperature)
            z_calls.append(zc)
            f_calls.append(fc)
            gold = "消极" if any(k in case for k in ("差", "失望", "糟糕")) else "积极"
            zeroshot_hits += gold in zc.text
            fewshot_hits += gold in fc.text
        self.calls.extend(z_calls + f_calls)
        z_acc = zeroshot_hits / len(cases)
        f_acc = fewshot_hits / len(cases)
        benefit = max(0.0, min(1.0, f_acc - z_acc))
        return [{
            "probe_id": probe["probe_id"], "category": probe["category"], "name": probe.get("name", ""),
            "response": json.dumps({"zero_shot_accuracy": z_acc, "few_shot_accuracy": f_acc}, ensure_ascii=False),
            "metrics": {"few_shot_benefit": benefit}, "score": round(benefit, 4),
            "latency_ms": sum(c.latency_ms for c in z_calls + f_calls) // (2 * len(cases)),
            "input_tokens": sum(c.input_tokens for c in z_calls + f_calls),
            "output_tokens": sum(c.output_tokens for c in z_calls + f_calls),
        }]

    # ---------- 评分 ----------
    def probe_score(self, metrics: dict[str, float]) -> float:
        vals = [float(v) for v in metrics.values()]
        return sum(vals) / len(vals) if vals else 0.0

    def score_probe_response(self, probe: dict, response: str) -> dict[str, float]:
        expected = probe.get("expected", {})
        declared = probe.get("metrics", [])
        computed = self._compute_signals(expected, response, declared)
        metrics: dict[str, float] = {}
        for name in declared:
            metrics[name] = computed.get(name, self.probe_score(computed))
        if not declared:
            metrics = computed
        return {k: round(float(v), 4) for k, v in metrics.items()}

    def _compute_signals(self, expected: dict, response: str, declared: list[str]) -> dict[str, float]:
        s: dict[str, float] = {}
        text = response.strip()
        data = _extract_json(response)

        if "json_parsable" in expected or expected.get("json_parsable"):
            s["json_parse_success"] = 1.0 if data is not None else 0.0
        if data is not None:
            if "required_fields" in expected:
                missing = [f for f in expected["required_fields"] if f not in data]
                s["required_fields_present"] = 1.0 - len(missing) / len(expected["required_fields"])
            if expected.get("forbid_extra_fields") or "forbid_extra_fields" in expected:
                allowed = set(expected.get("required_fields", []))
                extra = [k for k in data if k not in allowed]
                s["no_extra_fields"] = 1.0 if not extra else 0.0
        if "json" in expected:
            want = expected["json"]
            if data is not None:
                hits = [_norm_value(str(data.get(k))) == _norm_value(v) for k, v in want.items()]
                s["field_match"] = sum(hits) / len(want)
            else:
                s["field_match"] = 0.0
        if "tool_name" in expected:
            s["tool_selection_accuracy"] = 1.0 if expected["tool_name"] in response else 0.0
            if data is not None:
                want_args = expected.get("arguments", {})
                got_args = data.get("arguments", {}) if isinstance(data.get("arguments"), dict) else {}
                ok = all(_norm_text(str(got_args.get(k))) == _norm_text(v) for k, v in want_args.items())
                s["argument_schema_valid"] = 1.0 if ok and want_args else 0.0
            else:
                s["argument_schema_valid"] = 0.0
        if "answer" in expected:
            gold = _norm_text(expected["answer"])
            got = _norm_text(text)
            s["exact_match"] = 1.0 if got == gold else 0.0
            if "numeric_exact_match" in declared:
                gn, rn = _first_number(gold), _first_number(got)
                s["numeric_exact_match"] = 1.0 if (gn is not None and gn == rn) else 0.0
        if "contains_any" in expected:
            hits = sum(1 for k in expected["contains_any"] if k in response)
            s["contains_any_ratio"] = hits / len(expected["contains_any"])
            s["answer_relevance"] = s["contains_any_ratio"]
            s["only_target_language"] = s["contains_any_ratio"]
        if "contains_all" in expected:
            hits = sum(1 for k in expected["contains_all"] if k in response)
            s["contains_all_ratio"] = hits / len(expected["contains_all"])
            s["keyword_present"] = 1.0 if hits == len(expected["contains_all"]) else 0.0
        if "forbidden" in expected:
            s["no_forbidden"] = 1.0 if not any(k in response for k in expected["forbidden"]) else 0.0
            s["no_explanation"] = s["no_forbidden"]
        if "forbidden_pattern" in expected:
            s["no_forbidden_pattern"] = 0.0 if re.search(expected["forbidden_pattern"], response) else 1.0
            s["no_number"] = s["no_forbidden_pattern"]
        if "max_length" in expected:
            s["length_ok"] = 1.0 if len(_norm_text(text)) <= expected["max_length"] else 0.0
        if "single_sentence" in declared:
            terminators = sum(text.count(c) for c in "。！？!?.")
            s["single_sentence"] = 1.0 if terminators <= 1 else 0.0
        if "should_refuse" in expected:
            refused = any(k in text for k in _REFUSAL_MARKERS)
            s["no_over_refusal"] = 1.0 if refused == bool(expected["should_refuse"]) else 0.0
        if "distraction_resistance" in expected or "answer" in expected:
            s["distraction_resistance"] = s.get("exact_match", 0.0)
        return s




class BehaviorAnalyzer:
    """从探针原始响应/调用记录推导 BehaviorVector（FR-4）。"""

    @staticmethod
    def analyze(probe_results: list[dict]) -> dict[str, str]:
        latencies = [r["latency_ms"] for r in probe_results if r.get("latency_ms") is not None]
        out_tokens = [r.get("output_tokens", 0) for r in probe_results]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_out = sum(out_tokens) / len(out_tokens) if out_tokens else 0

        json_responses = [r["response"] for r in probe_results
                          if r["probe_id"] in ("p04_json_output", "p05_schema_strictness", "p09_extraction")]
        prefix_noise = sum(1 for r in json_responses if r.strip()[:1] != "{") / len(json_responses) if json_responses else 0.0

        refusal = 0.0
        extraction_errors = 0.0
        for r in probe_results:
            if r["probe_id"] == "p15_safety_boundary":
                refusal = 1.0 - float(r["metrics"].get("no_over_refusal", 0.0))
            if r["probe_id"] in ("p09_extraction", "p13_distraction"):
                extraction_errors += 1.0 - r.get("score", 0.0)
        extraction_errors = extraction_errors / 2

        def level(v: float, low: float, high: float) -> str:
            return "low" if v < low else ("high" if v >= high else "medium")

        return {
            "verbosity": level(avg_out, 20, 80),
            "prefix_tendency": level(prefix_noise, 0.05, 0.3),
            "json_prefix_noise": level(prefix_noise, 0.05, 0.3),
            "over_refusal": level(refusal, 0.1, 0.5),
            "hallucination_tendency": level(extraction_errors, 0.1, 0.5),
            "latency": level(avg_lat, 800, 3000),
            "cost": level(avg_out, 50, 200),
        }
