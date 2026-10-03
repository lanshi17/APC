"""Jev 决策模型 - 使用真实 jev-1.13.0 API

通过 LLM API 聚合服务调用 TypeSafe 的 System One 结构化决策模型
- 端点: /v1/systemone
- 价格: $0.0193/1M input tokens, $0/1M output tokens
- 模型: jev-1.13.0 (TypeSafe System One)
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional
from scipy.stats import beta


class JevClient:
    """Jev 决策客户端 - 使用真实 jev-1.13.0 API"""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        fallback_mode: str = "heuristic"
    ):
        """
        Args:
            api_base: API 端点基础 URL（聚合服务的 /v1，如
                ``https://opencode.ai/zen/v1``）；客户端 POST 到 ``<base>/systemone``。
            api_key: API 密钥
            model: Jev 模型名；默认取 ``JEV_MODEL``（如 ``jev-1.13`` / ``jev-1.13-free``）
            timeout: 单次请求超时（秒）
            fallback_mode: API 不可用时的回退模式
                - "heuristic": 启发式决策（贝叶斯后验）
                - "error": 抛出错误
        """
        self.api_base = api_base or os.getenv("JEV_API_BASE")
        self.api_key = api_key or os.getenv("JEV_API_KEY")
        self.model = model or os.getenv("JEV_MODEL", "jev-1.13")
        self.timeout = timeout
        self.fallback_mode = fallback_mode

    def select_best_arm(
        self,
        arms: List[Dict],
        task_context: Dict
    ) -> Dict[str, Any]:
        """
        使用 Jev 选择最优臂

        Args:
            arms: [{'name': 'z0', 'val_score': 0.669}, ...]
            task_context: {'task': 'financial', 'seed': 42}

        Returns:
            {
                'choice': 'z0',
                'confidence': 0.85,
                'probabilities': {'z0': 0.6, 'manual': 0.3, ...},
                'collapse_noul': 0.05,
                'alignment_score': 0.80,
                'reasoning': '...',
                'method': 'jev' | 'heuristic'
            }
        """
        if self.api_base and self.api_key:
            try:
                return self._call_jev_api(arms, task_context)
            except Exception as e:
                print(f"[JevClient] API call failed: {e}")
                if self.fallback_mode == "error":
                    raise
                print(f"[JevClient] Falling back to {self.fallback_mode} mode")

        # Fallback
        return self._heuristic_decision(arms, task_context)

    def _call_jev_api(self, arms: List[Dict], task_context: Dict) -> Dict[str, Any]:
        """调用真实 Jev（原生 System One 协议）。

        请求体为 ``{state, model, questions}``，与官方 curl 示例一致；响应体为
        ``{model, answers, usage}``。此前实现误用 OpenAI chat 格式
        （``messages`` / ``temperature`` / ``max_tokens``），真实端点会拒绝为
        400 ``Invalid request``。
        """
        import requests

        api_base = self.api_base.rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
        url = f"{api_base}/v1/systemone"

        state, criteria = self._build_native_state(arms, task_context)
        payload = {
            "state": state,
            "model": self.model,
            "questions": {
                "best_arm": {
                    "type": "choice",
                    "instructions": (
                        "Which arm maximizes expected holdout performance? Validation "
                        "is only 8 samples, so treat gaps inside the +/-0.007 noise band "
                        "as ties and prefer the simpler genome when they tie."
                    ),
                    "criteria": criteria,
                },
                "seed_collapse": {
                    "type": "noul",
                    "instructions": (
                        "Is it likely that at least one arm's validation score collapsed "
                        "because of an unlucky seed rather than a genuinely worse genome?"
                    ),
                },
                "signal_quality": {
                    "type": "score",
                    "instructions": "How reliable is this validation signal for ranking the arms?",
                    "criteria": [
                        "unreliable - differences are within measurement noise",
                        "weakly informative",
                        "moderately informative",
                        "strongly informative",
                        "decisive - clearly separates the arms",
                    ],
                },
            },
        }

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Jev API error: {response.status_code} {response.text[:300]}")

        result = response.json()
        parsed = self._parse_native_response(result, arms)
        parsed["method"] = "jev"
        parsed["model"] = result.get("model", self.model)
        parsed["usage"] = result.get("usage", {})
        return parsed

    @staticmethod
    def _build_native_state(arms: List[Dict], task_context: Dict):
        """把臂列表渲染成 Jev 的 ``state`` 文本与 choice 的 ``criteria`` 映射。"""
        lines = []
        for a in arms:
            suffix = ""
            if a.get("rank") is not None:
                suffix = f" (simplicity rank {a['rank']})"
            lines.append(f"  - {a['name']}: val_score={a['val_score']:.4f} (n=8){suffix}")
        state = (
            "Deciding which compiled prompt genome to deploy.\n"
            f"Task family: {task_context.get('task', 'unknown')}. "
            f"Seed: {task_context.get('seed', 42)}.\n"
            "Every arm was scored on the same 8-sample validation set; score "
            "differences inside +/-0.007 are within measured noise.\n"
            "Arms and validation scores:\n" + "\n".join(lines)
        )
        criteria = {}
        for a in arms:
            bits = [f"validation score {a['val_score']:.4f}"]
            if a.get("rank") is not None:
                bits.append(f"simplicity rank {a['rank']}")
            criteria[a["name"]] = "; ".join(bits)
        return state, criteria

    def _parse_native_response(self, result: Dict, arms: List[Dict]) -> Dict[str, Any]:
        """把原生 ``answers`` 映射回本客户端既有的返回契约。"""
        answers = result.get("answers", {}) or {}

        best = answers.get("best_arm", {}) or {}
        choice = best.get("choice") or max(arms, key=lambda a: a["val_score"])["name"]
        probs = {str(k): float(v) for k, v in (best.get("probabilities") or {}).items()}
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {a["name"]: 1.0 / len(arms) for a in arms}
        confidence = best.get("confidence")
        confidence = float(confidence) if confidence is not None else max(probs.values())

        noul = answers.get("seed_collapse", {}) or {}
        collapse_noul = float(noul.get("noul", 0.0))

        sq = answers.get("signal_quality", {}) or {}
        legend = sq.get("legend") or {}
        raw_score = float(sq.get("score", 0.0))
        # 原生 score 是 0..len(legend)-1 的等级均值，归一到 0..1
        alignment = 0.0
        if legend:
            top = max(len(legend) - 1, 1)
            alignment = max(0.0, min(1.0, raw_score / top))

        return {
            "choice": choice,
            "confidence": max(0.0, min(1.0, confidence)),
            "probabilities": probs,
            "collapse_noul": max(0.0, min(1.0, collapse_noul)),
            "alignment_score": alignment,
            "signal_score_raw": raw_score,
            "reasoning": (
                f"native /v1/systemone: choice={choice}, "
                f"noul={collapse_noul:.3f}, signal_score={raw_score:.2f}"
            ),
            "raw_answers": answers,
        }

    def _heuristic_decision(self, arms: List[Dict], task_context: Dict) -> Dict[str, Any]:
        """启发式决策（贝叶斯后验 + 崩溃检测）"""

        # 计算后验分布
        posteriors = []
        for arm in arms:
            correct = arm.get('val_correct', int(arm['val_score'] * 8))
            total = arm.get('val_total', 8)
            correct = max(0, min(correct, total))

            post = beta(correct + 1, total - correct + 1)
            posteriors.append({
                'name': arm['name'],
                'mean': post.mean(),
                'samples': post.rvs(10000)
            })

        # Choice: 后验均值最高
        best_arm = max(posteriors, key=lambda p: p['mean'])

        # 概率分布（Monte Carlo）
        n_samples = len(posteriors[0]['samples'])
        prob_best = {}

        for p in posteriors:
            is_best_count = sum(
                all(p['samples'][j] >= other['samples'][j] for other in posteriors)
                for j in range(n_samples)
            )
            prob_best[p['name']] = is_best_count / n_samples

        # 归一化
        total_prob = sum(prob_best.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in prob_best.items()}
        else:
            probabilities = {p['name']: 1.0/len(posteriors) for p in posteriors}

        confidence = max(probabilities.values())

        # Collapse detection
        scores = [a['val_score'] for a in arms]
        min_score = min(scores)
        max_score = max(scores)
        gap = max_score - min_score

        if min_score < 0.15 and gap > 0.4:
            collapse_noul = 0.95
        elif min_score < 0.20 and gap > 0.3:
            collapse_noul = 0.75
        elif min_score < 0.30 and gap > 0.2:
            collapse_noul = 0.50
        else:
            collapse_noul = 0.05

        # Alignment
        n_val = arms[0].get('val_total', 8)
        if n_val <= 4:
            alignment_score = 0.3
        elif n_val <= 8:
            alignment_score = 0.8
        else:
            alignment_score = 0.95

        return {
            'choice': best_arm['name'],
            'confidence': float(confidence),
            'probabilities': {k: float(v) for k, v in probabilities.items()},
            'collapse_noul': float(collapse_noul),
            'alignment_score': float(alignment_score),
            'reasoning': 'Heuristic: Bayesian posterior + collapse heuristic',
            'method': 'heuristic'
        }


if __name__ == "__main__":
    print("=== JevClient Unit Tests ===\n")

    # Test with heuristic fallback (no API configured)
    client = JevClient(fallback_mode="heuristic")

    # Test 1: Normal
    arms1 = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'manual', 'val_score': 0.665},
        {'name': 'apc-full', 'val_score': 0.615}
    ]

    result = client.select_best_arm(arms1, {'task': 'financial', 'seed': 42})
    print("Test 1 (Normal):")
    print(f"  Choice: {result['choice']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Collapse: {result['collapse_noul']:.3f}")
    print(f"  Method: {result['method']}")
    print()

    # Test 2: Collapse
    arms2 = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'apc-full', 'val_score': 0.133}
    ]

    result = client.select_best_arm(arms2, {'task': 'financial', 'seed': 44})
    print("Test 2 (Collapse):")
    print(f"  Choice: {result['choice']}")
    print(f"  Collapse Noul: {result['collapse_noul']:.3f}")
    print(f"  Method: {result['method']}")

    assert result['collapse_noul'] > 0.7, "Failed to detect collapse"
    assert result['choice'] == 'z0', "Should pick z0"
    print("  ✓ Collapse detected correctly")
    print()

    print("All tests passed! ✓")
    print("\nTo use real Jev API, set environment variables:")
    print("  export JEV_API_BASE='https://your-api-gateway.com'")
    print("  export JEV_API_KEY='your-key'")
