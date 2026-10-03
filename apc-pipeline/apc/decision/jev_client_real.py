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
        fallback_mode: str = "heuristic"
    ):
        """
        Args:
            api_base: API 端点基础 URL
            api_key: API 密钥
            fallback_mode: API 不可用时的回退模式
                - "heuristic": 启发式决策（贝叶斯后验）
                - "error": 抛出错误
        """
        self.api_base = api_base or os.getenv("JEV_API_BASE")
        self.api_key = api_key or os.getenv("JEV_API_KEY")
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
        """调用真实 jev-1.13.0 API"""
        import requests

        prompt = self._build_jev_prompt(arms, task_context)

        # 确保 API base 不以 /v1 结尾，避免重复
        api_base = self.api_base.rstrip('/')
        if api_base.endswith('/v1'):
            api_base = api_base[:-3]

        # 调用 /v1/systemone 端点
        response = requests.post(
            f"{api_base}/v1/systemone",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "jev-1.13.0",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are TypeSafe Jev, a System One structured decision model."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,  # 结构化决策，温度为 0
                "max_tokens": 800
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Jev API error: {response.status_code} {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 解析 JSON 响应
        parsed = self._parse_jev_response(content, arms)
        parsed['method'] = 'jev'

        return parsed

    def _build_jev_prompt(self, arms: List[Dict], task_context: Dict) -> str:
        """构建 Jev System One 风格的 prompt"""

        arms_text = "\n".join([
            f"  - {a['name']}: val_score={a['val_score']:.4f} (n=8)"
            for a in arms
        ])

        prompt = f"""**State**:
Task: {task_context.get('task', 'unknown')}
Seed: {task_context.get('seed', 42)}

**Arms (validation performance on n=8 samples)**:
{arms_text}

**Questions** (answer in JSON format):

1. **Choice**: Which arm maximizes expected holdout performance?
   Consider: validation scores, uncertainty (n=8 is small), simplicity as tie-breaker

2. **Noul** (Seed Collapse): Probability that one arm collapsed due to bad seed?
   Evidence: extreme score gap (e.g. 0.13 vs 0.67) indicates collapse

3. **Score** (Val-Holdout Alignment): How much trust validation scores (0-1)?
   Low trust if: small n, tight gaps, high variance expected

**Required JSON output**:
```json
{{
  "choice": "<arm_name>",
  "probabilities": {{"arm1": 0.5, "arm2": 0.3, ...}},
  "confidence": 0.85,
  "collapse_noul": 0.05,
  "alignment_score": 0.80,
  "reasoning": "<brief explanation>"
}}
```

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _parse_jev_response(self, content: str, arms: List[Dict]) -> Dict[str, Any]:
        """解析 Jev 返回的 JSON"""
        try:
            # 提取 JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = content[start:end]
            parsed = json.loads(json_str)

            # 验证必需字段
            required = ['choice', 'probabilities', 'confidence', 'collapse_noul', 'alignment_score']
            for field in required:
                if field not in parsed:
                    raise ValueError(f"Missing field: {field}")

            # 归一化概率
            probs = parsed['probabilities']
            total = sum(probs.values())
            if total > 0:
                parsed['probabilities'] = {k: v/total for k, v in probs.items()}

            # 限制范围
            parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
            parsed['collapse_noul'] = max(0.0, min(1.0, float(parsed['collapse_noul'])))
            parsed['alignment_score'] = max(0.0, min(1.0, float(parsed['alignment_score'])))

            return parsed

        except Exception as e:
            print(f"[JevClient] Parse error: {e}, content: {content[:200]}")
            raise

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
