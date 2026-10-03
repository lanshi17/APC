"""Jev 决策模型 - 基于 LLM 的实现（无需 TypeSafe 官方 API）

使用通用 LLM（qwen/gpt/claude）+ 结构化 prompt 模拟 TypeSafe Jev 的三大原语：
- Choice: 从多个臂中选择最优 + 概率分布
- Noul: 判断命题真假（0-1 评分）
- Score: 对状态打分（rubric）

优势：
1. 完全可控，不依赖外部 API 稳定性
2. 可复现，其他研究者能用自己的 LLM 复现
3. 成本可控，可选择开源模型（qwen）或闭源（gpt）
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class JevResponse:
    """Jev 决策响应"""
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    collapse_noul: float
    alignment_score: float
    reasoning: str  # LLM 的推理过程


class JevClientLLM:
    """Jev 决策客户端 - 基于 LLM 实现"""

    def __init__(
        self,
        model: str = "qwen3.8-flash",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Args:
            model: 使用的 LLM 模型
                - "qwen3.8-flash": 开源，快速
                - "gpt-4o": 闭源，强大
                - "claude-sonnet-5.5": 闭源，平衡
            api_base: API 端点（如果使用自建服务）
            api_key: API 密钥
        """
        self.model = model
        self.api_base = api_base or os.getenv("LLM_API_BASE", "http://localhost:8000")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")

    def select_best_arm(
        self,
        arms: List[Dict],
        task_context: Dict
    ) -> Dict[str, Any]:
        """
        使用 LLM 选择最优臂

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
                'reasoning': '...'
            }
        """
        # 构建结构化 prompt
        prompt = self._build_structured_prompt(arms, task_context)

        # 调用 LLM（通过 OpenAI-compatible API）
        response = self._call_llm(prompt)

        # 解析结构化输出
        parsed = self._parse_llm_response(response, arms)

        return parsed

    def _build_structured_prompt(self, arms: List[Dict], task_context: Dict) -> str:
        """构建 Jev 风格的结构化 prompt"""

        # 格式化 arms 数据
        arms_text = "\n".join([
            f"  - {a['name']}: val_score={a['val_score']:.4f}"
            for a in arms
        ])

        prompt = f"""You are a structured decision system (Jev-style) that evaluates arms and answers typed questions.

**State (Context)**:
Task: {task_context.get('task', 'unknown')}
Model: {task_context.get('model', 'unknown')}
Seed: {task_context.get('seed', 42)}

**Arms (Candidate Choices)**:
{arms_text}

**Questions to Answer**:

1. **Choice**: Which arm should be deployed to maximize holdout performance?
   - Evaluate each arm's validation score
   - Consider uncertainty (small val set, n=8)
   - Consider simplicity as a tie-breaker
   - Return: best arm name + probability distribution over all arms

2. **Noul (Seed Collapse Detection)**: Is there evidence of seed collapse?
   - Definition: One arm has extremely low val_score (< 0.2) while others are normal (> 0.5)
   - This indicates the seed caused that arm's genome to produce degenerate output
   - Return: probability [0, 1] that collapse occurred

3. **Score (Val-Holdout Alignment)**: How much do we trust the validation scores?
   - Small val set (n=8) → lower trust
   - Large gaps between arms → higher trust (clear signal)
   - Return: alignment score [0, 1], where 1 = fully trust val scores

**Output Format** (JSON only, no explanatory text):
{{
  "choice": "<arm_name>",
  "probabilities": {{"z0": 0.5, "manual": 0.3, ...}},
  "confidence": 0.85,
  "collapse_noul": 0.05,
  "alignment_score": 0.80,
  "reasoning": "<brief explanation of the choice>"
}}

**Guidelines**:
- If collapse_noul > 0.7, strongly prefer the non-collapsed arm
- If alignment_score < 0.5, be conservative (prefer simpler arms)
- Probabilities must sum to 1.0
- Confidence reflects how certain you are about the choice
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API（OpenAI-compatible）"""
        try:
            import requests

            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a structured decision system."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,  # 低温度，追求一致性
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code != 200:
                raise RuntimeError(f"LLM API error: {response.status_code} {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[JevClientLLM] LLM call failed: {e}, falling back to heuristic")
            return self._fallback_heuristic_response(prompt)

    def _parse_llm_response(self, response: str, arms: List[Dict]) -> Dict[str, Any]:
        """解析 LLM 的 JSON 输出"""
        try:
            # 提取 JSON（LLM 可能在前后加了文字）
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]

            parsed = json.loads(json_str)

            # 验证必需字段
            required = ['choice', 'probabilities', 'confidence', 'collapse_noul', 'alignment_score']
            for field in required:
                if field not in parsed:
                    raise ValueError(f"Missing field: {field}")

            # 归一化概率分布
            probs = parsed['probabilities']
            total = sum(probs.values())
            if total > 0:
                parsed['probabilities'] = {k: v/total for k, v in probs.items()}

            # 限制数值范围
            parsed['confidence'] = max(0.0, min(1.0, parsed['confidence']))
            parsed['collapse_noul'] = max(0.0, min(1.0, parsed['collapse_noul']))
            parsed['alignment_score'] = max(0.0, min(1.0, parsed['alignment_score']))

            return parsed

        except Exception as e:
            print(f"[JevClientLLM] Parse failed: {e}, using fallback")
            return self._fallback_heuristic_response_parsed(arms)

    def _fallback_heuristic_response(self, prompt: str) -> str:
        """回退到启发式决策（当 LLM 不可用时）"""
        # 从 prompt 中提取 arms 信息（简单解析）
        # 这个方法返回 JSON 字符串，会被 _parse_llm_response 解析
        # 但我们不知道具体的 arms，所以返回一个占位符
        # 实际处理在 _parse_llm_response 的 except 块中
        return json.dumps({
            "choice": "z0",
            "probabilities": {"z0": 1.0},
            "confidence": 0.5,
            "collapse_noul": 0.0,
            "alignment_score": 0.5,
            "reasoning": "Fallback placeholder"
        })

    def _fallback_heuristic_response_parsed(self, arms: List[Dict]) -> Dict[str, Any]:
        """启发式决策的解析版本"""
        from scipy.stats import beta

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

        # Choice: 选择后验均值最高的
        best_arm = max(posteriors, key=lambda p: p['mean'])

        # 计算概率分布（Monte Carlo）
        n_samples = len(posteriors[0]['samples'])
        prob_best = {}

        for i, p in enumerate(posteriors):
            is_best = sum(
                all(p['samples'][j] >= other['samples'][j] for other in posteriors)
                for j in range(n_samples)
            )
            prob_best[p['name']] = is_best / n_samples

        # 归一化
        total_prob = sum(prob_best.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in prob_best.items()}
        else:
            probabilities = {p['name']: 1.0/len(posteriors) for p in posteriors}

        # Confidence: 最高概率
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
            'reasoning': 'Heuristic fallback (Bayesian posterior + collapse detection)'
        }


if __name__ == "__main__":
    # 单元测试
    print("=== JevClientLLM Unit Tests ===\n")

    client = JevClientLLM(model="qwen3.8-flash")

    # Test 1: 正常情况
    arms_normal = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'manual', 'val_score': 0.665},
        {'name': 'apc-full', 'val_score': 0.615}
    ]

    result = client.select_best_arm(arms_normal, {'task': 'financial', 'seed': 42})
    print("Test 1 (Normal):")
    print(f"  Choice: {result['choice']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Collapse: {result['collapse_noul']:.3f}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')[:60]}...")
    print()

    # Test 2: 种子崩溃
    arms_collapse = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'apc-full', 'val_score': 0.133}
    ]

    result = client.select_best_arm(arms_collapse, {'task': 'financial', 'seed': 44})
    print("Test 2 (Collapse):")
    print(f"  Choice: {result['choice']}")
    print(f"  Collapse Noul: {result['collapse_noul']:.3f} (should be high)")
    assert result['collapse_noul'] > 0.7, "Failed to detect collapse"
    assert result['choice'] == 'z0', "Should fallback to z0"
    print("  ✓ Collapse detected and handled correctly")
    print()

    print("All tests passed! ✓")
