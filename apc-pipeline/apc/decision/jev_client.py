"""TypeSafe Jev API 封装，用于结构化决策

替换 auto_apc_gate.py 的硬编码规则，提供：
1. Choice 原语：从多个臂中选择最优
2. Noul 原语：检测种子崩溃
3. Score 原语：评估 val-holdout 对齐度

所有问题并行评估，返回 probabilities + confidence。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ChoiceResponse:
    """Choice 原语响应"""
    choice: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class NoulResponse:
    """Noul 原语响应"""
    noul: float  # 0-1，陈述为真的概率
    confidence: float


@dataclass
class ScoreResponse:
    """Score 原语响应"""
    score: float  # rubric 定义的范围
    probabilities: Optional[Dict[str, float]] = None
    confidence: float = 0.0


@dataclass
class JevEvaluationResponse:
    """Jev 并行评估响应"""
    choice: ChoiceResponse
    collapse_noul: NoulResponse
    alignment_score: ScoreResponse


class JevDecisionClient:
    """TypeSafe Jev API 客户端

    如果 TypeSafe API 不可用，fallback 到模拟模式。
    """

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.mock_mode = mock_mode or not self.api_key

        if not self.mock_mode:
            try:
                # 尝试导入 TypeSafe SDK
                from typesafe import TypeSafeClient
                self.client = TypeSafeClient(self.api_key)
                print("[JevClient] Using TypeSafe API")
            except ImportError:
                print("[JevClient] TypeSafe SDK not found, falling back to mock mode")
                self.mock_mode = True

        if self.mock_mode:
            print("[JevClient] Running in MOCK mode (for development)")

    def select_best_arm(
        self,
        arms: List[Dict[str, Any]],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        选择最优臂（替换 auto_apc_gate.py 的 pick 函数）

        Args:
            arms: [
                {'name': 'z0', 'val_score': 0.669, 'val_correct': 5, 'val_total': 8},
                {'name': 'apc-full', 'val_score': 0.133, 'val_correct': 1, 'val_total': 8},
                ...
            ]
            task_context: {
                'task': 'financial',
                'model': 'qwen3.8-flash',
                'seed': 42
            }

        Returns:
            {
                'choice': 'z0',
                'confidence': 0.95,
                'probabilities': {'z0': 0.85, 'full': 0.15},
                'collapse_noul': 0.05,
                'alignment_score': 0.82
            }
        """
        if self.mock_mode:
            return self._mock_select(arms, task_context)

        # 构建 Jev state
        state = {
            'arms': arms,
            'task': task_context.get('task', 'unknown'),
            'model': task_context.get('model', 'unknown'),
            'seed': task_context.get('seed')
        }

        # TypeSafe Jev 调用（真实模式）
        try:
            from typesafe import Choice, Noul, Score

            response = self.client.evaluate(
                state=state,
                questions=[
                    Choice(
                        question="Which arm should be deployed?",
                        options=[arm['name'] for arm in arms],
                        context={
                            'val_scores': {arm['name']: arm['val_score'] for arm in arms},
                            'arm_complexity': self._compute_complexity(arms)
                        }
                    ),
                    Noul(
                        statement="The best-scoring arm shows seed collapse",
                        evidence={
                            'extreme_val_deficit': max(a['val_score'] for a in arms) < 0.2,
                            'bimodal_pattern': self._check_bimodal(arms)
                        }
                    ),
                    Score(
                        question="How well will val scores predict holdout performance?",
                        rubric={
                            'sample_size': 'n_val >= 8',
                            'task_stability': 'low variance across seeds',
                            'distribution_match': 'val and holdout from same pool'
                        },
                        scale=(0, 1)
                    )
                ]
            )

            # 解析响应
            return {
                'choice': response.choice[0],
                'confidence': response.choice.confidence,
                'probabilities': response.choice.probabilities,
                'collapse_noul': response.noul[1],
                'alignment_score': response.score[2]
            }

        except Exception as e:
            print(f"[JevClient] TypeSafe API error: {e}, falling back to mock")
            return self._mock_select(arms, task_context)

    def _mock_select(
        self,
        arms: List[Dict[str, Any]],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        模拟 Jev 决策（用于开发和测试）

        基于启发式规则 + 贝叶斯后验估计
        """
        import numpy as np
        from scipy.stats import beta

        # 计算后验分布
        posteriors = []
        for arm in arms:
            correct = arm.get('val_correct', int(arm['val_score'] * 8))
            total = arm.get('val_total', 8)

            # 边界检查：确保 correct 在 [0, total] 范围内
            correct = max(0, min(correct, total))

            # Beta 后验 (alpha, beta 参数)
            alpha = correct + 1
            beta_param = total - correct + 1

            post = beta(alpha, beta_param)
            posteriors.append({
                'name': arm['name'],
                'val_score': arm['val_score'],
                'mean': post.mean(),
                'ci_95': post.interval(0.95),
                'samples': post.rvs(10000)
            })

        # Choice: 选择最优臂
        best_arm = max(posteriors, key=lambda p: p['mean'])

        # 计算概率分布（通过 Monte Carlo）
        prob_best = {}
        n_samples = len(posteriors[0]['samples'])

        for i, p in enumerate(posteriors):
            # P(arm_i 是最优) = 在多少样本中 arm_i >= 所有其他臂
            wins = 0
            for j in range(n_samples):
                is_best = all(
                    p['samples'][j] >= other['samples'][j]
                    for k, other in enumerate(posteriors) if k != i
                )
                if is_best:
                    wins += 1
            prob_best[p['name']] = wins / n_samples

        # 归一化
        total_prob = sum(prob_best.values())
        probabilities = {k: v / total_prob for k, v in prob_best.items()}

        # Confidence: 最优臂的优势
        confidence = probabilities[best_arm['name']]

        # Noul: 崩溃检测
        collapse_noul = self._compute_collapse_probability(arms)

        # Score: val-holdout 对齐度
        alignment_score = self._compute_alignment_score(arms, task_context)

        return {
            'choice': best_arm['name'],
            'confidence': float(confidence),
            'probabilities': {k: float(v) for k, v in probabilities.items()},
            'collapse_noul': float(collapse_noul),
            'alignment_score': float(alignment_score)
        }

    def _compute_collapse_probability(self, arms: List[Dict]) -> float:
        """计算种子崩溃概率（启发式）

        检测逻辑：
        1. 存在极端低分（< 0.2）
        2. 且与最高分差距大（> 0.4）
        """
        if not arms:
            return 0.0

        scores = [a['val_score'] for a in arms]
        min_score = min(scores)
        max_score = max(scores)
        gap = max_score - min_score

        # 双峰崩溃模式：高低分差距大
        if min_score < 0.15 and gap > 0.4:
            return 0.95
        elif min_score < 0.20 and gap > 0.3:
            return 0.75
        elif min_score < 0.30 and gap > 0.2:
            return 0.50
        else:
            return 0.05  # 正常情况

    def _compute_alignment_score(
        self,
        arms: List[Dict],
        task_context: Dict
    ) -> float:
        """计算 val-holdout 对齐度（启发式）"""
        n_val = arms[0].get('val_total', 8)

        # 小样本 → 低对齐度
        if n_val < 5:
            return 0.3
        elif n_val < 8:
            return 0.5
        else:
            return 0.8  # 充分样本

    def _compute_complexity(self, arms: List[Dict]) -> Dict[str, int]:
        """计算臂的复杂度（用于 Occam tie-break）"""
        complexity_order = {
            'zero-shot': 0, 'z0': 0,
            'manual': 1,
            'apc-safe': 2, 'safe': 2,
            'apc-full': 3, 'full': 3,
            'gepa': 4,
            'transfer-0': 1, 't0': 1,
            'transfer-ws': 2, 'ws': 2,
            'cold': 3
        }
        return {arm['name']: complexity_order.get(arm['name'], 5) for arm in arms}

    def _check_bimodal(self, arms: List[Dict]) -> bool:
        """检测双峰分布（崩溃信号）"""
        scores = [a['val_score'] for a in arms]
        if len(scores) < 2:
            return False

        # 简化版：最大-最小 > 0.5 且存在 < 0.2 的分数
        return (max(scores) - min(scores) > 0.5) and any(s < 0.2 for s in scores)


if __name__ == "__main__":
    # 单元测试
    client = JevDecisionClient(mock_mode=True)

    # 测试案例 1：正常情况
    arms_normal = [
        {'name': 'z0', 'val_score': 0.669, 'val_correct': 5, 'val_total': 8},
        {'name': 'manual', 'val_score': 0.665, 'val_correct': 5, 'val_total': 8},
        {'name': 'apc-full', 'val_score': 0.615, 'val_correct': 5, 'val_total': 8}
    ]

    result = client.select_best_arm(arms_normal, {'task': 'financial', 'seed': 42})
    print("Test 1 (Normal):")
    print(f"  Choice: {result['choice']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Collapse: {result['collapse_noul']:.3f}")
    print(f"  Alignment: {result['alignment_score']:.3f}")
    print()

    # 测试案例 2：种子崩溃
    arms_collapse = [
        {'name': 'z0', 'val_score': 0.669, 'val_correct': 5, 'val_total': 8},
        {'name': 'apc-full', 'val_score': 0.133, 'val_correct': 1, 'val_total': 8}
    ]

    result = client.select_best_arm(arms_collapse, {'task': 'financial', 'seed': 44})
    print("Test 2 (Collapse):")
    print(f"  Choice: {result['choice']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Collapse: {result['collapse_noul']:.3f} (should be high)")
    print(f"  Alignment: {result['alignment_score']:.3f}")
