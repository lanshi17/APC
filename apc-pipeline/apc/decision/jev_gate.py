"""Jev 版本的 AutoAPC-Select 门控

替换 scripts/auto_apc_gate.py 的启发式规则，提供：
1. 结构化决策（Choice/Noul/Score）
2. 置信度量化
3. 可学习的元参数（为 RSI 准备）
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from .jev_client import JevDecisionClient


class JevGate:
    """Jev 决策门控（无 RSI）"""

    def __init__(self, client: Optional[JevDecisionClient] = None):
        self.client = client or JevDecisionClient()

        # 可调节的元参数（后续 RSI 会学习这些）
        self.meta_params = {
            'collapse_threshold': 0.7,      # Noul > 此值 → 触发崩溃回退
            'alignment_threshold': 0.5,     # Score < 此值 → val 不可信
            'confidence_floor': 0.6         # Confidence < 此值 → 保守选择
        }

    def select(
        self,
        arms: List[Dict],
        task_context: Dict
    ) -> Dict[str, Any]:
        """
        使用 Jev 选择最优臂

        Args:
            arms: [{'name': 'z0', 'val_score': 0.669, ...}, ...]
            task_context: {'task': 'financial', 'model': 'qwen', 'seed': 42}

        Returns:
            {
                'deployed_arm': 'z0',
                'reason': 'collapse_detected',
                'jev_response': {...},
                'meta_params_used': {...}
            }
        """
        # 调用 Jev
        jev_response = self.client.select_best_arm(arms, task_context)

        # 决策逻辑（使用可学习的 meta_params）

        # 规则 1：崩溃检测
        if jev_response['collapse_noul'] > self.meta_params['collapse_threshold']:
            return {
                'deployed_arm': 'z0',  # 安全回退
                'reason': 'collapse_detected',
                'jev_response': jev_response,
                'meta_params_used': self.meta_params.copy()
            }

        # 规则 2：val 不可信
        if jev_response['alignment_score'] < self.meta_params['alignment_threshold']:
            return {
                'deployed_arm': 'z0',  # val 不可信，回退 z0
                'reason': 'low_val_alignment',
                'jev_response': jev_response,
                'meta_params_used': self.meta_params.copy()
            }

        # 规则 3：低置信度
        if jev_response['confidence'] < self.meta_params['confidence_floor']:
            # 不确定时，选择更简单的臂
            # 从概率分布中选择简单性最高且概率 > 0.1 的臂
            simple_candidates = [
                (name, prob) for name, prob in jev_response['probabilities'].items()
                if prob > 0.1
            ]

            if simple_candidates:
                # 按 simplicity 排序（z0 最简单）
                simple_order = ['z0', 'zero-shot', 'manual', 'safe', 'apc-safe',
                                'full', 'apc-full', 'gepa', 'transfer-0', 'cold']

                for simple_name in simple_order:
                    if any(name == simple_name for name, _ in simple_candidates):
                        return {
                            'deployed_arm': simple_name,
                            'reason': 'low_confidence_fallback',
                            'jev_response': jev_response,
                            'meta_params_used': self.meta_params.copy()
                        }

            # 如果没有合适候选，回退 z0
            return {
                'deployed_arm': 'z0',
                'reason': 'low_confidence_fallback',
                'jev_response': jev_response,
                'meta_params_used': self.meta_params.copy()
            }

        # 规则 4：正常选择
        return {
            'deployed_arm': jev_response['choice'],
            'reason': 'normal_selection',
            'jev_response': jev_response,
            'meta_params_used': self.meta_params.copy()
        }


if __name__ == "__main__":
    # 单元测试
    print("=== JevGate Unit Tests ===\n")

    gate = JevGate()

    # Test 1: 正常情况
    arms_normal = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'manual', 'val_score': 0.665},
        {'name': 'apc-full', 'val_score': 0.615}
    ]

    result = gate.select(arms_normal, {'task': 'financial', 'seed': 42})
    print("Test 1 (Normal):")
    print(f"  Deployed: {result['deployed_arm']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Confidence: {result['jev_response']['confidence']:.3f}")
    assert result['reason'] in ['normal_selection', 'low_confidence_fallback']
    print("  ✓ PASS\n")

    # Test 2: 种子崩溃
    arms_collapse = [
        {'name': 'z0', 'val_score': 0.669},
        {'name': 'apc-full', 'val_score': 0.133}
    ]

    result = gate.select(arms_collapse, {'task': 'financial', 'seed': 44})
    print("Test 2 (Collapse):")
    print(f"  Deployed: {result['deployed_arm']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Collapse Noul: {result['jev_response']['collapse_noul']:.3f}")
    assert result['deployed_arm'] == 'z0'
    assert result['reason'] == 'collapse_detected'
    print("  ✓ PASS\n")

    # Test 3: 小样本（低 alignment）
    arms_small = [
        {'name': 'z0', 'val_score': 0.75, 'val_total': 4},
        {'name': 'manual', 'val_score': 0.50, 'val_total': 4}
    ]

    result = gate.select(arms_small, {'task': 'math', 'seed': 42})
    print("Test 3 (Small Val Set):")
    print(f"  Deployed: {result['deployed_arm']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Alignment: {result['jev_response']['alignment_score']:.3f}")
    # 小样本应该触发 low_val_alignment
    assert result['deployed_arm'] == 'z0'
    print("  ✓ PASS\n")

    print("All tests passed! ✓")
