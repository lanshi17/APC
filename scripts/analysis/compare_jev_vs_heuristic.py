#!/usr/bin/env python3
"""对比 Jev 门控 vs 原启发式门控

在现有 12 组历史数据上对比两种决策方法：
1. 启发式（auto_apc_gate.py 的 pick 函数）
2. Jev 决策模型（结构化决策 + 置信度）

目标：验证 Jev 至少达到 5/12 exact oracle（持平或超越启发式）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加 apc 模块到路径
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "apc-pipeline"))

from apc.decision.jev_gate import JevGate

BENCH = REPO / "experiments" / "apcbench"


def load_historical_data() -> List[tuple[str, List[Dict]]]:
    """加载现有 12 组数据（复用 auto_apc_gate.py 的逻辑）"""
    groups = []

    # Financial (3 seeds)
    fin = json.loads((BENCH / "real_financial.json").read_text())["rows"]
    for seed in (42, 43, 44):
        g = [r for r in fin if r.get("seed", 42) == seed and r["validation_score"] is not None]
        order = {"zero-shot": 0, "manual": 1, "apc-safe": 2, "apc-full": 3}
        g.sort(key=lambda r: order.get(r["method"], 9))
        if len(g) >= 2:
            groups.append((f"financial-s{seed}", g))

    # Math
    m = json.loads((BENCH / "real_math.json").read_text())["rows"]
    math_arms = [r for r in m if r["validation_score"] is not None]
    if math_arms:
        groups.append(("math-s42", math_arms))

    # Transfer experiments
    for f in sorted(BENCH.glob("real_transfer_*.json")):
        d = json.loads(f.read_text())
        rows = d["rows"] if isinstance(d, dict) else d
        rows = [r for r in rows if r.get("validation_score") is not None]
        order = {"transfer-ws": 0, "transfer-0": 1, "cold": 2}
        rows.sort(key=lambda r: order.get(r["method"], 9))
        if len(rows) >= 2:
            groups.append((f.stem.replace("real_", ""), rows))

    return groups


def heuristic_pick(arms: List[Dict], noise_band: float = 0.007) -> Dict:
    """原启发式决策（auto_apc_gate.py 的 pick 函数）"""
    best_score = max(a["validation_score"] for a in arms)
    candidates = [a for a in arms if a["validation_score"] >= best_score - noise_band]
    return candidates[0]  # 按简洁优先排序后的第一个


def main():
    print("=" * 80)
    print("对比实验：Jev vs 启发式门控")
    print("=" * 80)
    print()

    groups = load_historical_data()
    print(f"加载了 {len(groups)} 组历史数据\n")

    jev_gate = JevGate()

    results = []

    print(f"{'组名':28s} {'启发式':14s} {'Jev':14s} {'Oracle':8s} {'H-Regret':9s} {'J-Regret':9s}")
    print("-" * 95)

    for group_name, arms in groups:
        # 启发式决策
        h_choice = heuristic_pick(arms)
        h_deployed = h_choice['method']
        h_holdout = h_choice['holdout_score']

        # Jev 决策
        jev_arms = [
            {
                'name': a['method'],
                'val_score': a['validation_score'],
                'val_correct': int(a['validation_score'] * 8),
                'val_total': 8
            }
            for a in arms
        ]

        task = group_name.split('-')[0]
        seed = int(group_name.split('-s')[1]) if '-s' in group_name else 42

        j_result = jev_gate.select(jev_arms, {'task': task, 'seed': seed})
        j_deployed = j_result['deployed_arm']

        # 找到对应的 holdout 分数（处理名称不匹配）
        matched_arm = next((a for a in arms if a['method'] == j_deployed), None)

        if not matched_arm:
            # 如果没有精确匹配，尝试映射 z0 -> zero-shot
            if j_deployed == 'z0':
                matched_arm = next((a for a in arms if a['method'] == 'zero-shot'), None)

            if not matched_arm:
                # 仍然没找到，回退到第一个臂
                print(f"\n[WARN] Jev selected '{j_deployed}' not found in {[a['method'] for a in arms]}, using first arm")
                matched_arm = arms[0]
                j_deployed = matched_arm['method']

        j_holdout = matched_arm['holdout_score']

        # Oracle
        oracle = max(a['holdout_score'] for a in arms)

        # Regret
        h_regret = oracle - h_holdout
        j_regret = oracle - j_holdout

        results.append({
            'group': group_name,
            'heuristic_deployed': h_deployed,
            'heuristic_regret': h_regret,
            'jev_deployed': j_deployed,
            'jev_regret': j_regret,
            'jev_reason': j_result['reason'],
            'jev_confidence': j_result['jev_response']['confidence'],
            'improvement': h_regret - j_regret
        })

        print(f"{group_name:28s} {h_deployed:14s} {j_deployed:14s} {oracle:.4f}   "
              f"{h_regret:+.4f}    {j_regret:+.4f}")

    print("-" * 95)

    # 统计
    h_exact = sum(1 for r in results if abs(r['heuristic_regret']) < 0.0001)
    j_exact = sum(1 for r in results if abs(r['jev_regret']) < 0.0001)
    h_within_noise = sum(1 for r in results if r['heuristic_regret'] < 0.007)
    j_within_noise = sum(1 for r in results if r['jev_regret'] < 0.007)

    print()
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"总组数: {len(results)}")
    print()
    print(f"启发式:")
    print(f"  - Exact oracle:      {h_exact}/{len(results)}")
    print(f"  - Within noise band: {h_within_noise}/{len(results)}")
    print()
    print(f"Jev 决策:")
    print(f"  - Exact oracle:      {j_exact}/{len(results)}")
    print(f"  - Within noise band: {j_within_noise}/{len(results)}")
    print()

    # 决策原因分布
    reason_counts = {}
    for r in results:
        reason = r['jev_reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("Jev 决策原因分布:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  - {reason:30s}: {count:2d}")
    print()

    # 改进情况
    improvements = [r['improvement'] for r in results]
    avg_improvement = sum(improvements) / len(improvements)
    print(f"平均 regret 改进: {avg_improvement:+.4f}")
    print(f"  (正值 = Jev 更好，负值 = 启发式更好)")
    print()

    # 详细对比（只显示不一致的案例）
    disagreements = [r for r in results if r['heuristic_deployed'] != r['jev_deployed']]
    if disagreements:
        print("=" * 80)
        print(f"决策不一致的案例 ({len(disagreements)} 个)")
        print("=" * 80)
        for r in disagreements:
            print(f"\n{r['group']}:")
            print(f"  启发式: {r['heuristic_deployed']} (regret {r['heuristic_regret']:+.4f})")
            print(f"  Jev:    {r['jev_deployed']} (regret {r['jev_regret']:+.4f})")
            print(f"  原因:   {r['jev_reason']}")
            print(f"  置信度: {r['jev_confidence']:.3f}")
            print(f"  改进:   {r['improvement']:+.4f}")

    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)

    if j_exact >= h_exact:
        print(f"✓ Jev 达到或超越启发式 ({j_exact} vs {h_exact} exact oracle)")
        if j_exact > h_exact:
            print(f"  → Jev 额外修复了 {j_exact - h_exact} 个失败案例")
    else:
        print(f"✗ Jev 未达到启发式水平 ({j_exact} vs {h_exact} exact oracle)")
        print(f"  → 需要调整 meta_params 或改进决策逻辑")

    print()

    # 保存结果
    output_file = REPO / "experiments" / "jev_vs_heuristic_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'total_groups': len(results),
                'heuristic': {
                    'exact_oracle': h_exact,
                    'within_noise': h_within_noise
                },
                'jev': {
                    'exact_oracle': j_exact,
                    'within_noise': j_within_noise
                },
                'avg_improvement': avg_improvement,
                'reason_distribution': reason_counts
            }
        }, f, indent=2)

    print(f"详细结果已保存到: {output_file}")
    print()

    return 0 if j_exact >= h_exact else 1


if __name__ == "__main__":
    sys.exit(main())
