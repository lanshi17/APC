#!/usr/bin/env python3
"""准备 APC 完整数据集用于 Jev + RSI 实验

从 experiments/apcbench/ 提取所有任务的 val 和 holdout 分数，
整理成统一格式供后续实验使用。

输出: experiments/apc_full_dataset.json
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def extract_arms_data(experiment: Dict, task_name: str) -> List[Dict]:
    """从实验文件中提取臂数据"""
    arms = []

    # APC 实验文件格式: {'meta': {...}, 'rows': [...]}
    if 'rows' in experiment:
        for row in experiment['rows']:
            if isinstance(row, dict) and 'method' in row:
                arms.append({
                    'name': row['method'],
                    'val_score': row.get('validation_score', 0.0),
                    'holdout_score': row.get('holdout_score', 0.0),
                    'seed': row.get('seed', 42),
                    'task': row.get('task', task_name),
                    'baseline_score': row.get('baseline_score', 0.0)
                })

    return arms


def prepare_full_dataset(experiments_dir: Path) -> Dict[str, Any]:
    """准备完整数据集"""
    dataset = {
        'metadata': {
            'version': '1.0',
            'description': 'APC Benchmark - Full Dataset for Jev + RSI Experiments',
            'num_tasks': 0,
            'num_arms': 0
        },
        'tasks': {}
    }

    # 扫描所有 real_*.json 文件
    real_files = list(experiments_dir.glob('real_*.json'))

    print(f"找到 {len(real_files)} 个实验文件\n")

    for filepath in sorted(real_files):
        task_name = filepath.stem.replace('real_', '')

        try:
            with open(filepath) as f:
                experiment = json.load(f)

            arms = extract_arms_data(experiment, task_name)

            if not arms:
                print(f"⚠️  跳过 {task_name}: 无有效臂数据")
                continue

            # 计算 oracle（holdout 最高分）
            oracle_arm = max(arms, key=lambda x: x['holdout_score'])

            # 按种子分组（如果有多个种子）
            seeds = set(arm['seed'] for arm in arms)

            for seed in seeds:
                seed_arms = [a for a in arms if a['seed'] == seed]
                task_key = f"{task_name}-s{seed}"

                oracle_arm_seed = max(seed_arms, key=lambda x: x['holdout_score'])

                dataset['tasks'][task_key] = {
                    'task': task_name,
                    'seed': seed,
                    'source_file': filepath.name,
                    'arms': seed_arms,
                    'oracle': {
                        'best_arm': oracle_arm_seed['name'],
                        'holdout_score': oracle_arm_seed['holdout_score']
                    },
                    'num_arms': len(seed_arms)
                }

                print(f"✓ {task_key:40s} {len(seed_arms):2d} 臂, oracle = {oracle_arm_seed['name']:15s} ({oracle_arm_seed['holdout_score']:.4f})")

        except Exception as e:
            print(f"❌ 处理 {task_name} 失败: {e}")

    # 更新元数据
    dataset['metadata']['num_tasks'] = len(dataset['tasks'])
    dataset['metadata']['num_arms'] = sum(t['num_arms'] for t in dataset['tasks'].values())

    return dataset


def main():
    """主函数"""
    repo_root = Path(__file__).parent.parent
    experiments_dir = repo_root / 'experiments' / 'apcbench'
    output_file = repo_root / 'experiments' / 'apc_full_dataset.json'

    print("="*70)
    print("准备 APC 完整数据集")
    print("="*70)
    print()

    if not experiments_dir.exists():
        print(f"❌ 实验目录不存在: {experiments_dir}")
        return

    # 准备数据集
    dataset = prepare_full_dataset(experiments_dir)

    # 保存
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print()
    print("="*70)
    print("完成！")
    print("="*70)
    print(f"任务数: {dataset['metadata']['num_tasks']}")
    print(f"总臂数: {dataset['metadata']['num_arms']}")
    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    main()
