# Week 2 实施计划：完整实验设计

**Status:** in-progress
**Created:** 2026-10-02

**日期**: 2026-10-02  
**目标**: 完成 Agent 论文 7 步实验范式的前 4 步

---

## 📋 本周任务清单

### Day 1-2: Setup & Main Results
- [x] ✅ Jev Client 实现（heuristic 模式）
- [x] ✅ JevGate 决策门控
- [x] ✅ 基线验证（3/6 exact oracle）
- [ ] 📝 准备完整数据集（12 组实验）
- [ ] 📊 Main Results Table 1-2
- [ ] 🔍 Qualitative Case Studies（2-3 个）

### Day 3-5: Ablation Studies
- [ ] 🧪 Ablation 1: 无崩溃检测
- [ ] 🧪 Ablation 2: 无置信度门控
- [ ] 🧪 Ablation 3: 无对齐度评分
- [ ] 🧪 Ablation 4: 纯随机选择
- [ ] 🧪 Ablation 5: 纯最高分选择

### Day 6-7: Error Analysis & Efficiency
- [ ] 📈 失败模式分类（val 高估 / 崩溃漏检 / 过度保守）
- [ ] 💰 成本分析（API 调用次数 vs 性能提升）
- [ ] ⚡ Pareto Front（性能 vs 成本）

---

## 📂 需要的脚本文件

### 1. 数据准备脚本
```
scripts/
├── prepare_full_dataset.py      # 整理全部 12 组数据
├── compute_oracle_bounds.py     # 计算理论上界
└── analyze_headroom.py          # Headroom 分类
```

### 2. 主实验脚本
```
scripts/
├── run_main_experiments.py      # Table 1-2 生成
├── generate_case_studies.py     # Qualitative 分析
└── export_latex_tables.py       # 导出 LaTeX
```

### 3. 消融实验脚本
```
scripts/ablations/
├── ablation_no_collapse.py      # 去掉崩溃检测
├── ablation_no_confidence.py    # 去掉置信度门控
├── ablation_no_alignment.py     # 去掉对齐度评分
├── ablation_random.py           # 随机基线
└── ablation_greedy.py           # 贪婪基线
```

### 4. 分析脚本
```
scripts/analysis/
├── error_analysis.py            # 失败模式分类
├── efficiency_analysis.py       # 成本-效益分析
└── generate_pareto_plot.py      # Pareto 曲线
```

---

## 📊 预期输出

### Table 1: Main Results
| Method | Exact Oracle | Within Noise | Avg Regret | Collapse Detected |
|--------|--------------|--------------|------------|-------------------|
| Random | 1.2/12 | 3.5/12 | +0.045 | 0/2 |
| Greedy (max score) | 4.8/12 | 8.2/12 | +0.012 | 0/2 |
| Heuristic | 5.0/12 | 10.0/12 | +0.008 | 2/2 |
| **JevGate (Ours)** | **6.5/12** | **11.2/12** | **+0.004** | **2/2** |

### Table 2: Ablation Studies
| Variant | Exact Oracle | Avg Regret | Notes |
|---------|--------------|------------|-------|
| Full JevGate | 6.5/12 | +0.004 | - |
| - No Collapse | 5.2/12 | +0.018 | 漏检 financial-s44 |
| - No Confidence | 5.8/12 | +0.007 | 过度激进 |
| - No Alignment | 6.0/12 | +0.005 | 微小退化 |

### Figure 1: Case Study
```
财务任务 (financial-s44):
  Arms: z0 (0.669), apc-full (0.133)
  
  ❌ Greedy: 选择 z0 → 种子崩溃 → 实际 0.133
  ✅ JevGate: 
     1. Collapse Noul = 0.95 (HIGH)
     2. 检测到 0.133 远低于 0.669
     3. 选择 apc-safe (0.657)
     4. Regret = 0.001
```

---

## 🎯 本周成功标准

1. ✅ **完整数据集**：12 组实验数据整理完毕
2. ✅ **Main Results**：Table 1-2 生成，Jev 超越基线
3. ✅ **Ablation**：5 个消融实验完成，证明每个组件必要
4. ✅ **Case Studies**：2-3 个可视化案例
5. ✅ **成本分析**：Pareto 曲线生成

---

## 💰 预算估算

| 实验类型 | API 调用次数 | 成本 (@ $0.0193/1M tokens) |
|---------|-------------|---------------------------|
| Main Results | 0 (heuristic) | $0 |
| Ablations | 0 (heuristic) | $0 |
| Case Studies | 0 (分析) | $0 |
| **Week 2 Total** | **0** | **$0** |

**优势**：Heuristic 模式完全免费！

---

## 📝 Next Action

**立即执行**（你选择）：

**A. 开始 Day 1**：创建 `prepare_full_dataset.py`，整理全部数据  
**B. 跳到 Day 3**：假设数据已就绪，直接做消融实验  
**C. 先看数据**：检查现有的 12 组数据是否完整  

**请回复 A/B/C，我立即开始！** 🚀
