# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts
### （工作草稿 v0.1：方法 + 受控实验 + 诚实局限；非投稿版）

## 1. 问题与主张

同一业务任务在不同 LLM 上需要不同 prompt（Sclar et al. 2023：格式敏感且模型相关；
FLASK Ye et al. 2024：模型间技能短板不同）。现有评测画像（HELM/FLASK）只排名、
不指导构造；现有编译器（DSPy）按模型重搜、但画像隐含不可解释（详见
`docs/literature/lit-compiler.md` G1–G6）。

APC 的主张：**显式能力画像（15 维）作为编译器中间表示**——
`TaskSpec → PromptGenome（10 基因）→ ModelProfile → PromptCompiler（规则编译）
→ TrialResult → 进化搜索（PGAM）→ 模型迁移`，全程可追溯（KR-3）。

## 2. 方法

### 2.1 PromptGenome：离散结构基因组

10 类基因（role/goal/instructions/constraints/examples/reasoning/
verification/output/style/layout），`search_space` 声明可变异位点
（本实验 13 位点）。相对连续软提示（Lester 2021；SPoT）：离散基因组
可重编译、可跨模型携带（§5 迁移）。

### 2.2 Profile-Guided Adaptive Mutation（PGAM，新算子）

- **先验**：`w = 1 + 2(1 − capability)`（截断 [0.5, 3.0]），弱能力维度多探索；
  未映射位点权重 1.0（`profile_prior_weights`）。
- **在线修正**：每代精英幸存者的变异位点 ×1.3（上限 5.0），下限 0.3
  （ε-保证；画像失配时 bandit 可纠正——呼应 APO 的 bandit 剪枝思想）。
- 与均匀变异的唯一差别是位点采样分布；其余（SHA 调度、预算记账、
  血统记录）完全一致，故 ablation 是干净的单变量对比。

### 2.3 预算感知种群 + Successive Halving

`P = min(P₀, max(E+1, (B−3)/(G+2)))`：小预算自动收缩，保证至少完成一代；
每代 dev_r1 淘汰一半 + dev_full 复评。dev 选型 / validation 定冠军 /
holdout 只跑冠军一次（EvoPrompt 铁律）。

### 2.4 迁移协议（FR-7）

`CapabilityDelta（15 维差）→ MigrationMutator 调整 seed → 目标小预算重搜 →
holdout 对比决策（adopt 当且仅当 ≥90% 源分数，KR-6）`。

## 3. 实验（APCBench，`experiments/apcbench/`）

- 任务：财务报告分析（dev 40 / validation 30 / holdout 30 / perturbation 30，
  生成器 `scripts/gen_financial_dataset.py`，种子固定）。
- 模型：glm / qwen / gpt（MockClient 确定性仿真；ground truth 见
  `experiments/apcbench/README.md`）。
- 方法：zero-shot / manual（强人工启发式）/ random-search /
  apc-full（均匀进化+SHA）/ apc-no-profile / apc-no-halving / apc-pgam。
- 统计：5 seeds × 3 模型；配对 bootstrap 95% CI（2000 次）。

### 3.1 主结果（b100，holdout）

| 对比 | Δ | 95% CI | 结论 |
|---|---|---|---|
| apc-full − zero-shot | +0.0093 | [0.0082, 0.0102] | 显著；搜索有效 |
| apc-pgam − zero-shot | +0.0101 | [0.0095, 0.0107] | 显著 |
| evolution − random | +0.0030 | [0.0022, 0.0038] | 显著；多步组合价值 |
| apc-pgam − apc-full | +0.0008 | [0.0000, 0.0017] | **边缘**；方差 0.0001 vs 0.0016 |
| apc-full − manual | −0.0009 | [−0.0017, −0.0001] | 强人工启发式仍领先 |
| SHA / rules 消融 | 0.0000 | — | 零结果（base 已满足规则） |

### 3.2 预算曲线（holdout 均值）

| budget | pgam | full | random | manual | zero-shot |
|---|---|---|---|---|---|
| 25 | 0.7501 | 0.7495 | 0.7497 | 0.7595 | 0.7493 |
| 50 | 0.7581 | 0.7581 | 0.7581 | 0.7595 | 0.7493 |
| 100 | 0.7600 | 0.7592 | 0.7561 | 0.7600 | 0.7499 |

50 evals 即达平台（任务单峰性所致）；random 在 b100 反降（单步空间+候选增多→dev 过拟合）。

### 3.3 迁移（4 方向 × 3 seeds，adapt 预算 30 vs native 100）

- recover（adapted/native）：1.0000 [0.9978, 1.0026]；KR-6 通过率 12/12；
  adapt−direct：+0.0001（直接迁移基本无损——异质性在优化器分辨率之下）。
- 结论：迁移协议以 **30% 预算达到原生重搜质量**（GEPA 式 rollout 论证），
  附带血统与 adopt/keep 决策审计链；衰减矩阵本身平坦是本仿真器的局限。

## 4. Limitations（投稿前必须解决）

1. **无真实 LLM 验证**：全部结论限于自带仿真器；ground truth 由作者编写，
   存在"拟合自己仿真器"的根本性质疑。投稿前必须在 ≥2 真实模型（1 开源 3 seeds
   + 1 闭源 1 seed，EvoPrompt 双轨制）上复现主对比。
2. **单任务**：仅财务报告分析；至少补 BBH 子集 + IFEval 风格 + JSON 约束任务
   （见 `docs/literature/lit-benchmark.md` §7 最小矩阵）。
3. **PGAM 增益边缘**：+0.0008 且 CI 触零；需更难的多峰任务或更大 seed 数
   才能确认/证伪画像先验的价值。
4. **SHA 与 rules 消融为零结果**：分别需要紧预算多代场景与规则违背型 base
   才能测出差异；当前不断言。
5. **Judge 可靠性**：规则可验部分全代码判；开放维度仅规则 Judge，无 LLM
   Judge、无人工 spot-check（50–100 例）。

## 5. Related Work（详见 `docs/literature/`）

优化器谱系：APE/OPRO/APO–ProTeGi → PromptBreeder/EvoPrompt → PromptWizard/
DSPy–MIPROv2 → GEPA（最强基线，投稿时必含）；画像与路由：HELM/FLASK/
FrugalGPT/RouteLLM；结构与迁移：Sclar 格式敏感、软提示（不可迁移）vs
离散基因组（可重编译）。
