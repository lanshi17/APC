# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts
### （工作草稿 v0.2：方法 + 四任务受控实验 + 诚实局限；非投稿版）

**摘要**：APC 把 prompt 从手工文本变成可编译的工程对象：任务规格（TaskSpec）→
离散基因组（PromptGenome，10 基因）→ 显式模型画像（ModelProfile，15 维）→
规则编译 → 进化搜索（含新算子 PGAM：画像先验 + 精英 bandit 变异）→ 模型迁移。
在自带确定性仿真器的四任务基准（APCBench：财务/合同/数学/约束遵循 × 3 模型）
上：搜索显著超越 zero-shot（+0.009 ~ +0.029）；进化在组合任务上超越随机搜索
（+0.003），在单位点任务上被随机搜索超越（−0.013，SHA 筛选噪声所致）；
PGAM 与均匀变异无差异（pooled +0.0004，零结果）；迁移以 30% 预算达到原生质量
（KR-6 12/12）。全部结论限于仿真器；真实 LLM 验证待补充。

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

> - 任务 A「财务报告分析」（dev 40 / validation 30 / holdout 30 / perturbation 30）；
>   任务 B「合同信息抽取」、任务 C「数学应用题」、任务 D「可验证约束遵循」
>   （均为 dev 40 / validation 30 / holdout 30；生成器与种子见各任务脚本）。
>   B/C 共用 `_prompt_skill` 基因动力学（不同领域逻辑）；D 用任务内动力学
>   （约束/句数/数字三通道，激活此前中性的 instructions/constraints 位点）。
>   perturbation 设计（仅 A）：干扰句 + 首个指标值替换（金标准保持清洁版）。
> - 模型：glm / qwen / gpt（MockClient 确定性仿真；ground truth 见 README）。
> - 方法：zero-shot / manual / random-search / apc-full（均匀进化+SHA）/
>   apc-no-profile / apc-no-halving / apc-pgam。
> - 统计：A 任务 5 seeds × 3 模型，B/C/D 3 seeds × 3 模型；配对 bootstrap 95% CI。

### 3.1 主结果（b100，holdout）

| 对比 | Δ | 95% CI | 结论 |
|---|---|---|---|
| apc-full − zero-shot | +0.0093 | [0.0082, 0.0103] | 显著；搜索有效 |
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

### 3.2b 第二任务：合同抽取（b50，holdout，n=9/方法）

| 方法 | 均值 | 对比 | Δ | 95% CI |
|---|---|---|---|---|
| apc-full / apc-pgam | 0.9844 | search − zero-shot | +0.0275 | [0.0130, 0.0420]，显著 |
| manual | 0.9814 | search − manual | +0.0031 | [0.0000, 0.0061]，不显著 |
| zero-shot | 0.9569 |（qwen 单项 0.9300 → 搜索修复至 0.9833，弱模型救援 +0.05）| | |

### 3.2c 第三任务：数学应用题（b50，holdout，n=9/方法）

| 对比 | Δ | 95% CI | 结论 |
|---|---|---|---|
| search − zero-shot | +0.0285 | [0.0191, 0.0372] | 显著 |
| search − manual | −0.0109 | [−0.0154, −0.0063] | 显著；manual 强（精确 count/策略手工最优） |
| pgam − full | 0.0000 | — | 无差异 |

跨任务一致性：搜索 ≫ zero-shot 在四任务成立（+0.009 / +0.028 / +0.029 / +0.029）；
搜索 vs manual：财务 −0.001、合同 +0.003（n.s.）、数学 −0.011、约束遵循 +0.024（显著）——
手工强启发式在单峰仿真器上难被超越；自动化的价值在免手工+迁移+血统。

### 3.2d 第四任务：可验证约束遵循（b50，holdout，n=9/方法）

| 对比 | Δ | 95% CI | 结论 |
|---|---|---|---|
| search − zero-shot | +0.0292 | [0.0171, 0.0399] | 显著 |
| search − manual | +0.0238 | [0.0116, 0.0341] | 显著；搜索首次显著超越 manual |
| random − evolution | +0.0132 | [0.0035, 0.0246] | 显著；方向反转（见下） |
| pgam − full | +0.0005 | [−0.0148, 0.0162] | 不显著 |

**优化器交叉分解**（任务 D 新增 no-halving 对照后）：
no-halving（0.9616）> random（0.9513）> SHA 进化（0.938）。
no-halving − random：+0.0103 [0.0022, 0.0205]（显著）——选择准确时，
多步进化的组合价值为正；
full − no-halving：−0.0235 [−0.0357, −0.0112]（显著）——SHA 的 dev_r1
小样本筛选在本任务上代价巨大。结论：**速度-精度权衡可量化、无免费午餐**；
SHA 利于组合困难任务（财务），全量选择利于单位点任务（约束遵循）。
PGAM 在此任务同样与 uniform 无差异（+0.0005）。

### 3.2e PGAM 跨任务 pooled（配对差合并，n=42）
pgam − uniform：+0.0004 [−0.0029, 0.0038]——**零结果**（CI 收窄后仍过零）。
画像先验在本仿真器上不带来可测量的增益（画像本身信息量弱：
三模型仅 3/15 维有差异；且最优盆地单一）。
保留 PGAM 的理由：方差更低（b100 std 0.0001 vs 0.0016）、先验可解释、
bandit 下限保证不更差；确认/证伪需真实多峰任务。

### 3.2f 稳健性（perturbation，冠军 genome，n=15/方法）

全方法 holdout→perturbation 下降 ≈ −0.010（与 genome 无关的数值替换偏移）；
无证据表明搜索冠军更脆弱（排序与 holdout 一致）。

### 3.3 迁移（4 方向 × 3 seeds，adapt 预算 30 vs native 100）

- recover（adapted/native）：1.0000 [0.9976, 1.0025]；KR-6 通过率 12/12；
  adapt−direct：+0.0001（直接迁移基本无损——异质性在优化器分辨率之下）。
- 结论：迁移协议以 **30% 预算达到原生重搜质量**（GEPA 式 rollout 论证），
  附带血统与 adopt/keep 决策审计链；衰减矩阵本身平坦是本仿真器的局限。

## 4. Limitations（投稿前必须解决）

1. **无真实 LLM 验证**：全部结论限于自带仿真器；ground truth 由作者编写，
   存在"拟合自己仿真器"的根本性质疑。`scripts/bench_real.py`
   烟囱已备好（同口径 2 样本链路，无 key 时明确 SKIP）；
   投稿前必须在 ≥2 真实模型（1 开源 3 seeds + 1 闭源 1 seed）上复现主对比。
2. **任务覆盖 4/6**：财务 + 合同 + 数学 + 约束遵循四任务，搜索≫zero-shot
   排序一致；仍缺真实推理基准（BBH/GSM8K）与开放式指令任务。
3. **PGAM 跨任务 pooled 零结果**：+0.0004 [−0.0029, 0.0038]（n=42）；
   画像先验的价值未被证实也不该被夸大。投稿级 claim 必须等待真实多峰任务。
4. **SHA 消融两面**：财务任务 full≡no-halving（零结果）；约束任务上
   SHA 代价 −0.0235（显著），而 rules 消融仍为零（base 已满足规则）。
   SHA 的取舍与任务结构有关，非普适加速器。
5. **Judge 可靠性**：规则可验部分全代码判；开放维度仅规则 Judge，无 LLM
   Judge、无人工 spot-check（50–100 例）。

## 5. 可复现清单（本仓库现状）

- [x] 一键全量复现：`bench_apc.py [25|50|100]` + `bench_contract/math/follow/transfer.py`
  + `eval_robustness.py` + `analyze_bench.py`（确定种子，全 Mock，无网络）
- [x] 均值从零重跑逐值一致（本轮验证：全部方法均值与提交版逐值相等）
- [x] CI 顺序无关（每对比独立 crc32 种子；分析代码增删不改变已有 CI）
- [x] 数据集生成器 + 种子随仓（`scripts/gen_*_dataset.py`）
- [x] 全部冠军 genome 入库（bench 行含 `champion_genome`，扰动复评可重做）
- [x] 40 单元测试全绿；`.env` 零密钥（ggshield + 模式双检）
- [ ] 真实 LLM 复现（`scripts/bench_real.py` 待 key）
- [ ] 第三方复现报告（待外部协作者）

## 6. Related Work（详见 `docs/literature/`）

优化器谱系：APE/OPRO/APO–ProTeGi → PromptBreeder/EvoPrompt → PromptWizard/
DSPy–MIPROv2 → GEPA（最强基线，投稿时必含）；画像与路由：HELM/FLASK/
FrugalGPT/RouteLLM；结构与迁移：Sclar 格式敏感、软提示（不可迁移）vs
离散基因组（可重编译）。
