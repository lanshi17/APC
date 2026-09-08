# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts
### （工作草稿 v0.2：方法 + 四任务受控实验 + 诚实局限；非投稿版）

**摘要**：APC 把 prompt 从手工文本变成可编译的工程对象：任务规格（TaskSpec）→
离散基因组（PromptGenome，10 基因）→ 显式模型画像（ModelProfile，15 维）→
规则编译 → 进化搜索（含新算子 PGAM：画像先验 + 精英 bandit 变异）→ 模型迁移。
在自带确定性仿真器的四任务基准（APCBench：财务/合同/数学/约束遵循 × 3 模型）
上：搜索显著超越 zero-shot（+0.009 ~ +0.029）；进化在组合任务上超越随机搜索
（+0.003），在单位点任务上被随机搜索超越（−0.013，SHA 筛选噪声所致）；
PGAM 与均匀变异无差异（pooled +0.0004，零结果）；迁移以 30% 预算达到原生质量
（KR-6 12/12）。真实 LLM 第一阶段验证（§3.4，四臂 × 3 任务，单模型 qwen3.8-flash）：
结构 genome 搜索的 accuracy 增益 ≈ 0——**base-root 臂三任务全部守住 zero-shot
−0.004 下限；profile 规则先验是净负债**（root 落后 0.2–0.53，rule-root 臂在余量
最大的 financial 预算内只修复一半,−0.101）；跨任务 genome 零适配直用即达
zero-shot 水平（较冷启动重搜 +0.20~+0.79）。多模型与 PGAM 验证待更多凭证。

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

### 3.4 真实 LLM 验证（第一阶段：qwen3.8-flash × 3 任务，`bench_real_full/transfer.py`）

设定：模型白名单 key，reasoning 模型（~29 s/样本），temp=0；判分与仿真基准**同一套
rule-judge**；三任务完全同口径 dev_r/val/holdout=5/8/20；主对比预算 8、迁移实验预算 6。

| 任务 | zero-shot | manual | apc-full(rule-root) | apc-safe(base-root) |
|---|---|---|---|---|
| contract | 0.9770 | 0.9773 | **0.9782** | 0.9773 |
| math | 0.8436 | 0.8439 | **0.8442** | 0.8436 |
| financial | **0.6712** | 0.6662 | 0.5704 | 0.6672 |

- **F1 天花板效应**：强 reasoning 模型上,规则可验任务的 accuracy 余量 ≤0.005；
  prompt 优化的真实价值在格式/约束维度与弱初值救援（F2），而非 accuracy。
- **F2 规则先验是真实模型上的主要风险；base-root 搜索是"无损失下限"，rule-root
  搜索是"高成本救援"**：profile 编译 root 的 dev 分 contract 0.1778 / math 0.6335 /
  financial 0.1286，全部比 zero-shot 低 0.2–0.53。b8 搜索把 rule-root 臂拉回
  examples-on 盆地，但**修复完全度随任务余量递增而递减**：contract 0.9782>z0、
  math 0.8442≈z0、financial 0.5704 仍 −0.101；预算降到 6 连修复位点都采不到
  （cold 0.1850）。对照臂 **apc-safe（base root + 同预算 b8）三任务全部落在
  zero-shot 的 −0.004 内**（0.9773/0.8436/0.6672）：搜索在花掉同样预算后既没赚
  也没亏，financial 冠军是 base+`goal.explicitness: high→low`（val 选择噪声,
  holdout −0.004）。结论：**真实模型上 profile 规则先验是净负债；结构 genome
  搜索的 accuracy 增益 ≈ 0，其价值仅是救援坏先验（且不保证修满）**。仿真器中
  apc-full≡zero-shot 的"安全"在真实模型上不成立——不安全的来源正是编译规则。
- **F3 跨任务 genome 迁移（两个方向,b6 同预算三臂）**：

| 迁移 | cold(搜) | transfer-0(零适配直用) | transfer-ws(续搜) |
|---|---|---|---|
| contract→math | 0.6402 | **0.8439** | 0.8441 |
| math→contract | 0.1850 | **0.9778** | 0.9582 |

  最强臂是**零适配直用源冠军**（transfer-0），0 预算即达本任务 zero-shot 水平；
  warm-start 续搜在饱和任务上收益≈0（math +0.0002）甚至为负（contract −0.0196,
  val_n=8 选择噪声选中过拟合冠军）；冷启动因要先付 F2 的修复成本而显著落后。
  机理：genome 全结构、few-shot 内容由编译期从 TaskSpec 注入 ⇒ 跨任务零损耗;
  最优盆地唯一（两任务冠军同为 examples-on），与仿真器"单峰"发现互相印证。
  **工程含义：部署顺序 transfer-0 → base/apc-safe → rule-root+预算≥修复成本；
  绝不默认冷启动重搜**。
- **F4 判分器分歧抽检（financial 冠军 20 例，self-LLM-judge，`real_judge_check.py`）**：
  accuracy 维度 rule 均值 0.206 vs LLM 0.830（Pearson 0.48 / Spearman 0.56，20/20 分歧
  ≥0.25）；constraint 维度反向：rule 恒 1.0 vs LLM 0.50。解读：rule-judge 是金标准容差
  检查、系统性严于自评 LLM，且两个判分器各自的严苛维度不同。⇒ 本文所有真实 LLM 排序
  结论在同口径 rule-judge 下有效，但**绝对分不可读作“人类质量分”**；独立第三方 judge
  （非自评）仍是缺口。self-judge 的宽容度也可能含模型自偏好偏置。
- 诚实边界：单模型、单 seed、无 CI；多模型差异与 PGAM 的真实验证仍缺（凭证白名单）。

## 4. Limitations（投稿前必须解决）

1. **真实 LLM 验证为单模型单 seed**（§3.4）：qwen3.8-flash × 3 任务 + 跨任务迁移
   已完成；所用 key 为模型白名单，无法加第二模型。投稿需 ≥2 真实模型
   （1 开源 3 seeds + 1 闭源），脚本已就绪（`--model <id>` 配 `.env` 即可），
   blocker 是凭证不是代码。ground truth 仍由作者编写（rule-judge）。
2. **任务覆盖 4/6**：财务 + 合同 + 数学 + 约束遵循四任务，搜索≫zero-shot
   排序一致；仍缺真实推理基准（BBH/GSM8K）与开放式指令任务。
3. **PGAM 跨任务 pooled 零结果**：+0.0004 [−0.0029, 0.0038]（n=42）；
   画像先验的价值未被证实也不该被夸大。投稿级 claim 必须等待真实多峰任务。
4. **SHA 消融两面**：财务任务 full≡no-halving（零结果）；约束任务上
   SHA 代价 −0.0235（显著），而 rules 消融仍为零（base 已满足规则）。
   SHA 的取舍与任务结构有关，非普适加速器。
5. **Judge 可靠性（已量化,见 §3.4 F4）**：开放维度 rule-judge 与 self-LLM-judge
   系统性分歧（accuracy 0.21 vs 0.83），排序结论依赖同口径比较；独立第三方
   judge 与人工 spot-check（50–100 例）仍缺。

## 5. 可复现清单（本仓库现状）

- [x] 一键全量复现：`bench_apc.py [25|50|100]` + `bench_contract/math/follow/transfer.py`
  + `eval_robustness.py` + `analyze_bench.py`（确定种子，全 Mock，无网络）
- [x] 41 单元测试全绿；`.env` 零密钥（ggshield + 模式双检）
- [x] Java 服务 13 测试全绿（CompilerRules/PromptRenderer/MockClient；`mvn test`）
- [x] CI 顺序无关（每对比独立 crc32 种子；分析代码增删不改变已有 CI）
- [x] 数据集生成器 + 种子随仓（`scripts/gen_*_dataset.py`）
- [x] 真实 LLM 第一阶段：`bench_real_full.py`（3 任务 × 4 臂:z0/manual/apc-full/apc-safe,
  同口径 dev5/val8/hold20/b8）+ `bench_real_transfer.py`（双向迁移对）+ `analyze_real.py`
  （表格聚合）；结果入库 `experiments/apcbench/real_*.json`,冠军入库 `real_*_champ*.json`
- [x] LLM-Judge 一致性抽检：`real_judge_check.py`（self-judge 弱效度标注）⇒ §3.4 F4
  判分器分歧量化；结果入库 `experiments/apcbench/real_judge_agreement_financial.json`
- [ ] 第三方复现报告（待外部协作者）

## 6. Related Work（详见 `docs/literature/`）

优化器谱系：APE/OPRO/APO–ProTeGi → PromptBreeder/EvoPrompt → PromptWizard/
DSPy–MIPROv2 → GEPA（最强基线，投稿时必含）；画像与路由：HELM/FLASK/
FrugalGPT/RouteLLM；结构与迁移：Sclar 格式敏感、软提示（不可迁移）vs
离散基因组（可重编译）。
2026 编译/载体/约束前沿（详见 `docs/literature/lit-2026-frontier.md`）：
Instruction Stacking Collapse（2608.02639）证编译价值 capability-dependent——与本
文 F2"规则先验净负债"同轴对峙（NL 重写编译 vs genome 编译）；PromptBridge
（2512.01420）证自然语言载体跨模型漂移严重——与 F3 结构 genome 跨任务零损耗
互为补集：**脆弱性在载体不在表示**；DUALFIX（2607.05121）规则演化回潮 vs 本文
PGAM 零结果——规则应进 genome 而非自由文本；CAPO（2608.16068）约束感知优化，
投稿时约束维度的最新基线候选。
