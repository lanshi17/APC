# Prompt 优化评测基准与方法论调研 (2023–2026)

> 目标读者: APC 框架顶会对齐。APC 现状对照: `apc/evaluation/` 已有 TrialScorer(accuracy/instruction_following/format/robustness/efficiency 加权合成)+checker(规则)+judge(LLM); `datasets/financial_analysis/` 已有 dev/validation/holdout 三划分。结论: 划分与双通道评分已对齐惯例,缺的是 seed/预算公平/跨模型矩阵。

## 1. 常用基准清单

| 基准 | 规模/形式 | 在 prompt 优化论文中的用法 | 来源 |
|---|---|---|---|
| BIG-Bench Hard (BBH) | 23 个推理任务 | 主战场: OPRO 相对人工 prompt 最高 +50%;EvoPrompt 最高 +25% | Suzgun et al. 2022, arXiv:2210.09261 |
| GSM8K | 8.5K 小学数学应用题 | OPRO +8% over human;MIPRO/DSPy 类数学 pipeline 必报 | Cobbe et al. 2021, arXiv:2110.14168 |
| BBH 子任务 | 从 23 任务中抽子集(如 EvoPrompt 用 7 个推理类) | 小预算论文的折中:全 BBH 太贵时报子集,但须声明抽样方式 | Guo et al.(EvoPrompt) 2023, arXiv:2309.08532 |
| IFEval | 25 种可验证指令 × ~500 prompts | 指令遵循能力的规则评分金标准 | Zhou et al. 2023, arXiv:2311.07911 |
| HELM | 16 核心场景 × 7 指标 | 引用其分类法:accuracy 之外必须报 robustness + efficiency | Liang et al. 2022, arXiv:2211.09110 |
| AIME-2025 | 竞赛数学 | GEPA 用它压 MIPROv2 +12% accuracy;2025–26 推理类新锚点 | Agrawal et al.(GEPA) 2025, arXiv:2507.19457 |
| MT-bench / Chatbot Arena | 多轮开放问答 + 人类偏好 | 只用于 judge 一致性验证,不做优化目标 | Zheng et al. 2023, arXiv:2306.05685 |

注:无"STRICT-JSON"标准基准之名;APC 应表述为 verifiable-constraint eval 而非引用不存在的基准。[INFERENCE: 命名核查结论]

## 2. 标准实验设置惯例

- **数据划分 (铁律: dev 选型 / test 上报)**: EvoPrompt §4.1 "pick the prompt with the highest score on the development set and report its score on the test set"——所有候选只能看 dev, test 只跑最终冠军一次。(EvoPrompt 2023)
- **预算报告**: 必须报告评估次数/rollouts,而非只报轮数。标杆 GEPA: "outperforms GRPO by 6% avg, up to 20%, using up to 35x fewer rollouts"——把预算效率变成第一-class 指标。(GEPA 2025; EvoPrompt 2023; OPRO arXiv:2309.03409)
- **少样本设置**: EvoPrompt §4.2 "prepend the demonstration consisting of one example per class"——跨方法对比时必须固定相同 demonstrations。(EvoPrompt 2023)
- **统计显著性 (诚实结论: 正式检验罕见,惯例是 multi-seed + std)**: EvoPrompt 开源模型报 3 seeds 均值±std,闭源因预算单 seed 并声明 "due to budget limitation"。Bootstrap/配对 t 检验几乎不用;若 APC 补上 bootstrap CI 反而是差异化加分。(EvoPrompt 2023 §4.1)
- **报告指标三件套**: (a) 绝对提升幅度 Δaccuracy(如 APO +31%);(b) 预算效率(分数/评估次数,GEPA 范式);(c) 跨任务×跨模型泛化矩阵。(Pryzant et al. 2023; GEPA 2025)

## 3. Ablation 与 Baseline 选取惯例

- **Baseline ladder (EvoPrompt Table 1 范式)**: Manual Instructions → 人类 prompt 仓 → 探索型自动方法(APE)→ 利用型编辑方法(APO/ProTeGi)。四层缺一不可:只比人工 prompt 会被质疑 strawman。
- **Optimizer 对 optimizer**: MIPRO 对 baseline optimizers(5/7 程序胜,最高 +13%);GEPA 同时对 RL(GRPO)和最强 prompt optimizer(MIPROv2, +10%)双线作战——新方法必须打败「上代最强同类」。
- **Ablation 粒度**: 以模块为单位整块移除。APC 对应: Profile 驱动规则开/关、基因逐类屏蔽、进化搜索 vs 单轮编译——至少 3 组 ablation。

## 4. LLM-as-judge 可靠性争议与缓解

- **争议点**: position bias、verbosity bias、self-enhancement bias、推理能力上限——GPT-4 级 judge 与人类偏好一致率 ~80%。(Zheng et al. 2023; Liu et al. G-Eval 2023)
- **缓解惯例**: 凡可验证的一律规则评分;jodge 只用于开放性维度;judge 与规则分同时上报并报告分歧率,关键结论抽 50–100 例人工 spot-check。

## 5. 顶会典型评审关注点 (NeurIPS/ICML/ICLR/ACL/EMNLP)

1. **可复现性**: 代码+全部最优 prompts+原始 completions 开源是标配。
2. **预算公平**: 对比方法必须等预算(等 trials/rollouts/LLM 调用次数);GEPA 把 "35x fewer rollouts" 写进标题级 claim 就是对此的回应。
3. **跨模型泛化**: 至少一开一闭(EvoPrompt: Alpaca-7b + GPT-3.5;DSPy: GPT-3.5 + llama2-13b)。只在一个模型上有效 = 过拟合。
4. **误差棒**: NeurIPS checklist 明确要求 error bars/multi-seed。
5. **测试集隔离**: dev 选型/test 上报,任何 test 反复查询都会被要求重做。

## 6. 2025–2026 最新趋势

- **Agentic prompt optimization**: 优化对象从单 prompt 扩展到 agent 系统。Meta Agent Search 证明搜出的 agent 可跨领域跨模型迁移。(Hu et al. ADAS 2024 arXiv:2408.08435)
- **Reflection > RL**: GEPA 用自然语言反思做 prompt 进化,6 任务平均超 GRPO 6%(最高 20%),rollouts 少 35 倍——高质量文本反馈比标量 reward 更适合 LLM 系统优化。(GEPA 2025)
- **Test-time scaling 与 prompt 优化的关系**: s1 预算强制把 AIME24 从 50% 推到 57%;sleep-time compute 把同等精度所需 test-time compute 降 ~5 倍。关系是**预算转移**:训练时搜索 vs 推理时计算。APC 编译器定位 = 把 test-time 成本前移到编译期摊销,应有一张"总成本(编译摊销+推理)vs 精度"Pareto 图。(s1 2025; sleep-time 2025)

## 7. APC 最小可信实验矩阵 (对齐顶会标准的下限)

| 维度 | 最小配置 | 说明 |
|---|---|---|
| 任务数 | ≥6: 推理×2、指令遵循×2、结构化输出×1、APC 自有领域×1 | 覆盖三类能力,每类 ≥1 可验证规则评分任务 |
| 模型数 | ≥2: 1 开源(3 seeds,均值±std)+ 1 闭源/大模型(1 seed,声明预算) | 照抄 EvoPrompt 的双轨制 |
| 基线数 | ≥4: 人工 prompt → APE/OPRO 类 → EvoPrompt/ProTeGi 类 → DSPy/MIPRO 类 + APC ablation ≥3 组 | 四层 ladder 缺一不可 |
| 预算公平表 | 所有方法统一报告总 LLM 调用次数/rollouts | GEPA 范式 |
| 评分 | 规则可验部分 100% 代码判;开放部分 judge + 分歧率 + 人工抽查 | §4 缓解措施 |
| 交付物 | 代码 + 全部最优 prompts/genomes + TrialResult 原始记录开源 | §5 可复现性 |

**一句话定位**: APC 现有 dev/validation/holdout + 规则/judge 双通道已踩中 2023–24 惯例;补齐「双模型矩阵 + 四层基线 + 预算公平表 + 开源产物」即达投稿下限,「bootstrap CI + 编译摊销 Pareto 图 + 跨模型迁移矩阵」是超出评审期待的加分项。
