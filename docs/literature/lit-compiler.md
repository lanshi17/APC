# Lit-Compiler：模型感知 Prompt 构造 × 模型能力评测 × 路由 × 迁移性综述

> 范围：2022–2026 前沿文献，为 APC（TaskSpec→PromptGenome→ModelProfile→PromptCompiler→TrialResult→进化→迁移）定位顶会级创新点。
> 方法说明：以下来自训练知识；venue/年份置信度高，但成稿前应对 arXiv 号做一次核验。[INFERENCE = 推断/待核验]

---

## 1. 模型感知提示适配（per-model prompt adaptation）

### [1] DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines — Khattab et al., arXiv 2023 / ICLR 2024
- **核心**：首次把"prompt 工程"重命名为"compiling"：用户写声明式 signature + module，teleprompt 优化器（COPRO/BootstrapFewShot/MIPRO）为**某一具体 LM** 自动编译出 instruction + few-shot demo。换模型 = 重新 compile。
- **与 APC 关系**：这是"按模型编译 prompt"最接近的先例，也是 APC 用 compiler 类比的最大合法性来源。**差异**：DSPy 的编译由"在该模型上的优化搜索"驱动，没有显式的、可解释的模型能力画像（ModelProfile）作为中间表示；APC 的 PromptCompiler 是**画像驱动的规则编译**（profile → 规则 → prompt），编译决策可审计、可解释。
- 一句话定位：DSPy = optimize-then-freeze per LM；APC = probe → profile → rule-compile。

### [2] OPRO (Yang et al., arXiv 2023) / ProTeGi (Pryzant et al., arXiv 2023)
- 文本梯度/beam-search 式 prompt 优化，优化过程绑定在某个打分模型上，产出 prompt 隐含过拟合到该模型。
- **证据价值**：说明"为某模型搜出的 prompt ≠ 通用 prompt"已是默认假设，但二者都**没有把模型差异显式建模**——APC 的 ModelProfile 正好补上这一缺环。

### [3] TextGrad (Yuksekgonul et al., NeurIPS 2024)
- 用文本反馈做"自动微分"，可联合优化复合 AI 系统中的多个 prompt/组件。
- **与 APC 关系**：TextGrad 优化的是梯度链路上的变量；APC 可把 ModelProfile 视为不可微的先验约束（哪几维弱 → 编译时加固哪几段），二者正交、可组合。[INFERENCE]

---

## 2. LLM 能力画像 / 评测体系（capability profiling）

### [4] HELM — Liang et al., arXiv 2211.09110 (Stanford CRFM, 2022)
- **核心**：scenario（任务场景）× metric（accuracy/robustness/fairness/efficiency…）矩阵式评测，反对单一分数排名。
- **与 APC 关系**：HELM 矩阵是 **15 维 capability vector 最直接的类似物**——APC 的 15 probes ≈ 为"财报分析任务族"定制的窄域 HELM：行从通用场景收敛为任务相关能力，列收敛为 TrialResult 可用的通过率/错误率。引用时可写："task-scoped HELM"。

### [5] FLASK — Ye et al., ICLR 2024
- **核心**：12 个细粒度技能，按"模型 × 技能 × 指令难度"三维打分；发现不同模型技能短板不同。
- **与 APC 关系**：粒度（12 维）与 APC 15 维最接近，且已实证"模型间技能画像不同"——这正是 APC "先探测再编译"的前提假设的最强论据。**差异**：FLASK 的画像只用于评测排名，未回灌到 prompt 构造。

### [6] BIG-bench — Srivastava et al., TMLR 2023（200+ 任务，2022 发布）
- 以任务为单位的能力分解；涌现/分项能力差异的经典证据。
- 对 APC 的价值：任务级能力分解的合法性 + "能力不可由单一分数代表"的论证弹药。

### [7] Chatbot Arena — Zheng et al., arXiv 2023 / NeurIPS 2024（LMSYS）
- 机制：众包 pairwise 对战 + Bradley-Terry 打分，MT-Bench 做静态对照。
- 对 APC 的价值：**反例**——Arena 给出的是单维 Elo，不含能力分解，无法指导 prompt 构造。这反衬 ModelProfile 的必要性：routing 选模型（Arena 够用）vs compiling prompt（需要画像）。

### [8] Model Cards — Mitchell et al., FAccT 2019
- per-model 档案（用途、性能、局限）的制度性先例。ModelProfile 可表述为"面向编译器的、可执行的 Model Card"。

---

## 3. 模型路由 / 选择（routing, capability–cost tradeoff）

### [9] FrugalGPT — Chen et al., arXiv 2305.05176 (2023)
- 级联（cascade）+ 打分器：便宜模型先答，不确定才升级；给出**成本–质量 Pareto 曲线**的量化方法。
- 对 APC 的价值：能力–成本权衡的**量化范式模板**（Pareto frontier + estimator）；APC TrialResult 已记录 token/延迟，可直接复用该分析框架。**差异**：FrugalGPT 在"模型粒度"做选择，APC 在"prompt 粒度"做适配。

### [10] RouteLLM — Ong et al., arXiv 2406.18665 (2024)
- 用偏好数据训练 router（强/弱模型二选一），以小代价保质量。
- 对 APC 的价值：router 需要的正是"任务–模型匹配度"信号——ModelProfile 是比单次打分更稳定的路由特征。[INFERENCE：可列为未来工作]

### [11] RouterBench — Hu et al., arXiv 2024
- 多 LLM 路由系统的 benchmark，统一度量质量–成本–延迟。
- 对 APC 的价值：评测方法学参考；APC 的跨模型评测报告可对齐其指标口径。

---

## 4. Prompt 跨模型迁移性（transferability）

### [12] APE — Zhou et al., ICLR 2023
- 除自动生成 instruction 外，还做了**跨模型复用 instruction 的初步实验**，观察到可迁移但非无损。
- 对 APC 的价值：硬提示跨模型迁移问题意识的最早实证之一；但只是副实验，无系统性衰减矩阵。

### [13] 软提示迁移：Lester et al., EMNLP 2021 + SPoT — Vu et al., ACL 2022
- 结论方向：软提示**绑定具体模型权重**（跨模型基本不可迁移）。
- 对 APC 的价值：论证"基因组（离散结构）比连续提示更适合作为迁移载体"的理论支点：离散 genome 可重编译，连续 prompt 不能。[INFERENCE]

### [14] Sclar et al., arXiv 2305.14477 (2023, ACL 2024) — 格式敏感性
- 核心：分隔符、大小写、枚举格式等"无意义"格式改动可引起大幅性能波动。
- 对 APC 的价值：① 解释为何迁移会衰减；② 直接支持 APC 把"结构"做成 genome 基因而非固定文本。

---

## 5. 结构化 Prompt 工程

- **Role 扮演**：Kong et al., arXiv 2308.07702 (2023) Role-Play Prompting；Xu et al. (2023) ExpertPrompting——role 段有独立增益证据。
- **格式消融**：Sclar et al. [14] 是最系统的格式消融；结论偏向"格式影响大且模型相关"，即**结构选择应模型特定化**——APC 编译器按画像选结构的直接依据。
- **业界证据（非论文，引用时标注）**：Anthropic Claude 文档（2024）推荐 XML 标签分段；OpenAI 提示工程指南推荐 Markdown 层级。[INFERENCE/待核验]

---

## 6. 编译器类比

- **DSPy [1]**："compiling declarative calls"是 prompt-compilation 提法最强先例。
- **LMQL** (Beurer-Kellner et al., 2023) /**Guidance** (Microsoft)：把约束下推到解码器，相当于编译器的"后端代码生成"。[INFERENCE]
- **Tracr** — Lindner et al., NeurIPS 2023：把程序**编译进 Transformer 权重**。方向相反，但可借用其术语框架：前端（TaskSpec/Genome DSL）→ 中间表示（ModelProfile 感知的 IR）→ 后端（各模型方言的 prompt 文本）。[INFERENCE：叙事建议]

---

## 7. 三个重点问题回答

**Q1：「先探测模型能力 → 再据此编译 prompt」两阶段范式有无先例？最接近的是什么？**
- 无精确先例。最接近的三块拼图：① **DSPy**（按模型编译，但无显式画像）；② **FrugalGPT/RouteLLM**（先刻画模型再决策，但粒度是"选哪个模型"）；③ **FLASK/HELM**（有画像，但只用于排名）。
- APC 的新颖组合 = **显式画像（15 维）作为编译器 IR + 规则编译 + 闭环进化修正**。投稿叙事："bridging evaluation profiles and prompt construction"。

**Q2：15 维 capability vector 有无类似物？**
- 有。**HELM 的 scenario×metric 矩阵**（结构类似物）、**FLASK 的 12 技能集**（粒度类似物）。APC 的差异化：窄域、可执行（直接驱动编译规则）。

**Q3：跨模型迁移性能衰减有无量化研究？**
- 系统性量化**基本空白**。只有 APE 的附带实验、软提示旁证、Sclar 的格式脆弱性解释。APC 的 KR-6 本身就是可发表的测量贡献。

---

## 8. APC 编译器范式空白点小结

| # | 空白点 | 支撑证据 | APC 对应设计 |
|---|---|---|---|
| G1 | 画像与构造脱节 | [4][5] 只排名不回灌 | ModelProfile → PromptCompiler 规则 |
| G2 | 编译无显式 IR | [1] | 15 维显式向量 + 可追溯 TrialResult |
| G3 | 决策粒度错位 | [9][10] | 模型粒度路由 × prompt 粒度编译双层 |
| G4 | 迁移衰减无标准测量 | [12][13][14] 均为侧证 | KR-6 ≥90% + 迁移衰减矩阵 |
| G5 | 结构选择的系统消融缺失 | §5 | 结构基因 + TrialResult 归因 |
| G6 | 能力–成本–prompt 复杂度三维权衡无人做 | [9][11] | TrialResult 含 token/延迟，可画三维 Pareto |

**顶会叙事一句**：APC = task-scoped HELM/FLASK-style profiling wired into a DSPy-style compiler, with genome-level evolution and a first systematic cross-model transfer-decay measurement.
