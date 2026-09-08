# 2026 前沿：prompt 编译 / 跨模型迁移 / 约束感知优化（arXiv 实查 2026-09-08）

来源：`export.arxiv.org/api/query`（sortBy=submittedDate，人工筛选相关条目）。
每篇给出：核心 claim → 与 APC 的关系（呼应/对峙/必须引用）。

## 1. Instruction Stacking Collapse — 编译价值的 capability-dependence
[2608.02639] *A Benchmark and the Capability-Dependent Value of Prompt Compilation*（2026-08）
- 24 条 verifier-checked 指令逐条堆叠（1→20），三个 production-tier 模型（Claude Sonnet 4.6、
  GPT-5-mini、Gemini 2.5 Flash）follow-rate 从 ~96% 非线性崩到 ~20%；根因是可复现的
  **成对冲突**（单条 "output JSON" 与九条其它约束不相容）。training-free 补救：instruction
  compiler 重写指令集，其收益**依赖模型能力**。
- 与 APC：同题共振最强的一篇，必须引用并对峙——他们的"编译"是**自然语言重写**（消解指令
  冲突），我们的是**离散 genome 编译 + 搜索**；他们证明编译价值随 capability 变化，我们
  的 §3.4 F2 证明 profile 规则先验在真实强模型上是**净负债**（rule-root 落后 base-root
  0.10–0.85）。合成一句立场：**编译器该被验证而非被信任——先验必须能被搜索否决**。

## 2. PromptBridge — "Model Drifting" 与我们的 transfer-0 构成互补证据
[2512.01420] *Cross-Model Prompt Transfer for Large Language Models*（2025-12）
- 核心观察：为模型 A 工程的自然语言 prompt 直接用于模型 B，通常显著差于 B 上重优
  （命名为 Model Drifting），跨配置普遍且严重。
- 与 APC：他们量的是**跨模型**自然语言 prompt 损耗；我们 §3.4 F3 量的是**跨任务**结构
  genome 零损耗（transfer-0 = 最强臂）。两个方向合起来支持一个更大主张：**脆弱性在
  自然语言载体，不在结构表示**——结构化 genome 换任务不损耗，换模型才需画像适配
  （PGAM/迁移协议的存在理由）。必须引用。

## 3. DUALFIX — 演化"规则"而非"prompt"，与 PGAM 同族但对象相反
[2607.05121] *From Failing to Passing: Evolving Natural Language Prompt Optimization Rules
for LLM Code Generation*（2026-07）
- 搜索并演化一组**自然语言变换规则**（error-agnostic、跨问题复用），再叠加执行反馈修复。
- 与 APC：证明"规则可搜索"仍是 2026 活跃范式；我们的反面对照是 PGAM pooled +0.0004
  零结果 + F2 规则先验负债——立场：规则的载体应进 genome（可编译可审计），而不是
  留在自由文本里。

## 4. CAPO — 约束感知的多目标 prompt 优化
[2608.16068] *Constraint-Aware Prompt Optimization for LLM Agents*（2026-08）
- primal-dual + pool-based rewrites + 自适应约束权重，在 agentic 基准上更可靠地抵达可行域。
- 与 APC：把约束当一等公民与我们 TaskSpec/RuleBasedChecker 同路；差异——CAPO 连续权重
  对偶法，我们离散 genome 约束基因（output.strictness/forbid_extra_fields）。投稿时作为
  约束维度的最新基线候选。

## 5. Atlas (Compiled Memory) — "编译"叙事的近期盟友
[2603.15666] *Compiled Memory: Not More Information, but More Precise Instructions*（2026-03）
- 把任务经验经三步晋升门编译进 system prompt 的子项结构（无微调/RAG），CUAD 合同分析
  上提升显著。
- 与 APC：共享"编译为指令结构"的世界观与合同场景；差异——他们编译经验流，我们编译
  画像+规格。可用于 §1 论证"prompt 即工程对象"的社区趋势。

## 综合定位（写进论文 §6 的一句话）
2026 年三条独立证据链——堆叠崩溃（编译价值 capability-dependent）、Model Drifting
（自然语言载体脆弱）、规则演化回潮（DUALFIX）——都把问题推向同一处：**需要可验证、
可审计、可被搜索否决的中间表示**。APC 的 genome 正是该表示；而 APC §3.4 的真实 LLM
四臂矩阵给出的告诫是：**编译规则本身必须被当成假设而非真理来评测**。
