# PRD — APC（Adaptive Prompt Compiler）v1.0

- **状态**：Draft
- **版本**：v1.0
- **产品负责人**：待指定
- **技术负责人**：待指定
- **目标发布阶段**：4 个迭代周期内完成可运行 MVP

---

## 1. 摘要

APC 是一个跨模型的 Prompt 编译、评测和优化系统。用户先定义业务任务，而不是手写一段固定 Prompt；系统再根据任务规格、模型能力画像和 PromptGenome，编译出模型可用的 Prompt，并保留完整实验记录。

v1.0 先服务于“财务报告分析”任务，接入 GLM、Qwen 和 GPT 三类模型。它要跑通从任务配置、模型探测、Prompt 编译、离线评测、搜索优化到模型迁移的完整闭环。

---

## 2. 联系人与职责

| 角色 | 负责人 | 职责 | 当前状态 |
|---|---|---|---|
| 产品负责人 | 待指定 | 确认任务价值、范围、验收指标和发布决策 | 待指定 |
| 业务专家 | 待指定 | 提供财报样本、金标准、风险判定规则 | 待指定 |
| Java 服务负责人 | 待指定 | 维护 Spring AI 业务服务、接口、存储与鉴权 | 待指定 |
| Python 管线负责人 | 待指定 | 维护 LangGraph 管线、评测、搜索和迁移 | 待指定 |
| 模型平台主管 | 待指定 | 管理 GLM/Qwen/GPT 凭证、模型版本、成本和限流 | 待指定 |
| 质量负责人 | 待指定 | 管理数据集版本、Holdout 集、评测可信度和发布门禁 | 待指定 |

**决策机制**：产品负责人确认业务指标；技术负责人确认接口、数据与工程质量；质量负责人确认评测门禁。任一项未满足时，不把候选 Genome 标为“最佳”或“可发布”。

---

## 3. 背景

### 3.1 问题

同一个业务任务在不同大模型上，常常需要不同的 Prompt 写法。模型升级、供应商替换或成本调整后，团队通常靠人工修改长文本 Prompt。这个做法有四个问题：

1. **不可迁移**：GLM 上有效的 Prompt 不一定在 Qwen 或 GPT 上有效。
2. **不可复现**：文本被手工改动后，很难知道效果变化来自哪里。
3. **不可比较**：没有统一探针、测试集和评分规则，无法判断“更好”是否真实。
4. **不可审计**：业务结论、模型输出、评测分数和成本记录分散，无法追溯。

### 3.2 为什么现在做

业务服务已使用 Spring AI，编排管线使用 LangGraph，能够以统一接口调用不同模型，并把探索流程做成可重复的图。系统已有 v1.0 项目骨架、财务报告分析 TaskSpec、基础 Genome、15 个模型探针、四类数据集与初版优化配置，具备实现 MVP 的基础。

### 3.3 产品原则

> 一等公民是 **TaskSpec**；二等公民是 **PromptGenome**；三等公民是 **ModelProfile**；最终 Prompt 只是编译产物。

不允许把手工编辑的一段 Prompt 当作唯一配置来源。每个 Prompt 必须可以追溯到其任务、Genome、模型画像、编译规则和实验记录。

---

## 4. 目标

### 4.1 产品目标

为固定业务任务提供一个可配置、可复现、可评测、可迁移和可追踪的 Prompt 优化闭环，使新模型接入时不需要从零人工重写 Prompt。

### 4.2 业务价值

| 对象 | 获得的价值 |
|---|---|
| 业务专家 | 通过 TaskSpec 表达任务、约束和结果标准，不必维护复杂 Prompt 文本。 |
| 提示词工程师 | 可以定位“任务、模型、Prompt 结构、样本”中哪一项影响分数。 |
| 模型平台团队 | 能比较 GLM/Qwen/GPT 的能力、质量、延迟与成本，并保留证据。 |
| 审计与质量团队 | 可以从一次输出追溯到输入数据、模型版本、配置和评分结果。 |

### 4.3 Key Results（首个稳定版本）

| 编号 | 可量化结果 | 验收方式 |
|---|---|---|
| KR-1 | 对同一个 TaskSpec，能为 GLM、Qwen、GPT 编译 Prompt。 | 三个模型各产生一份可保存的 `CompiledPrompt`。 |
| KR-2 | 完整运行 15 个 v1 Probe，并生成每个模型的 15 维能力画像和原始结果。 | Probe 结果和 ModelProfile 均落盘。 |
| KR-3 | 每次候选评测都记录任务、模型、Genome、Prompt、数据集版本、分数、Token 和延迟。 | 从任一 TrialResult 可向上追溯全部输入。 |
| KR-4 | 在至少 50 条带金标准样本上完成自动评测。 | 运行 dev/validation/holdout/perturbation 数据集，生成报告。 |
| KR-5 | 每个优化任务能在预算内评测至少 100 个候选 Genome。 | 默认优化配置 `budget=100`；超预算时停止新候选。 |
| KR-6 | GLM 的最佳 Genome 迁移到 Qwen 或 GPT 后，在相同 Holdout 集上达到源模型最佳分数的 ≥90%。 | 固定评分器、数据集版本和模型参数后比较分数。 |
| KR-7 | 对严格 JSON 任务，相对基线 Genome 的格式错误率下降 ≥50%。 | 用同一评测集比较 JSON 解析失败或 schema 不符合的比例。 |

### 4.4 非目标

v1.0 **不**包含：

- 在线用户反馈闭环或自动重训；
- 多租户、权限中心、计费系统或 Web Console；
- 自动生成业务数据集；
- 复杂贝叶斯优化、分布式 Ray 搜索或代理模型；
- 自动把模型内部思维过程暴露给用户；
- 对所有开放模型的通用适配承诺。

---

## 5. 目标市场与用户

### 5.1 目标用户

| 用户 | 要完成的工作 | 当前痛点 | 使用边界 |
|---|---|---|---|
| AI 业务开发者 | 把业务规则变为可靠模型调用 | Prompt 散落在代码里，模型切换成本高 | 能维护 YAML/JSON 配置和 CLI。 |
| 提示词工程师 | 选择 Prompt 结构并验证改动 | 靠主观试写，无法量化不同结构的作用 | 需要看评测报告、案例与成本。 |
| 业务分析师 | 让模型生成财报分析结论 | 输出可能虚构、漏掉风险或格式不稳定 | 只需维护 TaskSpec、样本和金标准。 |
| 模型平台工程师 | 接入或更换模型 | 无统一画像，难以判断模型是否适配任务 | 需要模型配置、Probe、成本和延迟数据。 |

### 5.2 首个业务场景：财务报告分析

输入为中文财务报告文本，单个文档最多 20,000 tokens。输出为严格 JSON，包含摘要、关键指标、风险和置信度。

业务约束：

1. 不得虚构输入中没有的数据。
2. 每个关键结论必须能追溯到输入材料。
3. 信息不足时必须明确说明。
4. 默认至少给出 3 个风险；确实不足时说明原因。
5. 最多 25,000 输入 tokens、2,000 输出 tokens、15 秒延迟。

### 5.3 进入条件

新任务进入 APC 前必须提供：

- 有版本号的 TaskSpec；
- 至少一个可运行的 PromptGenome；
- Dev、Validation、Holdout、Perturbation 四类数据集；
- 可运行的规则评分或独立 Judge；
- 业务负责人认可的质量权重与成本上限。

---

## 6. 价值主张

### 6.1 用户价值

| 用户问题 | APC 的做法 | 可观察结果 |
|---|---|---|
| “模型换了，Prompt 要重写。” | 用能力差异调整 Genome，并以目标模型重新编译和搜索。 | 迁移产生候选 Genome 与迁移报告。 |
| “这次改动为什么分数变好？” | 每次试验保存 parent genome、变异内容、Prompt、输出、评分和成本。 | 可比较 TrialResult 与候选差异。 |
| “模型看起来能输出 JSON，但经常混入解释。” | Probe 检测 JSON 可靠性；Compiler 增加 schema、禁用额外字段和格式自检。 | JSON 解析与 schema 分数可量化。 |
| “模型是否真的理解业务？” | 通过金标准、扰动集、规则评分和独立 Judge 分别测准确、约束、格式、稳健性。 | Holdout 和 Perturbation 分数可审计。 |

### 6.2 差异化

APC 不把 Prompt 当作静态文案库。它将 Prompt 分解成可变的 Genome，将模型行为量化为 Profile，并把最终 Prompt 作为可重建的编译结果。这避免了“每个模型维护一份长文本”的重复工作和失控风险。

---

## 7. 解决方案

### 7.1 用户流程

```mermaid
flowchart LR
    A[创建或更新 TaskSpec] --> B[选择基础 PromptGenome]
    B --> C[运行模型 Probe]
    C --> D[生成 ModelProfile]
    D --> E[编译模型专用 Prompt]
    E --> F[运行 Dev / Validation 评测]
    F --> G[搜索候选 Genome]
    G --> H[Holdout / Perturbation 门禁]
    H --> I[保存最佳 Genome 和 Trial]
    I --> J[迁移至新模型]
```

### 7.2 体验与接口

v1.0 是配置文件与 CLI 优先产品；服务端提供 REST API 供后续控制台或其他业务系统接入。

| 操作 | CLI | 服务端接口 | 输出 |
|---|---|---|---|
| 校验任务 | `apc task validate --config <task.yaml>` | 后续 `POST /api/tasks/validate` | 校验结果与字段错误。 |
| 运行探针 | `apc probe run --model <id> --suite v1` | 后续 `POST /api/probes/runs` | 原始模型响应和逐项评分。 |
| 构建画像 | `apc profile build --model <id>` | 后续 `POST /api/profiles` | ModelProfile JSON。 |
| 编译 Prompt | `apc prompt compile ...` | `POST /api/compile` | CompiledPrompt 与 token 估算。 |
| 运行评测 | `apc eval run ...` | 后续 `POST /api/evaluations` | TrialResult 和 case results。 |
| 优化 | `apc optimize run ...` | 后续 `POST /api/optimizations` | 最佳 Genome、分数和过程记录。 |
| 迁移 | `apc migrate run ...` | 后续 `POST /api/migrations` | CapabilityDelta、候选和目标模型分数。 |

### 7.3 核心功能需求

#### FR-1：TaskSpec 管理

系统必须：

- 读取 YAML 格式的 TaskSpec；
- 校验 `task_id`、名称、目标、输入、输出、约束、质量权重、成本上限和评测配置；
- 将 `output.schema` 作为严格结构化输出的唯一来源；
- 对任务和数据集保存版本；
- 拒绝缺少 Holdout 或 Perturbation 要求的高风险任务配置。

**验收**：`financial_analysis.yaml` 可校验；缺少 `task_id` 或输出 schema 的严格 JSON 任务返回明确错误。

#### FR-2：PromptGenome 与编译

Genome 必须支持：

- role、goal、instructions、constraints、examples、reasoning、verification、output、style、layout 十类基因；
- `search_space` 指定允许变异的值；
- XML、Markdown 或纯文本分段；
- 将输入以 `{{input}}` 占位符保留到运行时填充；
- 生成唯一 `prompt_id`、`genome_id`、`model_id`、模板版本、token 估算和元数据。

Compiler 必须依据 ModelProfile 调整：

| Profile 信号 | 编译动作 |
|---|---|
| few-shot benefit > 0.25 | 启用示例，至少 2 条。 |
| few-shot benefit < 0.08 | 禁用示例，避免无效 token 成本。 |
| JSON reliability < 0.90 | 强制 schema、禁止额外字段、提高格式严格度。 |
| JSON reliability > 0.96 | 可降低重复 schema 描述。 |
| reasoning > 0.90 且 self-verification benefit > 0.25 | 使用 hidden analysis，并启用验证。 |

**验收**：同一 Genome 对不同 Profile 生成的 Prompt 可不同；原始 Genome 不得在编译时被原地修改。

#### FR-3：模型适配

必须定义统一 `BaseModelClient`，至少支持：

- MockClient：无外部凭证时的可重复验证；
- OpenAI-compatible API：可用于 GPT、DashScope compatible-mode、Zhipu compatible API；
- Spring AI `ChatClient`：Java 业务服务中的统一调用入口。

每次调用必须记录：模型标识、模型版本、温度、输入 tokens、输出 tokens、延迟、重试次数和原始响应位置。凭证只从环境变量或受控密钥系统读取，绝不写入 TaskSpec、Genome、Probe、数据集或日志。

#### FR-4：Probe 与 ModelProfile

v1 探针集必须包含以下 15 个能力维度：

1. instruction following；2. multi-constraint following；3. long context；4. JSON reliability；5. schema strictness；6. table understanding；7. math；8. reasoning；9. information extraction；10. few-shot benefit；11. self-verification benefit；12. tool usage；13. robustness to distraction；14. Chinese semantic；15. safety boundary。

系统必须：

- 用 YAML 保存 probe、期望输出和评分指标；
- 以固定温度运行探针；
- 保存原始响应及每个指标的分数；
- 聚合为 `[0,1]` 的 CapabilityVector；
- 保存 BehaviorVector（冗长度、JSON 前缀噪声、拒答倾向、延迟、成本等）。

**验收**：`probes/v1/` 有 15 个可读 YAML；一次 suite 运行产生 15 条原始记录和一个 Profile 文件。

#### FR-5：评测

评测必须独立于 Compiler。每个样本至少输出：

- `accuracy`；
- `instruction_following`；
- `format_score`；
- `constraint_score`；
- `robustness`；
- `efficiency_score`；
- 综合 `score`；
- 输入/输出 token、平均延迟、方差和 case results。

评分规则：

```text
score = 0.50 × accuracy
      + 0.20 × instruction_following
      + 0.15 × format
      + 0.10 × robustness
      + 0.05 × efficiency
```

财务报告任务默认先做规则检查（JSON 可解析、字段完整、额外字段），再使用独立 Judge 模型评估准确、约束、忠实性和完整性。Judge 的模型不得与被评测模型相同，除非结果显式标注为“非独立 Judge”。

**验收**：无效 JSON 的 `format_score` 为 0；缺失或多余字段降低格式分；同一输入、固定模型参数和固定 Judge 时，结果可重复。

#### FR-6：优化

v1.0 使用 **Evolutionary Search + Rule-based Mutation + Successive Halving** 的轻量实现。默认配置：3 代、20 个候选、保留 5 个精英、预算 100 次。

允许变异的基因包括：

- 是否启用示例及其数量；
- 推理策略；
- 是否启用验证及验证类型；
- 输出严格度；
- 约束位置；
- 文风冗长度。

优化规则：

1. 只从 Genome 的 `search_space` 和受支持变异中生成候选。
2. Dev 集用于快速淘汰，Validation 集用于精英排序。
3. Holdout 集只用于最终候选与当前最佳候选的发布门禁。
4. Perturbation 集必须用于稳健性检查。
5. 达到预算、成本或延迟上限后停止，不再生成新调用。
6. 任何候选不得因仅优化 Dev 分数而覆盖 Holdout 最佳候选。

#### FR-7：模型迁移

迁移输入为源模型最佳 Genome、源 Profile、目标 Profile 和相同 TaskSpec。系统必须：

1. 计算 15 维能力差异 `CapabilityDelta`；
2. 用差异生成目标模型的 seed Genome；
3. 在目标模型上运行优化；
4. 比较源与目标的 Holdout 分数、格式错误率、成本与延迟；
5. 保存迁移报告和最终决策。

迁移示例：目标模型 few-shot benefit 更高时增加示例；JSON reliability 更低时提高格式约束并打开格式自检；schema strictness 更高时可减少重复自然语言格式说明。

### 7.4 数据、产物与可追踪性

| 类型 | 位置 | 最低保留内容 |
|---|---|---|
| 任务配置 | `configs/tasks/` | TaskSpec 原文、版本、校验结果。 |
| 模型配置 | `configs/models/` | provider、model、base URL、调用参数；不含 API key。 |
| PromptGenome | 配置与数据库 | Genome 原文、父 Genome、来源、变异说明。 |
| Probe | `probes/v1/` | 版本、输入、期望、评分指标、原始响应和得分。 |
| 数据集 | `datasets/<task>/` | dev/validation/holdout/perturbation、版本、样本 ID。 |
| 编译 Prompt | `artifacts/prompts/` | prompt text、模型、Genome、模板版本、token estimate。 |
| 输出 | `artifacts/outputs/` | 每个样本的模型原始输出、调用信息。 |
| 评测 | `artifacts/evaluations/` | 指标、Judge 结果、TrialResult、数据集版本。 |
| 画像 | `artifacts/profiles/` | CapabilityVector、BehaviorVector、Probe 原始结果。 |

持久化层开发期使用 SQLite；生产可迁移到 PostgreSQL。数据库至少需要 TaskSpec、Model、Genome、CompiledPrompt、Trial 五类记录。

### 7.5 技术边界

| 模块 | 技术 | 职责 |
|---|---|---|
| `apc-service` | Java 17、Spring Boot 3.3、Spring AI 1.0、JPA | 对外 API、模型客户端抽象、Prompt 编译、存储。 |
| `apc-pipeline` | Python 3.11+、LangGraph、Pydantic | Probe、Profile、评测、优化、迁移和 CLI。 |
| `configs/` | YAML / JSON | 可审查的业务与实验配置。 |
| `artifacts/` | JSON/JSONL/文本 | 不可丢失的实验产物和审计证据。 |

服务和管线的边界：Java 服务负责业务系统调用、编译和数据访问；Python 管线负责耗时实验编排。双方交换的对象必须与 TaskSpec、PromptGenome、ModelProfile、CompiledPrompt、TrialResult 的 JSON 结构兼容。

### 7.6 风险与假设

| 假设 / 风险 | 影响 | 应对 |
|---|---|---|
| 金标准样本不够或标签质量低 | 优化会得到错误方向。 | 发布前由业务专家抽样复核；Holdout 必须与 Dev 隔离。 |
| Judge 偏向某个模型或输出风格 | 分数虚高或虚低。 | 使用独立 Judge；保留规则分与 Judge 分；定期抽样人工复审。 |
| 只优化 Dev 集 | 过拟合且线上效果不稳定。 | 强制 Validation、Holdout、Perturbation 分层门禁。 |
| 模型 API 行为或版本静默变化 | 历史结果无法比较。 | 每次运行记录 provider/model/version/参数/时间。 |
| 隐藏推理提示不被模型支持 | 质量无法提升，且可能增加成本。 | 将 reasoning 作为可搜索基因；以实际评测决定是否启用。 |
| 探针集覆盖不足 | Profile 无法代表真实业务能力。 | v1 先固定 15 维；按真实失败案例增补 v1.x Probe。 |
| 输入含敏感财务信息 | 可能有合规和泄露风险。 | 数据集脱敏；凭证隔离；按环境设置保留期与访问控制。 |

---

## 8. 发布计划

### 8.1 v1.0（首个可运行 MVP）

**范围**：

- 一个业务任务：财务报告分析；
- 三类模型配置：GLM、Qwen、GPT；
- 一套基础 PromptGenome；
- 15 个 v1 Probe；
- MockClient 和 OpenAI-compatible 调用通路；
- Spring AI 业务服务和 LangGraph 管线；
- Rule-based Checker、独立 Judge 接口、TrialResult；
- 轻量遗传搜索和基础迁移；
- SQLite 记录与 JSON/JSONL 产物落盘；
- CLI 与基础 REST API。

**发布门禁**：

1. Java 服务可构建并通过测试；
2. Python 管线可运行 StateGraph；
3. TaskSpec 校验、Prompt 编译、探针、评测、优化、迁移均有一次端到端结果；
4. 所有产物可从 TrialResult 追溯；
5. 财报任务的严格 JSON 输出有可量化格式错误基线；
6. 凭证不进入 Git、日志或 artifact。

### 8.2 发布后立即跟进（v1.1）

- 完善真实 LLM Judge、样本执行器与成本记录；
- 支持至少 50 条经过复核的财报样本；
- 对 100 次候选搜索实施预算与 Successive Halving；
- 生成 HTML/Markdown 评测报告；
- 增加数据集、模型和 Genome 版本指纹；
- 增加 PostgreSQL 生产配置和 API 鉴权。

### 8.3 后续版本（v2+）

- 模型相似度和更精细迁移策略；
- 多模型联合优化与 Universal Prompt；
- 自动化 Judge 校准和人工复核工作流；
- 在线 A/B 测试与反馈闭环；
- Web Console、多租户与权限管理；
- 更大规模的代理模型和分布式优化。

### 8.4 不可发布情形

以下任一项成立时，禁止把候选标为可发布：

- 没有独立 Holdout 结果；
- JSON 严格任务存在未解释的格式退化；
- 迁移结果只在 Dev 集好于基线；
- 试验缺少模型版本、数据集版本、Genome 或 Prompt 记录；
- 达到成本/延迟上限却未记录；
- API Key、原始敏感数据或不应保留的推理内容出现在产物中。
