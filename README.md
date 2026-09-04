# APC — Adaptive Prompt Compiler v1.0

> **一等公民是 TaskSpec / 二等公民是 PromptGenome / 三等公民是 ModelProfile / 四等公民才是 Prompt 文本**

可配置 · 可复现 · 可评测 · 可迁移 · 可追踪 的跨模型 Prompt 编译与优化系统。

```
TaskSpec → PromptGenome → ModelProfile → PromptCompiler → ModelClient → RuleChecker+Judge → TrialResult → EvolutionaryOptimizer → Migration
```

## 架构

| 层 | 技术栈 | 职责 |
|---|---|---|
| `apc-service` | Java 17 + Spring Boot 3.3 + **Spring AI 1.0** | 业务服务：Task/Genome/Profile/Prompt/Trial 的 CRUD、编译服务、迁移服务、统一 ModelClient 抽象 |
| `apc-pipeline` | Python 3.11+ + **LangGraph** + Pydantic | 管线：Probe运行 → 画像构建 → 评测 → 搜索优化 → 迁移，StateGraph 编排 |

```
APC/
├── apc-service/          # Spring AI 业务服务
├── apc-pipeline/         # LangGraph Python 管线
├── configs/              # Task / Model / Optimizer 配置
├── probes/v1/            # 15 个模型探针 YAML
├── datasets/             # dev / validation / holdout / perturbation
├── artifacts/            # 编译产物、输出、评测、画像
└── tests/
```

## 快速开始

### 1. 启动业务服务

```bash
cd apc-service
mvn spring-boot:run
# http://localhost:8080/api
# Swagger: http://localhost:8080/swagger-ui.html
```

### 2. 安装 Pipeline

```bash
cd apc-pipeline
pip install -e ".[dev]"
# or: uv sync

# 校验任务
apc task validate --config configs/tasks/financial_analysis.yaml

# 运行探针
apc probe run --model glm --suite v1

# 构建画像
apc profile build --model glm

# 编译 Prompt
apc prompt compile --task configs/tasks/financial_analysis.yaml --model glm --genome configs/genomes/base.json

# 评测
apc eval run --task financial_report_analysis_v1 --model glm --prompt artifacts/prompts/xxx.txt --dataset datasets/financial_analysis/dev.jsonl

# 优化
apc optimize run --task financial_report_analysis_v1 --model glm --budget 100

# 迁移 GLM → Qwen
apc migrate run --task financial_report_analysis_v1 --source-model glm --target-model qwen
```

### 3. LangGraph 可视化

```bash
cd apc-pipeline
python -m apc.graph --visualize  # 输出 Mermaid 图
```

## 核心概念

- **TaskSpec** — 任务规格（输入/输出/约束/推理/权重/成本/评测）
- **PromptGenome** — 模型无关的 Prompt 基因组（role/goal/instructions/constraints/examples/reasoning/verification/output/style/layout）
- **ModelProfile** — CapabilityVector(15维) + BehaviorVector(7维)，由 Probe Set 探测生成
- **CompiledPrompt** — Genome + TaskSpec + ModelProfile → PromptText 的编译产物
- **TrialResult** — 一次评测的完整分数与开销

## 技术选型

- Java: Spring Boot 3.3.5, Spring AI 1.0.0, SQLite/PostgreSQL (JPA), MapStruct, Springdoc
- Python: pydantic>=2, pyyaml, sqlalchemy, typer, rich, httpx, tenacity, numpy, pandas, scikit-learn, langgraph, langchain

## 工程验收

- [ ] TaskSpec 编译出 ≥3 模型的 Prompt
- [ ] 15 Probe 可运行并生成画像
- [ ] ≥50 样本自动评测
- [ ] ≥100 候选搜索
- [ ] 全量实验可追踪/可复现
- [ ] GLM → Qwen/GPT 迁移分数 ≥90%

## 许可证

MIT
