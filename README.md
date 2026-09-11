<div align="center">

# APC — Adaptive Prompt Compiler

**跨模型的 Prompt 编译、评测与优化系统**

> 一等公民是 TaskSpec · 二等公民是 PromptGenome · 三等公民是 ModelProfile · 四等公民才是 Prompt 文本

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java-17-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.3-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)](apc-pipeline/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 为什么是 APC

同一个业务任务在不同大模型上需要不同的 Prompt 写法。模型升级、供应商替换或成本调整后，团队只能人工修改长文本 Prompt,导致:不可迁移、不可复现、不可比较、不可审计。

APC 把 Prompt 从"手工维护的静态文案"变成**可编译、可评测、可优化的工程对象**:

```
TaskSpec → PromptGenome → ModelProfile → PromptCompiler → ModelClient → RuleChecker+Judge → TrialResult → EvolutionaryOptimizer → Migration
```

| 问题 | APC 的做法 | 可观察结果 |
|---|---|---|
| 模型换了,Prompt 要重写 | 能力差异调整 Genome,目标模型重新编译+搜索 | 迁移产生候选 Genome 与迁移报告 |
| 改动为什么分数变好 | 每次试验保存 parent genome、变异、Prompt、评分、成本 | 可比较的 TrialResult |
| JSON 经常混入解释 | Probe 检测 JSON 可靠性;Compiler 强制 schema | 格式错误率可量化 |
| 模型是否真的理解业务 | 金标准 + 扰动集 + 规则分 + 独立 Judge | Holdout/Perturbation 分数可审计 |

## 架构

| 层 | 技术栈 | 职责 |
|---|---|---|
| [`apc-service`](apc-service/) | Java 17 + Spring Boot 3.3 + **Spring AI 1.0** | 业务服务:Task/Genome/Profile/Prompt/Trial 的 CRUD、编译服务、迁移服务、统一 ModelClient 抽象 |
| [`apc-pipeline`](apc-pipeline/) | Python 3.11+ + **LangGraph** + Pydantic | 管线:Probe 运行 → 画像构建 → 评测 → 搜索优化 → 迁移,StateGraph 编排 |

```
APC/
├── apc-service/          # Spring AI 业务服务(Java 17)
├── apc-pipeline/         # LangGraph Python 管线
│   └── apc/
│       ├── cli.py        # apc 命令行(typer)
│       ├── compiler/     # Genome→Prompt 编译器 + Profile 驱动规则
│       ├── core/         # TaskSpec / PromptGenome / ModelProfile / TrialResult
│       ├── evaluation/   # Runner + RuleBasedChecker + Judge + 数据集加载
│       ├── graph/        # LangGraph StateGraph 端到端编排
│       ├── models/       # MockClient / OpenAI-compatible 客户端 + 工厂
│       ├── optimizer/    # 进化搜索 + Successive Halving
│       ├── profiler/     # 15 维探针运行器 + 画像构建
│       ├── storage/      # SQLAlchemy 仓储 + SQLite
│       └── transfer/     # CapabilityDelta 模型迁移
├── configs/              # Task / Model / Genome / Optimizer 配置
│   ├── tasks/            # financial_analysis.yaml(财务报告分析)
│   ├── models/           # glm.yaml / qwen.yaml / gpt.yaml(不含密钥)
│   ├── genomes/          # base.json 基础基因组
│   └── optimizer/        # default.yaml(3 代 × 20 候选 × 预算 100)
├── probes/v1/            # 15 个模型能力探针 YAML
├── datasets/             # dev / validation / holdout / perturbation
├── artifacts/            # 编译产物、输出、评测、画像、迁移报告
├── tests/               # 集成测试入口(目录保留)
└── PRD-APC.md            # 产品需求文档(v1.0)
```

## 快速开始

### 0. 环境要求

- Java 17+(业务服务)、Maven
- Python 3.11+(管线)
- 无任何 LLM API Key 时自动回退 `MockClient`,全流程可离线复现

### 1. 配置凭证(可选)

```bash
cp .env.example .env   # 填入 OPENAI_API_KEY / DASHSCOPE_API_KEY / ZHIPUAI_API_KEY
```

`.env` 已被 gitignore。仓库 `.env` 是凭证唯一事实来源(import `apc.models.factory` 时以 `override=True`
加载,遮蔽 shell 残留旧值);字段优先级:`.env`/`APC_<MODEL_ID>_<FIELD>` 环境变量 > `configs/models/*.yaml` > 内置默认。
**任意自定义 OpenAI 兼容网关,无需注册供应商**:`APC_<新ID>_MODEL` + `APC_<新ID>_BASE_URL` +
`APC_<新ID>_API_KEY`(或 `API_KEY_ENV` 指向存 key 变量)三行即成新模型;
老配置可用 `api_key_env` 字段声明 key 变量名。详见 [.env.example](.env.example)。

### 2. 启动业务服务(Java)

```bash
cd apc-service
mvn spring-boot:run
# http://localhost:8080/api
# Swagger: http://localhost:8080/swagger-ui.html
```

### 3. 安装管线(Python)

```bash
cd apc-pipeline
pip install -e ".[dev]"        # 或 uv sync

# 校验任务
apc task validate --config configs/tasks/financial_analysis.yaml

# 运行探针 → 构建画像
apc probe run --model glm --suite v1
apc profile build --model glm

# 编译 Prompt
apc prompt compile --task configs/tasks/financial_analysis.yaml --model glm --genome configs/genomes/base.json

# 评测
apc eval run --task financial_report_analysis_v1 --model glm --prompt artifacts/prompts/xxx.txt --dataset datasets/financial_analysis/dev.jsonl

# 优化(3 代 × 20 候选 × 预算 100)
apc optimize run --task financial_report_analysis_v1 --model glm --budget 100

# 迁移 GLM → Qwen
apc migrate run --task financial_report_analysis_v1 --source-model glm --target-model qwen
```

> 注:`apc probe/profile/optimize/migrate` 默认走 Mock,加 `--no-mock` 走真实 API(需对应凭证)。

### 3b. 真实 LLM 基准(需 key)

```bash
# 主对比 4 臂:zero-shot / manual / apc-full(rule-root) / apc-safe(base-root),同判分口径
.venv/bin/python scripts/bench_real_full.py --task contract --model qwen
.venv/bin/python scripts/bench_real_full.py --task financial --methods apc-safe  # 单臂增量合并
# 跨任务 genome 迁移三臂:cold / transfer-0(零适配) / transfer-ws(续搜)
.venv/bin/python scripts/bench_real_transfer.py --source contract --target math --model qwen
# 扰动鲁棒性(4 genome × financial perturbation 集)与 LLM-Judge 一致性抽检
.venv/bin/python scripts/bench_real_robust.py --task financial --n 20
.venv/bin/python scripts/real_judge_check.py --task financial --genome champ
# 汇总成论文表格
.venv/bin/python scripts/analyze_real.py
```

结果入库 `experiments/apcbench/real_*.json`,冠军 genome 入库 `artifacts/optimizations/real_*_champ*.json`。
真实模型发现(F1–F11:饱和天花板、规则先验负债、GEPA/ESPO 官方基线对垒全平、token-starved
判别实验、HLE-exact 高难对垒与 headroom 类型学)见
[docs/paper-apc.md](docs/paper-apc.md) §3.4。完整复现序列(离线层 + 真实 API 层、断点续跑
与配对同日等坑位注记)见 [REPRODUCE.md](REPRODUCE.md)。

### 4. LangGraph 可视化

```bash
cd apc-pipeline
python -m apc.graph --visualize  # 输出 Mermaid 图
```

### 5. 测试

```bash
cd apc-pipeline
pytest tests/ -q     # 41 个测试,全 Mock,无网络
```

## 核心概念

| 概念 | 说明 |
|---|---|
| **TaskSpec** | 任务规格:输入/输出/约束/推理/权重/成本/评测,以 `{{input}}` 占位符运行时填充 |
| **PromptGenome** | 模型无关的 Prompt 基因组:role/goal/instructions/constraints/examples/reasoning/verification/output/style/layout 十类基因 |
| **ModelProfile** | CapabilityVector(15 维)+ BehaviorVector(7 维),由 Probe Set 探测生成 |
| **CompiledPrompt** | Genome + TaskSpec + ModelProfile → PromptText 的编译产物,含唯一 prompt_id 与 token 估算 |
| **TrialResult** | 一次评测的完整分数与开销,可向上追溯全部输入 |

## 编译规则(Profile 信号驱动)

| Profile 信号 | 编译动作 |
|---|---|
| few-shot benefit > 0.25 | 启用示例,至少 2 条 |
| few-shot benefit < 0.08 | 禁用示例,避免无效 token 成本 |
| JSON reliability < 0.90 | 强制 schema、禁止额外字段、提高格式严格度 |
| JSON reliability > 0.96 | 可降低重复 schema 描述 |
| reasoning > 0.90 且 self-verification benefit > 0.25 | 使用 hidden analysis 并启用验证 |

## 工程验收(v1.0)

- [x] TaskSpec 编译出 ≥3 模型的 Prompt
- [x] 15 Probe 可运行并生成画像
- [x] ≥50 样本自动评测
- [x] ≥100 候选搜索
- [x] 全量实验可追踪/可复现
- [x] GLM → Qwen/GPT 运移分数 ≥90%

## 许可证

[MIT](LICENSE) © 2026 lanshi17
