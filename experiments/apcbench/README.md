# APCBench 实验说明（受控合成评测环境 v2.2）

> 诚实声明：因无真实 LLM API 凭证（`.env` 为空），以下全部结果来自仓库自带的
> `MockClient` 确定性仿真器。仿真器的 ground-truth 基因效应如下表所示——
> **阅读本说明即可复现全部"最优解"**，因此本基准衡量的是*优化器效率*
> （给定结构化搜索空间，谁用更少评估找到最优），而非绝对任务精度。
> 真实 LLM 验证是明确的未来工作（见 `docs/paper-apc.md` §Limitations）。

## 搜索空间（`configs/genomes/base.json`，13 位点）

| 位点 | 取值数 | 仿真器中的真实效应 |
|---|---|---|
| `examples.enabled` × `examples.count` | 2 × 4 | 数量 0→1→2 uphill；≥2 饱和；超饱和惩罚（见下） |
| `reasoning.strategy` | 4 | 模型相关亲和（见下）+ comment 丢失通道 |
| `verification.enabled` × `verification.type` | 2 × 3 | 禁用→风险只剩 2 条；type 影响 skill 小量 |
| `output.strictness` | 3 | low→schema 缺失→缺失字段腐蚀（gpt 免疫，missing_field=0） |
| 7 个装饰位点（instructions/style/constraints/goal/role/style/style.verbosity） | 2–4 | **本任务中性**（检验优化器聚焦能力） |

## Ground-truth 异质性（`mock_client.py`）

- 示例增益/饱和：glm（0.06/3）、qwen（0.09/2）、gpt（0.035/1）；超饱和 −0.06/个
- 推理亲和：glm 最爱 decompose，qwen 最爱 checklist，gpt 最爱 hidden
  （best−second ≈ 0.03 skill + dropout 差，属亚分辨率差异）
- 置信度校准：level≥2→0.78、=1→0.66、=0→0.60（金标准均值 ≈0.75）
- 探针画像（真实 `ProbeRunner` 结果）：三模型仅在 table_understanding /
  math / self_verification_benefit 上有差异；few_shot_benefit 全 0，
  json_reliability 全 0.5——画像信息量弱是已知局限

## 运行方法

```bash
.venv/bin/python scripts/bench_apc.py [BUDGET]   # 默认 100；输出 bench_results_b{BUDGET}.json
.venv/bin/python scripts/bench_transfer.py       # 迁移矩阵 → transfer_results.json
.venv/bin/python scripts/analyze_bench.py        # 配对 bootstrap → summary.json
```

## 主要结果（`summary.json`，b100，15 runs/方法）

- search ≫ zero-shot：apc-full +0.0093 [0.0082, 0.0102]（显著）
- evolution ≫ random：+0.0030 [0.0022, 0.0038]（显著；多步组合价值）
- PGAM ≈ uniform：+0.0008 [0.0000, 0.0017]（边缘；方差 0.0001 vs 0.0016）
- manual ≥ search：−0.0009（强人工启发式仍具竞争力；自动化的价值在免手工+迁移+血统）
- SHA / Profile-rules 消融：0.0000（零结果；base 已满足规则）
- 迁移：recover 1.0000 [0.9978, 1.0026]，KR-6 12/12，adapted@30 ≈ native@100
