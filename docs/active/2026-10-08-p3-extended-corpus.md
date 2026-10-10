# P3 扩展语料：实施说明与协议

**Status:** in-progress
**Created:** 2026-10-08
**Plan:** `02_apc_paper/P3_IMPLEMENTATION_PLAN.md`（Week 1–3 的代码部分）
**Code:** `code/jev_rsi/corpus_extended/`、`code/jev_rsi/analysis_extended.py`

---

## 1. 本次实施范围

P3 计划共 4 周。论文仓库 `02_apc_paper` 不在本会话的可写工作区（只读），
因此本次只实施 **Week 1–3 的代码与数据部分**：

| 计划任务 | 状态 | 产出 |
|---|---|---|
| C1 对抗摘要数据 | ✅ | `corpus_extended/scenario_c/{source_articles.json,groups.json}` |
| C2 Val/Holdout 评分 | ✅ | `corpus_extended/scenario_c/scores.json` |
| C3 集成数据加载器 | ✅ | `jev_rsi/data.py::load_all_groups` |
| A1 创意写作任务 | ✅ | `corpus_extended/scenario_a/tasks.json` |
| A2 200 臂生成 | ✅ | `corpus_extended/scenario_a/arms.json` |
| A3 Judge 评分 | ✅ | `corpus_extended/scenario_a/scores.json` |
| A4 集成 | ✅ | 同上 |
| B1 图像数据 | ✅ | `corpus_extended/scenario_b/{images/,source_images.json}` |
| B2 Caption 生成 | ✅ | `corpus_extended/scenario_b/arms.json` |
| B3 Val/Holdout 评分 | ✅ | `corpus_extended/scenario_b/scores.json` |
| B4 集成 | ✅ | 同上 |
| E1 重跑主实验 | ✅ | `results/jev_rsi_extended_results.json` |
| E2 场景级分析 | ✅ | `jev_rsi/analysis_extended.py` + `results/jev_rsi_extended_analysis.json` |
| 验收 | ✅ | `scripts/p3_acceptance.py`（10 pass / 1 partial / 1 fail，见 §1.1） |
| 追加：B 组内排序 holdout | ✅ | `build_scenario_b.py --stage rank`（绕开 vision judge 天花板） |
| 追加：judge 方差跨模型探针 | ✅ | `corpus_extended/probe_judge_variance.py` |
| 追加：尺度重标定敏感性 | ✅ | `analysis_extended.scale_sensitivity` |
| P1–P4 论文整合 | ⏸️ 需论文仓库写权限 | `scripts/p3_paper_tables.py` 生成 LaTeX 片段 |

### 1.1 验收结果（`python scripts/p3_acceptance.py`）

| 检查 | 结果 |
|---|---|
| C 组数 / 臂数 / 每组 ≥2 作弊臂 | 10 / 60 / min 3 ✅ |
| C 解耦验收线 `val>0.4 且 holdout<0.3` ≥15 | **0** ⚠️（该线在本数据不可满足，见 5.2） |
| A 组数 / 臂数 | 20 / 200 ✅ |
| A 高方差组（std>2.0）≥5 | **0/20** ❌（前提不可复现，见 5.2） |
| B 组数 / 臂数 | 15 / 90 ✅ |
| 全部组 ≥78 / 决策组 ≥70 | 88 / 77 ✅ |
| 扩展组全部为决策组 | 45/45 ✅ |

---

## 2. 与计划的偏差（全部为环境约束，非结果导向）

计划假定可用 `gpt-4o` / `gpt-4o-vision` 并跑 BERTScore。实测环境如下。

### 2.1 模型替代：gpt-4o → qwen3.8-flash

`.env` 中 `OPENAI_API_KEY` 为空；唯一可用凭证是 `DASHSCOPE_API_KEY`，其 key
限制只放行 262 个目录模型中的两个：

| 角色 | 计划 | 实际 | 说明 |
|---|---|---|---|
| 文本生成/judge | gpt-4o | `qwen3.8-flash` | 主模型 |
| 对照/备用 | — | `qwen3.8-27b` | 同一 key 放行，可复现对照 |
| 视觉生成/judge | gpt-4o-vision | `qwen3.8-flash` | 实测接受 image_url / data URI |
| Embedding（未使用） | — | `qwen3.7-text-embedding` | 备用语义指标 |

`qwen3.8-flash` 是混合推理模型：默认把 reasoning 链计入 completion
（约 +1200 token/次）。**所有生成调用**显式传 `enable_thinking=false`；
**judge 调用保留推理链**（`max_tokens=200/500`），与仓库对 gpt-6 的处理一致。
两者都写进每个产物的 `provenance`。

### 2.2 Holdout 指标：BERTScore 的参考文本

计划伪代码用 `rouge.get_scores(summary, article)` 和 `bert_score(summary, article)`，
即以**整篇原文**为参考。这在 CNN/DailyMail 上不是标准协议，且数值退化：摘要是
~60 词、原文 300–520 词，ROUGE-L F1 的 recall 上限约 0.2，所有臂都被压在 0.15
以下，无法区分任何臂。

本实施改为以**数据集自带的人工摘要（highlights）**为参考，这是摘要评测的标准
做法。为透明起见，同时记录以原文为参考的变体（`rougeL_vs_article`、
`bertscore_raw_vs_article`）。

### 2.3 BERTScore 用 baseline-rescaled 数值

`bertscore(..., rescale_with_baseline=True)` 作为 `holdout_score`。原始（未
rescale）F1 对任何通顺文本都在 0.9 上下，永远不可能低于计划验收线 0.3；
rescale 后无关文本≈0、高质量≈0.6–0.8，计划里的 `holdout < 0.3` 才有意义。
原始值同样记录在 `metrics.bertscore_raw_vs_reference`。
模型为计划指定的 `microsoft/deberta-xlarge-mnli`（`transformers` 固定 <5，
5.x 与 bert_score 的 tokenizer 调用不兼容）。

### 2.4 分数尺度统一到 0–1

Scenario A/B 的 judge 是 0–10；`core.NOISE_BAND=0.007`、`ALIGN_SCALE=0.021`
以及门控阈值都定义在冻结语料的 0–1 尺度上。因此 `val_score` / `holdout_score`
一律归一化到 0–1，原始 0–10 数值保留在 `judge_raw`。

### 2.5 extractive 臂不用 LLM

计划要求用 gpt-4o 生成"抽取式"摘要；LLM 写出的摘要不是抽取式。extractive 臂
改为真正的 TF-IDF 句子选择器（确定性、可复现），关键词/模板/随机臂也都是确定性
构造。只有 `abstractive` 需要调用模型。

### 2.6 Occam 序（arms 的保守度排序）预先登记

`data.py::FAMILY_ORDER` 新增三族，顺序在观测任何分数**之前**确定：

* `summarization`: lead3 < extractive < abstractive < template_filled < keyword_stuffed < random_highlight
* `creative_writing`: zero_shot < few_shot_2ex < cot < step_by_step < constrained < persona_* < creative_mode < temperature_*
* `image_captioning`: factual < concise < verbose < technical < child_like < poetic

各自的"无适配臂"（`BASE_ARM_NAMES`）为 `lead3` / `zero_shot` / `factual`，
即门控的兜底目标。

### 2.7 计划自身的算术不一致

计划多处写 "78 组 = 43 + 10 + 20 + 15"，但 43+10+20+15 = **88**；
"decision=70" 也与同一段的 "32 + 45" 不一致。验收脚本同时报告实际值与计划值，
不做静默对齐。

---

## 3. 数据来源（全部真实、可追溯）

| 场景 | 来源 | 数量 |
|---|---|---|
| C | `abisee/cnn_dailymail` 3.0.0 / validation，取 300–520 词文章 | 10 篇 + 人工 highlights |
| A | 手写任务库（5 类 × 4） | 20 任务 × 10 策略 = 200 臂 |
| B | `nlphuji/flickr30k` TEST（5 条人工描述/图） | 15 图 × 6 风格 = 90 臂 |

CNN/DailyMail 与 Flickr30k 都经 HuggingFace `datasets-server` JSON API 抓取，
不下载整包数据集；图片已落盘到 `scenario_b/images/`。

**所有数字来自真实运行**：生成文本存在 `arms.json`，judge 分数与 5 次原始打分
存在 `scores.json`。API 响应另有磁盘缓存 `_cache/llm/`（已 gitignore）与调用账本
`_cache/usage.jsonl`，重复运行零成本、逐字节一致。

---

## 4. 复现

```bash
cd /mnt/data/Projects/DataScience/Analysis/02_APC

# 0) 依赖（一次性）：CPU torch + bert-score + rouge-score，transformers<5
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python \
    bert-score rouge-score "transformers<5"

export HF_HOME=$PWD/jev_rsi/corpus_extended/_cache/hf   # BERTScore 模型缓存
export MPLCONFIGDIR=/tmp/mpl

# 1) 源材料
python -m jev_rsi.corpus_extended.fetch_sources cnn_dailymail --n 10
python -m jev_rsi.corpus_extended.fetch_sources flickr30k --n 15

# 2) 三个场景（缓存命中时秒级重放）
python -m jev_rsi.corpus_extended.build_scenario_c
python -m jev_rsi.corpus_extended.build_scenario_a
python -m jev_rsi.corpus_extended.build_scenario_b          # arms + scores + rank

# 2b) 附加：judge 方差跨模型探针（约 300 次调用）
python -m jev_rsi.corpus_extended.probe_judge_variance \
    --models qwen3.8-flash qwen3.8-27b --groups 5 --draws 6

# 3) 验收 + 主实验 + 场景分析
python scripts/p3_acceptance.py
python -m jev_rsi.experiments --corpus extended     # -> results/jev_rsi_extended_results.json
python -m jev_rsi.analysis_extended                # -> results/jev_rsi_extended_analysis.json
python scripts/p3_paper_tables.py                  # -> results/p3_paper_tables.tex
```

冻结语料的旧结果可用 `python -m jev_rsi.experiments --corpus frozen` 复现，
产物仍是 `results/jev_rsi_results.json`，两条路径互不覆盖。

---

## 5. 结果

语料：**88 组 / 485 臂 / 77 个决策组**（冻结 43 + C 10 + A 20 + B 15）。
完整机器可读结果见 `results/jev_rsi_extended_results.json` 与
`results/jev_rsi_extended_analysis.json`，论文表格片段见
`results/p3_paper_tables.tex`。

### 5.1 主表（决策组，Jev-static vs val-argmax）

| 场景 | n | greedy | heuristic | Jev (static) | Jev W/L/T | paired p |
|---|---|---|---|---|---|---|
| 原冻结语料 | 32 | 18/32 (+0.0125) | 13/32 (+0.0130) | 11/32 (+0.0138) | 3/12/17 | 0.035 |
| C 对抗摘要 | 10 | 6/10 (+0.0239) | 6/10 (+0.0239) | 6/10 (+0.0239) | 0/0/10 | 1.00 |
| A 创意写作 | 20 | 8/20 (+0.0355) | 8/20 (+0.0355) | 5/20 (**+0.1655**) | 4/9/7 | 0.27 |
| B 图像描述 | 15 | 15/15 (+0.0000) | 15/15 (+0.0000) | 15/15 (+0.0000) | 0/0/15 | 1.00 |
| **合计（决策组）** | **77** | **47/77** | 42/77 | 37/77 | 7/21/49 | **0.0125** |

全语料配对检验：`greedy − jev_static` 均值 **−0.0343**，
95% CI [−0.0658, −0.0082]（Jev 更差）。扩展到 3 个新场景后，
"结构化门控 ≥ val-argmax"这一假设**没有被支持，反而被更强地拒绝**。

### 5.2 逐个场景：为什么没赢

**Scenario C（对抗摘要）—— 前提不成立：便宜作弊臂在 val 上也不占优。**
6 臂均值（val ROUGE-L F1 / holdout rescaled BERTScore）：

| 臂 | 类型 | val | holdout |
|---|---|---|---|
| lead3 | baseline | 0.235 | 0.225 |
| extractive | normal | 0.151 | 0.129 |
| abstractive | normal | 0.252 | **0.276** |
| template_filled | gaming | 0.126 | −0.029 |
| keyword_stuffed | gaming | 0.171 | **−0.135** |
| random_highlight | gaming | 0.121 | 0.086 |

作弊臂在 holdout 上确实崩塌（BERTScore 为负），**但在 val 上同样低**：
ROUGE-L F1 对"关键词堆砌"不给分（F1 惩罚长度/精度）。因此 val-argmax
在 **0/10** 组被作弊臂捕获，greedy 与 Jev 完全相同。
计划验收线 `val>0.4 且 holdout<0.3` 无法触达：参考人工摘要只有 ~30 词，
所有臂的 ROUGE-L F1 上限约 0.38，**该验收线在本数据上不可满足**。
另记：`rougeL_vs_article` 变体下也只有 1/10 组被捕获。
均值 Spearman(val, holdout) = **+0.50**（正相关，无解耦可利用）。

**Scenario A（高方差 judge）—— 前提不成立：judge 方差远低于计划假设。**
200 臂，val = 单次打分，holdout = 同 prompt 5 次独立打分的均值。
实测每臂 judge 标准差均值 **0.47**（0–10 分制），**0/20** 组超过计划的 2.0。
跨模型探针（`probe_judge_variance.py`，5 组 × 10 臂 × 6 次）：

| judge | 每臂 sd 均值 | 最大 sd | sd>2.0 的臂 |
|---|---|---|---|
| qwen3.8-flash | 0.556 | 1.374 | 0/50 |
| qwen3.8-27b | 0.675 | 1.344 | 0/50 |

推理型 judge 在 temperature=1.0 下过于稳定，"高方差 judge"场景在可用模型上
**无法构造**。门控在此处**损失最大**：它 15/20 次退回到兜底臂 `zero_shot`，
而 `zero_shot` 在创意写作里并不安全（多组中 `persona_*`/`few_shot` 明显更好），
平均 regret 从 0.036 涨到 0.166。

**Scenario B（多模态对齐）—— 度量饱和：vision judge 分辨率不足。**
vision judge 对 6 种风格几乎全部给 9–10 分（factual 1.000、concise 0.993、
technical 0.990、verbose 0.953、child_like 0.920、poetic 0.860），
所有策略 15/15 全对、regret 0。为绕开天花板，追加**组内排序** holdout
（同一模型看图像+6 条候选，输出严格排序）：

| 风格 | 排序 holdout 均值 |
|---|---|
| technical | 0.893 |
| factual | 0.880 |
| verbose | 0.440 |
| concise | 0.427 |
| poetic | 0.200 |
| child_like | 0.160 |

均值 Spearman(val, vision 排序) = **+0.527**（15 组中仅 1 组为负）：
文本 judge（看人工参考）与视觉 judge（看图）**一致**，
且 val-argmax（`technical`，val=0.833）在排序上也是次优仅次 `factual`。
不存在"文本 judge 被诗意描述骗到"的解耦。

### 5.3 尺度迁移（门控真正的失效机制）

门控阈值在原语料上标定，新场景的 val 尺度完全不同：

| 场景 | val 中位数 margin | 相对 NOISE_BAND(0.007) | alignment 饱和比例 |
|---|---|---|---|
| 原语料 | 0.0006 | 0.1× | 28% |
| C | 0.0257 | 3.7× | 60% |
| A | 0.0000 | 0.0× | 35% |
| B | 0.1000 | 14.3× | 67% |

用每个场景自身的 val 间距重标定尺度（`analysis_extended.scale_sensitivity`）：
C 中静态门控**更差**（4/10, +0.0435），但**场景内 LOO-RSI 能回到 greedy 持平**
（6/10, +0.0239）——说明损失来自阈值/尺度失配，而非三个原语本身无用。

### 5.4 边界条件（Jev 何时真的赢）

77 个决策组中 Jev 赢 **7** 组、输 **21** 组。7 个胜场：
`financial-s45`、`financial_gpt56terra-s42`、
`transfer_financial_to_math_gpt6-s42`（原语料），
`creative_dialogue_01`、`creative_scene_01`、`creative_story_02`、
`creative_story_03`（Scenario A）。
共同特征：**val 排序最靠前的臂与"最保守臂"重合或接近**，
即门控的兜底选择恰好是对的——这正是原论文里已被报告的机制，
P3 没有发现任何新的胜场区间。

一个反特征同样清楚：当**兜底臂本身不安全**（Scenario A 的 `zero_shot`）
或**val 度量与 holdout 度量正相关且同尺度**（Scenario B/C）时，
门控只会引入损失。


---

## 6. 已知局限

1. **尺度迁移**：门控的 `NOISE_BAND` / `ALIGN_SCALE` 在冻结语料上标定；新场景的
   val 尺度（ROUGE-L、归一化 judge 分）差 0–14 倍，导致 alignment 规则在多数组
   饱和为 1.0。`analysis_extended.py::calibration` 与 `scale_sensitivity` 量化了
   这一点，但没有改动冻结协议本身。
2. **三个扩展场景的前提均未成立**：C 的"高 val 作弊臂"、A 的"高方差 judge"、
   B 的"度量分歧"都没能在真实模型/数据上复现（详见 5.2）。因此本轮得到的是
   *关于前提的负面结果*，而不是"门控在有解耦时仍然失败"的强命题。
   计划中这些前提是文字假设，本实施把它们变成了可测量的验收项——这本身就是
   可报告的结果。
3. **单次 judge 的采样噪声**：Scenario A 的 val 是单次采样，重新运行（清空缓存）
   会得到不同数值；协议本身即如此设计，缓存保证本仓库内可复现。
4. **BERTScore 的已知弱点**：rescaled BERTScore 对词序不敏感，对"通顺但错误"的
   文本可能偏宽松；这是选择它作为 holdout 的代价，已在分析中如实呈现。
5. **`has_collapse` 的定义在新尺度上失真**：其判据含 `val < 0.35`，而 C 场景所有
   臂的 ROUGE-L 都低于 0.35，于是 25 组被判为"collapse 组"。这是尺度伪影，
   已由 5.3 的 calibration 表说明，未用于任何结论。
6. **图片版权**：Flickr30k 图片按其研究用途引用，未再分发到论文仓库。
7. **`qwen3.8-27b` 仅用于 judge 方差探针**，未跑全量第二模型对照（成本与计划
   时间盒不成比例；探针已足以否定"judge 高方差"这一前提）。

---

## 7. Week 4：论文整合与投稿包（2026-10-09 完成）

论文正文的整合由本仓库的 `docs/paper_patch/` 工具产出，全部改动写入
`02_apc_paper`（论文仓库），本仓库不保存稿件副本：

| 脚本 | 作用 |
|---|---|
| `patch_manuscript.py` | 把 §5.6（Table 8/9/10）、摘要/引言/结论更新、Discussion 段落幂等插入 `negative-result.md`；锚点缺失或重复即报错 |
| `verify_numbers.py` | 从 JSON 产物重新推导论文引用的 45 个数字，逐个在正文中查证，任一不符即失败 |
| `make_supplementary.py` | 由结果 JSON 生成 `supplementary.md` → `supplementary.pdf`（7 页：协议、逐组表、诊断、标定、获胜特征、复现） |
| `make_tmlr.py` | 复用论文自身 `build.py` 的 body 转换，换成 TMLR 前导（`jmlr.cls` + `tmlr.sty`），把 9 处 arXiv 号改写为 `\citep{}` 并以 `tmlr.bst` 生成文献表 → 15 页投稿版 |
| `update_paper_docs.py` | 向 `RESULTS_ANALYSIS.md` §12、`EXPERIMENT_PROTOCOL.md` §5、`README_JEV_RSI.md` 追加 P3 章节，并在冻结语料主表处加指向 77 组全量结果的注 |
| `install_week4.sh` | 一键执行上述全部步骤 + 备份 + 投稿包（`bash docs/paper_patch/install_week4.sh`） |

产物落点：`manuscript_negative/negative-result.{md,tex,pdf}`（15 页）、
`manuscript_negative/supplementary.{md,pdf}`（7 页）、`manuscript_negative/tmlr/`、
`manuscript_negative/code.zip`、`submission_tmlr/`（投稿包：TMLR PDF/TEX、
`references.bib` + `tmlr.bst` + `tmlr.sty` + `.bbl` + `figures/`、补充材料、
`README_reproduce.md`、`code.zip`——可在包目录内独立编译，已实测 9 条引用全部解析）。

**与计划的两处落点偏差（已在论文仓库跟踪表中记录）**：

1. 计划写"主表更新为 78 组"，实际做法是**保留冻结语料的 Table 1**，另加
   Table 8 给出三场景 + 77 决策组合计——改写既有主表会让 §5.1–5.5 的全部数字
   失去上下文。计划中的 78 与 70 本身也是计划方 43+10+20+15 / 决策组数的算术
   误差，实测为 88 组 / 77 决策组（`scripts/p3_acceptance.py` 同时报告两者）。
2. TMLR 版式落在 `manuscript_negative/tmlr/`，`negative-result.tex` 仍是
   草稿版式（`build.py` 生成）。原因是草稿版式便于阅读与批注，且
   `build.py` 的注释已说明"camera-ready 时替换前导"，故未覆盖原文件。

**TMLR 构建的两个本地修补**（均入库 `docs/paper_patch/vendor/`）：

1. TeX Live 的 `jmlr.cls` 已定义 `aftertitskip` / `beforetitskip` /
   `interauthorskip` / `aftermaketitskip`，而上游 `tmlr.sty` 用 `\newlength`
   重复定义会导致编译中断。修补版（四处 `\@ifundefined` 守卫）作为
   `docs/paper_patch/vendor/tmlr.sty` 入库，`make_tmlr.py` 会在下载的上游副本上
   自动施加同一补丁。
2. `jmlr.cls` 在导言区声明 `plainnat.bst`，与正文的
   `\bibliographystyle{tmlr}` 冲突——BibTeX 报 "Illegal, another `\bibstyle`
   command" 后回退到 plainnat，正文 9 处引用全部渲染成 `(?)`。
   `make_tmlr.py` 在首次 pdflatex 后清掉 `.aux` 中多余的 `\bibstyle` 行，
   再按 pdflatex → bibtex → pdflatex ×2 固定跑四遍（latexmk 在 bbl 刚生成后
   不会重跑 LaTeX）。文献库为 `vendor/references.bib`，样式 `vendor/tmlr.bst`。

**复现论文数字**：

```bash
cd .                                   # 代码仓库
python -m jev_rsi.experiments --corpus frozen    # 冻结产物逐字节不变
python -m jev_rsi.analysis_extended
python -m jev_rsi.experiments --corpus extended
python scripts/p3_acceptance.py                  # 10 pass / 1 partial / 1 fail
python docs/paper_patch/verify_numbers.py --md ../02_apc_paper/manuscript_negative/negative-result.md
```
