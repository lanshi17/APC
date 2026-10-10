# P3 任务跟踪表

**计划文档**: `P3_IMPLEMENTATION_PLAN.md`
**开始日期**: 2025-01-03
**完成日期**: 2025-01-04 (Weeks 1–4)
**执行方式**: Claude 实现 + 自验收（原计划的 dst/omp 未参与）

---

## 📋 任务清单

### Week 1: Scenario C (对抗样本) — ✅ 完成

| 任务 | 状态 | 验收标准 | 实际产出 |
|------|------|---------|---------|
| **C1** 构造对抗摘要数据 | ✅ 已完成 | 10组×6臂=60臂 | `code/jev_rsi/corpus_extended/scenario_c/{groups,tasks}.json` — 10组×6臂=60臂 |
| **C2** Val/Holdout 评分 | ✅ 已完成 | ROUGE + BERTScore | `scenario_c/scores.json`：ROUGE-L vs **人工要点**(val) + 基线重标定 BERTScore(vs 摘要) |
| **C3** 集成到数据加载器 | ✅ 已完成 | `load_all_groups()` | `jev_rsi/data.py`：`load_all_groups()` = 53 组（冻结 43 + C 10） |
| **验收** Week 1 | ✅ 通过 | 运行验收脚本通过 | 10 组/60 臂；`scripts/p3_acceptance.py` |

**偏离**：计划按“摘要 vs 全文”计算 ROUGE-L，实测所有臂 F1 ≈ 0.2 上限，无法构造 `val>0.4` 的捕获带；改为按数据集人工要点计分，并同时记录全文对照指标（8 个指标全部留档）。

### Week 2: Scenario A (高方差 Judge) — ✅ 完成

| 任务 | 状态 | 验收标准 | 实际产出 |
|------|------|---------|---------|
| **A1** 设计创意写作任务 | ✅ 已完成 | 20个任务，5种类型 | `scenario_a/tasks.json`：20 任务 × 5 类（故事/诗/对话/场景/寓言） |
| **A2** 生成 10 种 Arm | ✅ 已完成 | 200臂 (20×10) | `scenario_a/arms.json`：200 臂，全部真实模型生成 |
| **A3** Judge 评分 | ✅ 已完成 | 单次+5次平均 | `scenario_a/scores.json`：val=1 次抽样，holdout=5 次均值（temperature 1.0） |
| **A4** 集成数据加载器 | ✅ 已完成 | `load_all_groups()` | 63 组 |
| **验收** Week 2 | ✅ 通过（含 1 项 FAIL 记录） | 方差分析通过 | `judge_variance_probe.json`：flash σ̄=0.556，27b σ̄=0.675，100 臂无一 σ>2.0 |

**偏离**：计划假设 judge 高方差（≥5 组 σ>2.0）。实测 0/20 组、0/100 臂达到该阈值 → 该前提在真实模型上不可达，**保留为论文的负结果之一**，不通过调参伪造。

### Week 3: Scenario B (多模态) + 完整实验 — ✅ 完成

| 任务 | 状态 | 验收标准 | 实际产出 |
|------|------|---------|---------|
| **B1** 图像数据收集 | ✅ 已完成 | 15张图+参考描述 | `scenario_b/`：Flickr30k 15 图 × 5 参考描述 |
| **B2** Caption 生成 | ✅ 已完成 | 90臂 (15×6风格) | `scenario_b/arms.json`：90 臂 |
| **B3** Val/Holdout 评分 | ✅ 已完成 | 文本judge vs 图文judge | `scenario_b/scores.json` + 追加"同图排序"holdout（10 分制视觉分饱和） |
| **B4** 集成数据加载器 | ✅ 已完成 | `load_all_groups()` | 88 组 = 冻结 43 + C 10 + A 20 + B 15（计划的"78"是其自身 43+10+20+15 的算术误差） |
| **E1** 重跑主实验 | ✅ 已完成 | 完整结果 | `jev_rsi/results/jev_rsi_extended_results.json`（77 决策组） |
| **E2** 消融与场景分析 | ✅ 已完成 | 场景级对比表 | `jev_rsi/analysis_extended.py` → `jev_rsi_extended_analysis.json`；获胜特征：7 胜全部是“退避到事后更优的兜底臂”（理由恒为 `low_val_alignment`），21 负全部是不必要的退避（补充材料 §S6） |
| **验收** Week 3 | ✅ 通过 | 核心发现确认 | 见下"核心结果" |

### Week 4: 论文整合 — ✅ 完成

| 任务 | 状态 | 验收标准 | 实际产出 |
|------|------|---------|---------|
| **P1** 更新 Results 章节 | ✅ 已完成 | 3个场景完整写入 | `manuscript_negative/negative-result.md` §5.6（含 Table 8/9/10）+ 摘要/引言/结论同步 |
| **P2** 更新 Discussion | ✅ 已完成 | 反映P3实际发现 | §7 新增"Extended-corpus boundary conditions"（"情况 2：P3 仍为负结果"） |
| **P3** TMLR 格式转换 | ✅ 已完成 | 编译无错 | `manuscript_negative/tmlr/negative-result-tmlr.pdf`（15 页，`jmlr.cls` + `tmlr.sty` + **`tmlr.bst` 作者-年份文献表**：正文 9 处 arXiv 号改写为 `\citep{}`，由 `references.bib` 生成） |
| **P4** 补充材料准备 | ✅ 已完成 | code.zip + supplementary | `manuscript_negative/supplementary.pdf`（7 页，含获胜/失败特征 §S6）+ `manuscript_negative/code.zip` = `submission_tmlr/code.zip` + `README_reproduce.md` |
| **终审** | ⏸️ 待人工确认 | 全文通读，投稿就绪 | 数字已由 `docs/paper_patch/verify_numbers.py` 逐条对账（45 项）；draft 15 页 / TMLR 15 页 / 补充材料 7 页均零错误 |

---

## 🎯 里程碑

| 里程碑 | 目标日期 | 标志 | 状态 |
|--------|---------|------|------|
| **M1**: Scenario C 完成 | 2025-01-10 | 53组数据可用 | ✅ |
| **M2**: Scenario A 完成 | 2025-01-17 | 63组数据可用 | ✅ |
| **M3**: 全部数据 + 实验完成 | 2025-01-24 | 88组结果产出 | ✅ |
| **M4**: 论文投稿就绪 | 2025-01-31 | TMLR 提交 | ✅（待人工终审） |

---

## 📊 核心结果（全部为真实模型运行）

| 策略 | 77 决策组 exact | 平均 regret |
|------|----------------|------------|
| `greedy`（validation argmax） | **47/77** | +0.01754 |
| `heuristic` | 42/77 | +0.01773 |
| `jev`（静态门） | 37/77 | +0.05184 |

配对检验 `greedy − jev_static`：−0.0343，95% CI [−0.0658, −0.0082]，W/L/T = 7/21/49，p = 0.0125。

- **Scenario C**：0/10 组被 gaming 臂捕获；val 上限 0.380，pre-registered 捕获带（val>0.4）不可达。门与基线 6/10 打平。
- **Scenario A**：门 5/20 vs 基线 8/20（+0.1655 vs +0.0355），15/20 次退回 `zero_shot` 兜底臂——该臂在创意写作上并不安全。
- **Scenario B**：视觉 judge 饱和（6 风格中 5 个 ≥0.86）；文本 judge 与视觉排序 Spearman +0.527，所有策略 15/15。
- **失败机理**：门阈值随任务尺度漂移（0.1×–14.3× `NOISE_BAND`）；按场景重新标定后 Scenario C 的 LOO-RSI 恢复到与基线持平（6/10）。

---

## 📝 状态图例

- ⏸️ 待开始 / 待人工确认
- 🔄 进行中
- ✅ 已完成
- ⚠️ 有问题（记录在案）
- ❌ 失败/阻塞（保留为负结果）

---

## 🔁 复现

```bash
cd ../02_APC        # the code repository (sibling of this paper repo)
python -m jev_rsi.corpus_extended.build_scenario_{a,b,c}   # 命中磁盘缓存，无 API 花费
python -m jev_rsi.experiments --corpus extended
python -m jev_rsi.analysis_extended
python scripts/p3_acceptance.py
python docs/paper_patch/verify_numbers.py                  # 论文数字对账
```

---

**更新日志**:
- 2025-01-03: 初始化任务跟踪表
- 2025-01-04: Weeks 1–4 全部完成；记录 3 项计划前提实测不成立（负结果）与 5 项协议偏离

---

## 📦 交付清单对照（Week 4 结束）

| 计划要求 | 实际路径 | 状态 |
|---|---|---|
| `manuscript_negative/negative-result.pdf`（11+ 页） | `manuscript_negative/negative-result.pdf`（**15 页**） | ✅ |
| `manuscript_negative/negative-result.tex`（TMLR 格式） | `manuscript_negative/tmlr/negative-result-tmlr.tex` + `references.bib` + `tmlr.bst`（TMLR 版式与 natbib 文献表）；`negative-result.tex` 保留草稿版式 | ✅（落点偏差见下） |
| `manuscript_negative/supplementary.pdf` | `manuscript_negative/supplementary.pdf`（**7 页**，全部表格由结果 JSON 自动生成） | ✅ |
| `manuscript_negative/code.zip` | `manuscript_negative/code.zip` = `submission_tmlr/code.zip`（2.1 MB，345 文件，匿名泄漏扫描通过） | ✅ |
| `README_reproduce.md` | `submission_tmlr/README_reproduce.md` | ✅ |
| 投稿包 | `submission_tmlr/`（TMLR PDF/TEX、补充材料、README、code.zip） | ✅ |
| Results 含全部 3 个场景 | `negative-result.md` §5.6（Table 8/9/10） | ✅ |
| 主表更新为 78 组数据 | 冻结 Table 1 标题改为 frozen corpus (32) 并加注指向 **Table 8**（三场景 + **77** 决策组合计）；`RESULTS_ANALYSIS.md` §3.1 与 `README_JEV_RSI.md` 主结果表同样加指引 | ✅（计划数字本身有误，见下） |
| Discussion 反映 P3 发现 | §7 "Extended-corpus boundary conditions"（情况 2：仍为负结果） | ✅ |
| 论文编译通过（无 LaTeX 错误） | draft 15 页 / TMLR 15 页 / 补充材料 7 页，零错误 | ✅ |
| 符合 TMLR 格式要求（页数、参考文献样式） | `jmlr.cls` + `tmlr.sty` + `tmlr.bst`；正文 9 处 arXiv 号 → `\citep{}`，文献表由 `references.bib` 生成 | ✅ |
| 补充材料完整 | 协议、逐组表、诊断、标定、获胜特征、复现命令 | ✅ |
| 数字可追溯 | `code/docs/paper_patch/verify_numbers.py` 逐条核对 45 个数字 | ✅ |
| 代码仓库产物 | `jev_rsi/corpus_extended/`、`load_all_groups()`、`jev_rsi_extended_results.json`（**88 组**）、`analysis_extended.py` | ✅ |

**两处落点偏差**：(1) TMLR 版式落在 `manuscript_negative/tmlr/`，顶层的
`negative-result.tex` 仍是 `build.py` 生成的草稿版式（其注释本就说明 camera-ready
时替换前导）；(2) 计划写"主表更新为 78 组"，实际做法是保留冻结语料的 Table 1 并
另加 Table 8 给出全量合计——改写主表会让 §5.1–5.5 的数字失去上下文。计划中的
78 与 70 是计划方 43+10+20+15 与决策组数的算术误差，实测为 **88 组 / 77 决策组**
（`code/scripts/p3_acceptance.py` 同时报告计划值与实测值）。

**TMLR 构建的两个本地修补**（均入库为 `code/docs/paper_patch/vendor/`）：
(1) TeX Live 的 `jmlr.cls` 已定义 `aftertitskip` 等四个长度，上游 `tmlr.sty` 用
`\newlength` 重复定义会中断编译 → 加 `\@ifundefined` 守卫；
(2) `jmlr.cls` 默认在导言区声明 `plainnat.bst`，与正文的 `\bibliographystyle{tmlr}`
冲突会让 BibTeX 报 "Illegal, another \bibstyle command" 并回退到错误样式（引用
全部变 `(?)`）→ 构建脚本在首次 pdflatex 后清理 `.aux` 中多余的 `\bibstyle` 行，
再跑 bibtex + 两次 pdflatex。
