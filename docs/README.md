# 文档索引

项目文档按生命周期分类。根 [README.md](../README.md) 是项目总览。

## 目录结构

```
docs/
├── README.md          # 本索引(分类规则 + 文档清单)
├── PRD-APC.md         # 产品需求文档(长期有效)
├── REPRODUCE.md       # 完整复现序列(长期有效)
├── active/            # 进行中:计划、状态跟踪
├── planned/           # 未开始
├── archive/           # 已完成
│   ├── plans/         # 完成的计划/进展快照
│   └── codereview/    # 已批准的代码评审
├── codereview/        # 进行中的代码评审
└── templates/         # plan.md / codereview.md 模板
```

## 分类规则

新文档放 `active/` 或 `planned/`,并在文首加状态头:

```markdown
**Status:** planned | in-progress | completed
**Created:** YYYY-MM-DD
**Completed:** YYYY-MM-DD
```

- 计划完成 / 快照定稿 → 移入 `archive/plans/`
- 代码评审全部通过 → 移入 `archive/codereview/`
- 命名:`YYYY-MM-DD-<kebab-case>.md`(进展快照保留日期后缀)

## Active(进行中)

| 日期 | 文档 | 状态 |
|---|---|---|
| 2026-10-08 | [2026-10-08-p3-extended-corpus.md](active/2026-10-08-p3-extended-corpus.md) | P3 扩展语料(Scenario A/B/C)实施与协议,in-progress |
| 2026-10-02 | [JEV_RSI_WEEK2_PLAN.md](active/JEV_RSI_WEEK2_PLAN.md) | Week 2 实验计划,in-progress |
| 2026-10-02 | [JEV_API_STATUS.md](active/JEV_API_STATUS.md) | Jev API 状态跟踪,in-progress |

## Planned(未开始)

(空)

## Archive(已完成)

| 日期 | 文档 | 说明 |
|---|---|---|
| 2026-10-02 | [PROGRESS_SUMMARY_20261002.md](archive/plans/PROGRESS_SUMMARY_20261002.md) | Jev+RSI Phase 1 进展快照 |

长期参考文档:`[PRD-APC.md](PRD-APC.md)`、`[REPRODUCE.md](REPRODUCE.md)`。
