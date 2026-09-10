# F11 预注册(gepa/espo 臂结果落盘前锁定)

时间锚:本文档 commit 时 HLE pilot 四臂(z0/champs×3)已定案、gepa/espo 两臂在跑
且无任何 holdout 数字落盘(real_hle.json 仅 4 行)。

## 已知前提(pilot 定案)
- z0 holdout30:acc .100(3/30;clean 30/30,err 后补)/ fmt .833(≈全部由网络错误贡献,
  真实协议违反≈0)/ TrialScorer hold .392
- 三个 transfer champ(math/contract/financial genome)全部 .388-.402,与 z0 同带:
  **genome 结构先验在知识型难任务零增益;协议余量假设未兑现**(强模型指令跟随足够)

## 预测(二选一,判据先锁)
P1(tie 分支):gepa/espo 臂 val 搜索后 holdout acc 落在 z0±2SE(.10±.11)与
hold±.02 带内 → **知识型 headroom 下反思通路同样不可达** → F11 = regime map 的
headroom 类型学补全:格式型 headroom(F10 仿真正例)结构通路可赢;知识型 headroom
任何 prompt 通路不可达(瓶颈在权重非提示)。六→八臂全平,与 F1-F9 自洽。

P2(反思胜出分支):某臂 acc ≥ .24(≈+1SE)→ 反思通路发现了领域启发式 →
报告胜出机制与内容(冠军 prompt 的实质规则)。

P3(genome 反胜分支,不预期):search/champ 臂 acc 高出于所有反思臂 >2SE → 结构先验
在 HLE 有正面价值(pilot 数据已反对此分支)。

## 判读协议
- holdout 同 seed 921 配对;跨臂差 <.015 视为带内(qwen 同日漂移带 .007 的 2 倍
  作保守地板,HLE 题目方差更大)
- call-error 行按零分计入(所有臂同一网络环境,公平);错误率报出
- 反思内容 diff 定性分析随结果附
