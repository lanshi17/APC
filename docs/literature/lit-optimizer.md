# Lit-Optimizer：自动 Prompt 优化算法前沿调研 (2022–2026)

> 说明：arXiv 直读验证 12 篇 ID 与摘要，细节压缩，存疑标待核对。参考文献见各条目；另 Cui survey 2502.18746 作 taxonomy。
> 待补：APO 数据集名/精确预算、PromptWizard 139 调用的 PDF 一校、BBTv2 8k 前向的 PDF 一校；HELM/ModelCards/RouteLLM/Hyperband 背景引用；PromptBreeder venue（Nature MI 存疑）。

- **P1 APE** (Zhou, 2211.01910, ICLR23)：指令当程序，LLM 提候选 forward/reverse，子集打分多轮重采样；空间=单条 NL 指令；基准 24 任务 zero-shot；19/24 超人类；无预算调度；APC 下限基线。
- **P2 OPRO** (Yang, 2309.03409)：轨迹（候选，得分）排序入 meta-prompt，LLM 看轨迹生成下一批，循环爬山；空间=NL 指令；GSM8K +8%，BBH +50%（弱基线相对值）；百步×8 候选无调度；轨迹回放思想源头。
- **P3 APO/ProTeGi** (Pryzant, 2305.03495, EMNLP23)：minibatch 反例→NL 梯度批评→反方向编辑→beam+bandit 剪枝；初版 +31%；bandit 剪枝=预算调度最早近似物；APC 反思变异算子前身。
- **P4 PromptBreeder** (Fernando, 2309.16797)：双层进化（任务 prompt×变异 prompt）；空间非结构化文本；GSM8K/SVAMP/StrategyQA/Ethos；超 CoT/Plan-and-Solve；固定代数无 SHA；最接近 APC 外壳，差结构化基因+画像。
- **P5 EvoPrompt** (Guo, 2309.08532, 2023)：GA 轮盘赌+LLM 交叉变异，DE 语言差分 donor；31 数据集+BBH，GPT-3.5/Alpaca；BBH +25%；pop 约 10×gen 约 10 无调度；APC 直接对标，SHA 打其均匀预算软肋。
- **P6 PromptWizard** (Agarwal, 2405.18369, MSR24)：意图分析→critique/synthesis 精修→合成正负样例→验证，一次性编译；45 任务；约 139 调用[待核对]+成本表；任务侧画像=TaskSpec 唯一前身（无模型侧）；APC 预算对比标杆。
- **P7 GEPA** (Agrawal, 2507.19457, 2025)：Genetic-Pareto，轨迹→NL 反思诊断→测试更新→Pareto 合并；6 任务/AIME-2025；超 GRPO 平均 6%（最高 20%）省 35×rollout，超 MIPROv2 逾 10%（AIME +12%）；最强竞品，基线必含且同预算比；APC 差异=基因组+画像+迁移。
- **P8 DSPy** (Khattab, 2310.03714, ICLR24)：prompt 编译器提出者，Signature+模块→变换图→teleprompter 搜指令 demos；唯一结构化（模块级，基因内仍整段）；HotPotQA/数学/agent；+25~65% 超 few-shot；命名碰撞：论证 DSPy=单程序单模型搜索，APC=画像映射+基因进化+迁移，可叠加。
- **P9 MIPRO** (Opsahl-Ong, 2406.11695, EMNLP24)：感知提议+minibatch surrogate+元优化（学提议分布，非模型画像）；7 程序 Llama-3-8B；5/7 胜最高 +13%；surrogate 与 SHA 正交可叠加；基线 No.2。
- **P10 TextGrad** (Yuksekgonul, 2406.07496, NeurIPS24)：文本 autograd，TGDM 反传；GPQA/LeetCode/BBH；GPT-4o 51→55，代码 +20%；引用证反思已成标配。
- **P11 GrIPS** (Prasad, 2203.07281, 2022)：短语四算子+贪心爬山+patience，API-only；8 分类任务；+4.3pp≈人工改写；预算对齐先驱；APC 基因变异前身。
- **P12 RLPrompt** (Deng, 2205.12548, EMNLP22)：策略网络生成离散 token，reward 稳定化；few-shot 分类/风格迁移；gibberish 可跨模型迁移；APC 迁移弹药。
- **P13 BBTv2** (Sun, 2205.11200, 2022)：子空间+CMA-ES 连续 prompt，V2 逐层分治；≈全量微调；进化可行性证明；APC 选离散换可解释/可移植/API-only。
- **P14 Prompt tuning** (Lester, 2104.08691, EMNLP21)：冻模型学软 prompt，XXL 打平全量微调；定位差异：软要梯度 vs 离散要 API；软绑架构 vs 离散可迁移=APC 迁移合法性；可叠加先 APC 后 LoRA。

## 空白点四问

- **Q1 空间与优化器分类**：空间三类（整段 NL/邻域/模块），优化器三类（反思/进化/选择贝叶斯）；10 基因位点化无先例；反思×进化×SHA 三合一无先例。
- **Q2 画像驱动编译空白=主 claim**：全方法模型解耦，迁移仅事后观察；最近似为任务侧画像（PromptWizard）与提议元学习（MIPRO）；claim 收紧为"模型侧向量→规则前向映射"。
- **Q3 显式 SHA 先例未发现**：近似 APO bandit（必引）/MIPRO surrogate（叠加）/Hyperband（补引）/GEPA rollout 货币（模仿）；主张首次显式引入+SHA on/off ablation。
- **Q4 效果幅度**：弱基线 +25~65%，强基线 +2~13%；必对标 GEPA 同预算/MIPROv2/Wizard 成本表；基准 BBH+GSM8K+AIME 系+多阶段程序；加分=跨模型迁移矩阵+画像距离 vs 损失。

**创新排序**：画像编译 > 基因组 > 迁移协议 > SHA。
