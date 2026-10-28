# Experiment Audit — Kang ctrl checkpoint selection

**Verdict: WARN（范围限制；未发现完整性缺陷）**  
**Review:** fresh same-family Codex；provisional。请求 gpt-6-astra / ultra，实际模型标识未暴露。

- **A. 目标来源 PASS：** 原始计数的 logCP10k 与数据集细胞类型注释；没有以模型输出构造目标。独立核对9,577个细胞元数据/标签，32个细胞的13,608个表达目标精确一致；六折基因面板和标准化均精确复现 TRAIN-only 计算。证据：`work/sae_masked_ac/prepare.py:24–37`、`pipeline.py:32–40`；审查 trace `003-independent-raw-check.json`。
- **B. 分数归一化 PASS：** 主指标为原尺度 MAE；辅助 MSE 使用 TRAIN 阳性表达标准差。改善百分比以参考模型 MAE 为分母。证据：`work/sae_expression_probe/run.py:163–167`、`work/sae_checkpoint_selection/report.py:19–24`。
- **C. 结果存在及对应 PASS：** 独立复算54行表达指标、36行所选模型15NN F1；18个新 step1000 状态均逐张量等于旧模型；18个主体→新头的选点与 checkpoint hash 一致。493项原输入、12项补充来源、288项选点锁文件均通过。六供体等权汇总正确，3个 evidence-check 值存在。证据：`verification.json`、trace `004-independent-final-check.json`。
- **D. 执行路径 PASS：** 实际保存的126个主体 checkpoint、18个新头及预测对应被调用的训练/选点/指标路径。证据：`work/sae_checkpoint_selection/run.py:129–167`；`verify.py:25–90`。
- **E. 范围 WARN：** 历史反复探索的 Kang ctrl，6供体×3配对种子，训练折重叠；固定一种阳性token遮蔽；warmup、主体与新头复用 VALID；有限探针预算、粗粒度选点网格。18个配对不是18个独立供体。报告已披露这些限制。
- **F. 类型：** 表达与细胞标签均为真实数据来源的评价目标；遮蔽重建仍仅衡量表达读出，不构成独立生物学或因果验证。

来源绑定缺口已在完整运行前修复。旧实验封存清单共2,613项文件在预审中全部一致。

保留的结论：本协议下，VALID选点改善表达读出。必须同时报告相对 C1000 的15NN下降；不支持“全部下游任务更优”“稀疏性必要”或普遍生物学优势。无需为完成本轮诊断追加训练。
