# SAE relation continuation — 实验完整性审查

日期：2026-10-19。审查者：**gpt-6-astra / ultra**，独立上下文 `fork_turns=none`；canonical agent `/root/sae_relation_audit`。opaque agent ID 在保留上下文不可用，未编造。语义审查等级：**same-family / provisional**；确定性数值核验单独记为通过。

## Overall Verdict: WARN

## Integrity Status: warn

未发现未修复的实质计算错误。WARN 来自历史探索性数据复用、仅六供体及一处旧来源元数据残留；不代表发现伪造指标。54 次训练与150份选中读出/预测均完成。主候选失败属于真实实验结果，不是完整性失败。

## A. Ground Truth Provenance: PASS

原始 `kang.h5ad` 的 cell_type 注释与9577行缓存标签、供体、cell ID、条件逐行一致；八类计数为1092/2307/4099/753/171/407/60/688。六折各四TRAIN、一VALID、一TEST，角色行集合不重叠；18个旧SAE缓存的54个分区行号与标签一致。分类是 `real_gt`；重建及关系保持是 `self_supervised_proxy`，不能当作真实生物关系。

证据：`/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/prepare.py:40–46`；`work/sae_refine/run.py:66–79`；`work/sae_complementarity/core.py:61–66`；`work/sae_relation/run.py:59`。

## B. Score Normalization: PASS

Macro-F1由真实类别混淆矩阵计算，固定八类，缺失类别按0处理；无预测自身最大值归一化。训练损失以固定TRAIN中心均方为分母，未缩放分类成绩。KL为teacher→student，原h教师detach，中心化均值来自TRAIN。实际平衡批次每查询192个不同供体锚点；float64概率行和检查有效。

Phase1 NumPy/einsum与logsumexp独立KL误差9.71e-17，最大概率行和误差1.34e-15；三个参数组KL梯度均非零。证据：`work/sae_relation/run.py:96–134,174–216`。

## C. Result File Existence and Numerical Checks: PASS

| 独立检查 | 范围与结果 |
|---|---|
| 来源/锁 | 60个来源文件、930个锁内文件哈希一致 |
| 神经网络 | 18个旧checkpoint及54个最终checkpoint；独立NumPy forward最大误差6.65e-07；旧缓存完整重建误差0 |
| 梯度与更新 | 18个初始化梯度重算；54组实际参数差、mean不变、decoder列范数及所有记录学习率核对 |
| 批次 | 18组1000×256索引重新生成，与持久化数组逐项相同 |
| VALID | 全部1650个alpha候选用独立Cholesky求解，F1与选择一致；选中VALID分数最大误差3.63e-12 |
| TEST | 150份、239425个重复模型-细胞预测；手工混淆矩阵全部指标误差0；150份512维导出逐项核对 |
| 汇总 | 9个方法/variant汇总、54个供体汇总、10个预定对比、1200个逐类指标、72个逐类汇总以及gate独立重算；最大汇总误差2.23e-14 |
| 证据预检 | 15/15精确指标值在来源文件存在；这只验证存在性 |

现有 `verify.py:66–87` 只复算选中alpha并读取其他trial分数；本次独立脚本补齐全部1650候选，未发现遗漏候选改变胜者。未运行优化器更新，也未重放完整历史/本次优化轨迹。日志、checkpoint及数值复算共同支持执行记录，不是对所有未记录行为的保证。

独立可复算记录：`work/sae_relation/review_trace/independent_numeric.py/.json` 与 `metrics_helper.py/.json`。全部输入SHA256（1496项）保存在相应JSON和trace，未在正文展开。

## D. Dead Code: PASS

训练目标、监控、梯度、分类指标、汇总和gate均有已执行入口及实际输出。支持的调用入口是直接执行runner或通过importlib加载；没有将该runner作为通用模块名`run`导入后宣称成功。证据：`work/sae_relation/run.py:174–229,294–424`；`verify.py:30–125`。

## E. Scope / Selection: WARN

所有训练、读出选择先于新TEST评价锁定；只用VALID标签选11档Ridge alpha，不进行TRAIN+VALID重拟合。训练checkpoint固定step1000；弱重建是唯一主候选，强约束及固定一半混合为次要。原18个checkpoint历史上按VALID MSE选epoch20，已核对为最早最小值。

当前仍是历史上已探索的同一个Kang队列、六供体、训练折相互重叠。供体bootstrap及推进gate均为描述性；18次fit或重复细胞预测不能当成18个独立供体。MSE continuation相对old20同时改变额外更新、均衡采样、Adam重置、lr日程，不能隔离“训练时长”。固定1000步也不证明收敛。

证据：`PROTOCOL.json`；`work/sae_relation/run.py:47–60,311–369,372–403`；`report.py:65–72`。

## F. Evaluation Type

- 最终分类：`real_gt`（数据集cell_type注释）。
- MSE/KL/关系保持及参数诊断：`self_supervised_proxy`。
- 未运行PBMC外部确认、因果干预、SDCV监督特征发现或原论文逐字节复现。

## Minor upstream metadata: WARN

`/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/data_manifest.json:36–38` 仍继承“排除Megakaryocytes”；实际缓存含60个该类，`prepare.py:40–46` 明确补提取并扩展八类，当前协议/切分/评价一致。这是旧描述残留；本审查记录其含义，不修改已冻结来源。原始少掉的row7894已定位，但本次未重新计算其输入QC门槛。

## Claim impact and actions

预设弱重建相对原h −0.1734 pp、相对匹配MSE −0.1272 pp，主假设不受支持且gate=false。强约束仅可记为受供体集中影响的次要积极信号，详见CLAIMS_FROM_RESULTS.md。保持原主分析，停止本固定实验；不因TEST结果换主候选或增加训练。报告需展示上述主/次区别与强约束集中性；其余现有报告无需数值更正。

最终REPORT、tracker与EXECUTION_LINK已重新核对并封存，见AUDIT_SEAL.json；1496项稳定输入重新计算SHA256仍一致。
