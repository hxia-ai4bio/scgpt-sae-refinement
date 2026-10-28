# F898 / F1387 实验完整性审查

审阅：GPT-6-Astra / ultra，`collaboration.spawn_agent`，`fork_turns=none`；executor 精确型号 unknown。独立读取原始文件，未重跑 scGPT。**same-family / provisional**；数值通过不等于生物学验证。

**Overall verdict: WARN；integrity_status: warn。** 未发现伪造真值、自我性能归一化、幻影结果或未修复的计算错误。当前成果可以作为候选基因贡献和通路注释交付。

| 检查 | 判断 | 证据与结论 |
|---|---|---|
| A 真值与代理 | PASS | `F898/run.py:15–18,52–56`、`expression_bcells.py:8–17`、`report.py:64–72`：标签读出、模型敏感性、通路成员和同RNA偏相关分开，未作为独立机制真值。 |
| B 分数归一化 | PASS | `F898/analyze.py:30–40,97–105`，F1387同位置：保留raw baseline/activation/drop，分层均值与条件均值分别报告。 |
| C 结果存在及数值 | PASS | 两特征各36对缓存、1152组合替换输入及完成标志存在。独立重建选择、聚合与每特征288个训练排序位置；34项来源hash通过。全部12307项通路成员、54候选条目、340基因链接通过；40000随机集合分层通过，93项尾概率/组合数学精确DP抽核通过。 |
| D 死代码 | PASS，已修复 | 主函数有调用和产物：`F898/analyze.py:107–110`、`pathways.py:51–55`。报告重复gate追加已移除；`package.py:26`已补04_expression.py。 |
| E 范围 | WARN | `run.py:24–36`：每特征8个TRAIN原型、单历史TEST供体4个原型、复用参照；`report.py:65–72,80–81`已保留限制。 |
| F 类型 | PASS | 细胞标签读出是数据注释比较；输入贡献/组合替换为self_supervised_proxy；表达偏相关为同RNA代理；ORA是固定注释关联。没有功能实验GT。 |

以上代码简写相对 `work/sae_B898_1387/`；结果位于 `outputs/sae_B898_1387/`。

必须保留的解释边界：

- 参照是同供体同类型零激活近邻，**不是QC匹配细胞**。F898 TRAIN平均检出基因数372.25/735.79、TEST395.25/1026.25；F1387也存在差异（`feature_summary.csv:2–3`）。
- TRAIN供体间贡献秩相关中位数F898为0.231、F1387为0.076；高排名不等于供体稳定，EMP3、S100A10等只有1/8原型覆盖（`feature_summary.csv:2–3`；`F898/gene_rankings_train.csv:15,17`）。HLA-DPA1和HLA-DPB1测试贡献反向，报告已明示。
- 训练排序未用新TEST贡献选基因；测试组合替换是在每个原型中选择可替换基因，不能写成统一40基因清单或独立生物干预（`analyze.py:73–89`）。非RPL/RPS排序没有自己的专门匹配随机对照。
- ORA背景为各特征实际TRAIN覆盖基因，保留无注释基因；基因列表及背景同步去RPL/RPS。全局BH跨2特征×2列表×3库，主MC与exact分别校正（`pathways.py:36–65`）。
- 粗分箱仍残留连续差异：F898非RPL/RPS列表检测率0.56185，对照0.52386，z=2.499（`matching_diagnostics.csv:21`）。exact消除MC采样误差，不能消除此可交换性假设。
- 普通ORA显著116/12307项，主匹配与精确敏感性显著均为0，全部匹配q和exact q均为1（`pathway_summary.csv:2–13`）。最小MC p约0.000500，最小exact p约0.000579；**不能将无发现主要归咎于MC分辨率**。
- 共27411个单基因改变输入及1920个非零组合替换输入均无目标TopK/encoder差异或目标退出TopK（`TopK_gate_diagnostics.csv`；统计代码`report.py:25–35`）。没有稀疏门控优势证据。
- RNA偏相关不是独立证据；embedding/PCA来自SAE所选细胞上的诊断坐标；F160原型、参照和检验家族不同，只能描述比较。

数值独立复核保存于 `work/sae_B898_1387/review_trace/`。最大聚合误差<1e-16；手工全局BH误差<1.2e-14；93项精确尾概率误差<7e-16。未重新计算全部模型前向；慢/快backend重放依据运行期保存的验证与hash绑定缓存，审阅另外重算了缓存选择、聚合、排序及通路统计。

当前任务不要求新增实验。交付保留以上限定即可。审阅时REPORT/RESULTS/TRACKER的hash为snapshot；随后仅刷新审查状态和文件索引，最终交付文件由FINAL_MANIFEST绑定。
