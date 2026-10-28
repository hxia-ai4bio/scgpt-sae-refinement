# 实验完整性审查

日期：2026-10-19（Asia/Singapore）。审查者：新鲜上下文 gpt-6-astra / ultra；same-family / provisional。执行者精确型号 unknown。

**Overall verdict: WARN；Integrity status: warn。数值与来源核验 PASS，未发现未修复的实质完整性错误。** WARN 对应证据范围和生物学解释限制。

以下缩写代码路径相对于项目内 `work/sae_B_remaining/`；结果表相对于 `outputs/sae_B_remaining/`。

## A. Ground truth provenance — PASS

AUROC使用原始数据细胞类型标签，独立检查已核对 H5→缓存标签映射与供体划分，并从全部2048个特征重算TRAIN eligibility与AUROC，六特征及数值完全一致。基因替换的终点是模型自身得分，通路是外部注释成员，均未伪装成生物学ground truth。证据：`work/sae_interpretability/common.py:56-61`；`work/sae_B_remaining/run.py:37-48`；`review_trace/independent_check.json`。

## B. Score normalization — PASS

单基因/组合替换保留原始激活差；AUROC、偏相关、ORA和MC均使用常规定义，无模型自身最大值归一化。参照→原型→供体逐级等权；未token覆盖记0，同时保留条件均值与覆盖数。证据：`run.py:64-73`、`analyze.py:10-40,98-104`、`pathways.py:50-65`。

## C. Result existence and mappings — PASS

6个特征均完成：216个参照对、6912个组合输入。F137另经独立缓存/近邻/汇总/随机顺序核算，分层均值最大误差约1e-16。最终独立检查确认39911项、480条top40记录、158个候选条目、740条候选基因映射及all9的720条基因/72条重叠记录；来源数值一致。根执行器precheck为9项verified。原生通路核验检查120000随机集及419项exact DP，最大误差约4.44e-16。证据：`verification.json`（各F目录）、`pathway_verification.json`、`work/sae_B_remaining/review_trace/final_evidence_check.json`。

## D. Dead code — PASS

计算与核验函数由实际pipeline调用，结果文件可追溯；未见只定义不执行却用于声明结果的指标。证据：`pipeline.py:4-10`、`finish_pipeline.py:4-9`、`analyze.py:107-110`、`pathways.py:21-77`、`expression_bcells.py:12-17`。

## E. Scope — WARN

固定K32、seed42；每特征4个TRAIN供体各2原型，历史单TEST供体4原型，最多3个可复用同供体B细胞参照。F1125/F1409最差TRAIN供体AUROC低于0.70。跨供体归因排名一致性较弱，不能由高AUROC推出稳定基因功能。

| 特征 | TRAIN平均AUROC | 最低供体AUROC | TRAIN top40在TEST正向/覆盖 | 供体排名Spearman中位数 |
|---|---:|---:|---:|---:|
| F137 | 0.877 | 0.864 | 19/27 | 0.085 |
| F370 | 0.731 | 0.713 | 27/39 | 0.171 |
| F146 | 0.746 | 0.710 | 17/31 | 0.038 |
| F1373 | 0.734 | 0.704 | 17/25 | 0.086 |
| F1125 | 0.701 | 0.678 | 14/26 | 0.086 |
| F1409 | 0.727 | 0.668 | 24/35 | 0.065 |

通路背景、5..500成员、RPL/RPS同步移除、无注释背景保留、全局BH家族正确；匹配仅在粗分箱内，连续指标仍有残差。**39911项中普通ORA显著155项，主MC与exact各0项；最小q分别0.39907和0.18403。** exact阴性说明本轮结果不能仅归因于有限MC分辨率，但仍受相同分箱假设约束。证据：`pathways.py:38-67`、`matching_diagnostics.csv`、`all_pathway_results.csv`。

各特征原型不同；embedding/PCA仅同批SAE原型上的诊断坐标。目标坐标未观察到TopK退出或超过3e-5的encoder差异，此结论不覆盖整个字典的成员变化。另列50条TEST正向候选记录是事后展示，不构成独立复现。证据：`report.py:34-58,71-78`、`TopK_gate_diagnostics.csv`、`DESCRIPTIVE_FOLLOWUP_RULE.json`。

## F. Evaluation type — WARN

- 细胞标签AUROC：`real_gt`。
- 输入替换与同RNA偏相关：`self_supervised_proxy`，没有独立生物学终点。
- 通路ORA：外部注释关联；按本技能分类属于无独立生物学GT的`self_supervised_proxy`，不是功能ground truth。

## Claim impact and action

支持完成六特征的描述性基因输入值敏感性与候选注释；不支持稳定功能模块、通路活性、因果机制或全局可解释性优势。现有报告已明确这些边界。当前请求无需新增实验；保留阴性结果和未过滤表，仅刷新审查状态/索引后打包。

审查使用的科学输入SHA256见JSON和`work/sae_B_remaining/review_trace/audited_hashes.json`。REPORT/RESULTS/tracker哈希为审查时snapshot；状态/索引刷新后的最终文件由FINAL_MANIFEST绑定。语义判断仍为same-family/provisional。
