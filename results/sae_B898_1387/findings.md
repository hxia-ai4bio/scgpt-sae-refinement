# 本轮限定研究结论

2026-10-18：fresh GPT-6-Astra/ultra、fork_turns=none完成same-family/provisional审阅。

- experiment-audit: WARN；无未修复计算错误。34项来源hash、两特征缓存汇总、全部通路成员/全局BH及93项尾概率抽核通过。
- result-to-claim: yes，仅对候选贡献与注释任务；不支持稳定功能模块或因果机制。
- 普通ORA显著116/12307，主匹配与exact全局显著均0。阴性结果保留，不以MC分辨率解释掉。
- 对参照QC、覆盖稀疏性、单历史测试供体、同RNA证据、TopK无门控变化和跨特征不同原型保持限定。
- 复现包遗漏04_expression.py和HTML循环重复gate追加已修复。任务不要求新增实验；允许最终状态/索引刷新与封存。
