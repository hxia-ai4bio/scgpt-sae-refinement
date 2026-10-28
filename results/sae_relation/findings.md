# Research Findings — SAE relation continuation

2026-10-19；固定协议已完成，未追加训练。

- **主判断：不支持预设弱关系收益。** Weak reconstruction=82.580512%，对原h −0.173448 pp，对匹配MSE −0.127207 pp；gate=false。
- **保留次要信号：** strong reconstruction=83.475216%，对原h +0.721256 pp、对MSE +0.767497 pp，但两项仅3/6供体改善；patient_1016贡献集中。不升级为主方法。
- **计算真实性：** 54训练、150选中预测/向量、1650 VALID候选、全部分组/对比/区间/gate独立复算一致；没有通过增加训练或改结果“修复”负结果。
- **不能归因：** old20→MSE提升不能唯一归因训练时长；当前记录不识别弱项失败的唯一机制。关系几何不是生物真值。
- **范围：** 一个历史探索性Kang队列，六供体训练折重叠；不作独立确认、显著性、因果或普遍SAE无用论断。
- **小型来源问题：** upstream manifest仍含旧的Megakaryocytes排除字段；实际当前八类包含60例，原始标签逐行核对一致。
- **约束：** 保持弱主/强次及固定一半次要地位；保留全部供体和seed；不按TEST选择λ、混合系数或checkpoint。
- **路由：** 当前实验停止。未来若获授权，以新队列事先锁定强约束假设和强控制；不自动启动补充实验。

审查者gpt-6-astra/ultra，canonical `/root/sae_relation_audit`；**same-family/provisional**。稳定输入共1496项SHA256及完整request/response/parsed judgment/routing保存在两套指定trace；opaque agent id不可用，未编造。最终呈现文件已重渲染并封存，见AUDIT_SEAL.json。
