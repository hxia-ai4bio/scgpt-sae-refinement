# Claims from results

**claim_supported: yes — 仅限预先限定的 checkpoint 选择诊断。**  
**Confidence: medium；same-family / provisional；integrity: WARN（探索性范围）。**

在历史 Kang ctrl 六供体、三个配对种子和固定遮蔽/探针协议下，仅用完整 VALID 表达目标选择 SAE checkpoint，使冻结后新线性探针的 TEST 表达 MAE 从原始 h 的0.325955及固定 C1000的0.334610降至0.320772；两项比较均为18/18配对、6/6供体改善。固定15NN仍优于原始h，但低于C1000。

| 指标 | 原始 h | 固定 C1000 | VALID所选 C |
|---|---:|---:|---:|
| 表达 MAE ↓ | 0.325955 | 0.334610 | **0.320772** |
| clean 15NN MacroF1 | 78.1001% | **83.0073%** | 81.8670% |
| masked 15NN MacroF1 | 76.0808% | **81.4598%** | 80.9919% |

表达 MAE 相对原始 h 降低1.5900%，相对 C1000降低4.1355%。13条轨迹选50步，5条选100步。该结果说明此前固定1000步的负向表达读出比较依赖 checkpoint；不应从该终点推广为“SAE整体无用”。同时，当前选点也不是所有任务的共同最优点。

不支持：稀疏性必要、普遍表征改进、不可恢复的信息损失或已证明的过拟合机制。历史 TEST 重用、VALID多阶段复用、六供体训练折重叠、单一固定遮蔽与有限探针预算均限制外推。

证据：`SUMMARY.json`、`contrasts.csv`、`selected_checkpoints.csv`、`knn_summary.csv`、`verification.json`及独立审查 trace。全部主指标按供体内三种子平均后，再等权平均六供体。

**Route:** 确认本轮限定结论并停止；完成当前诊断不需额外实验。更广的泛化或机制主张需要另行授权的独立证据。
