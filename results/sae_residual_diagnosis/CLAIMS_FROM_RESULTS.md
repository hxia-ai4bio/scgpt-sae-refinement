# SAE residual diagnosis: result-to-claim assessment

**claim_supported: partial**  
**integrity_status: warn**  
**review_independence: same-family; acceptance_status: provisional**  
Confidence is high for the descriptive selection/numerical findings, medium for the bounded mechanism interpretation, and low for extrapolation to new donors, different training objectives, or all SAE features.

## What the results support

The old residuals were mostly discarded by the training/selection pipeline: among 18 Gate fold–seed decisions, 13 had all three beta candidates select epoch0; three more rejected their learned correction at fresh Ridge selection; only two retained a correction. This exactly locates the rejection stage. It does not by itself explain the representation's intrinsic value.

The model was capable of updating. The new trajectories have nonzero gradients, changed gates and changed residuals. In the completed joint Gate trajectories the final raw residual ratio averaged 0.595%, while the ratio after the head's TRAIN standardization averaged 4.03%; raw norms hide some change because the shared mean is large. Probed maximum gate magnitudes stayed below the 0.5 bound. The largest penalty-to-CE theta-gradient ratio over all probed arms/modes was about 0.00245 on the diagnostic batch. All three historical beta settings, including beta0, had 13/18 Gate epoch0 selections. These observations do not support the simple story of disconnected gradients, an always-zero correction, or a dominant residual penalty.

There is a real checkpoint/readout mismatch. In the seed42 joint Gate replay, three of six folds have a probed fresh-Ridge VALID score above the original selected checkpoint before the original stopping epoch (folds 1, 3 and 4). Thus the CE-head checkpoint rule can discard a checkpoint preferred by the downstream Ridge criterion on the same VALID. Across the common checkpoint grid, the mean Ridge-versus-head selector gap is 0.15068 percentage points. This is a selected validation discrepancy, not the expected gain of replacing the rule on new data.

Training longer or freezing the head exposes only limited exploratory gains and does not establish SAE decoder specificity:

| Residual | CE head | Mean of each fold's best VALID gain, pp |
|---|---|---:|
| Learned decoder Gate | Joint | +0.198412 |
| Rotated decoder Gate | Joint | +0.182671 |
| Learned decoder Gate | Frozen | +0.409722 |
| Rotated decoder Gate | Frozen | +0.531768 |

All four values maximize over VALID checkpoints and Ridge regularization. Frozen rotated Gate exceeds frozen learned Gate on this aggregate. At fixed epoch40, mean Gate gain is −0.358907 pp for joint and +0.126707 pp for frozen; these fixed-endpoint values avoid epoch maximization but still use VALID-selected Ridge alpha and historically explored donors. A stable learned-SAE advantage is not demonstrated.

The expanded classifier `[h_std, eta z_std]` has mean selected VALID gain +0.742332 pp for learned SAE and +0.523762 pp for random TopK. Their mean difference is +0.218570 pp, with learned better in three donors, worse in one, tied in two. This is compatible with some task-relevant nonlinear features being accessible through an enlarged readout. The same validation selection and random-control limitations prevent treating it as proof of SAE-specific usefulness or independent improvement.

## What the results do not support

- A new TEST/PBMC improvement, independent generalization gain, or a confirmed gain from switching checkpoint rules.
- The conclusion that SAE features are intrinsically useless. Only a frozen K32 representation family, one new seed, a bounded gate formulation, and specific readouts/objectives were diagnosed.
- The opposite conclusion that the SAE learned biologically meaningful or uniquely useful features. Random features also help under the expanded search; the controls are only partially matched and no biological intervention is evaluated here.
- A unique causal blame on early stopping, the CE head, sampling weights, regularization, or insufficient training duration. The checkpoint discrepancy is shown, while several other differences remain hypotheses.
- A claim that the residual is merely an invertible linear change of coordinates. Correction-prediction Q² averages about 0.48–0.69 across arms/modes and is only a linear predictability diagnostic.

## Suggested claim revision

“在这组历史供体和既定训练方案下，SAE残差多数在检查点选择阶段退回原表征。独立回放发现，CE头选点与最终Ridge读出存在小幅验证偏好差异；梯度与残差确实更新，简单归因于未训练或强惩罚并不成立。延长训练、冻结头和扩大读出虽暴露部分探索性信号，但尚未形成优于旋转或随机对照的可靠SAE特异优势。因此当前证据定位了流程限制，不能判定SAE特征本质无用，也不能宣布新的泛化收益。”

## Missing evidence and next action

For the user's current explanatory question, stop here and retain the qualified diagnosis. Do not automatically launch broader training. If the direction is pursued later, freeze the readout/checkpoint rule and donor/class weighting together, compare expanded learned/random branches with matched effective feature statistics and random-seed budgets, and evaluate once on genuinely untouched biological units. More searches on these same VALID donors would not resolve the generalization gap.

Routing: narrow the claim to the descriptive diagnosis; no submission-facing performance claim. No repeat audit is required for status/index refresh and final packaging. Numerical and provenance evidence is recorded separately from this provisional semantic judgment.
