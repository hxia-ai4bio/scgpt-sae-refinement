# SAE residual diagnosis: experiment integrity audit

Date: 2026-10-19 (Asia/Singapore). Reviewer: requested GPT-6-Astra ultra, fresh read-only reviewer. Same-family / provisional; executor exact model unknown. Exact backend identity is not separately exposed to this reviewer.

**Overall verdict: WARN. Integrity status: warn. No unresolved material computation error was found in the bounded checks.** The warning concerns exploratory selection, the scope of causal attribution, incompletely matched controls, and recorded numerical warnings. It does not invalidate the descriptive diagnosis.

## A. Ground-truth provenance — PASS

Classification labels come from Kang dataset cell-type annotations, not model outputs. The reviewer independently mapped all 9,577 cached rows to `kang.h5ad` annotations. `prepare.py:16-23,40-46` establishes provenance; `diagnose.py:30-37` extracts only the historical TRAIN/VALID rows and checks the cached labels/row IDs. All six splits have four TRAIN donors and disjoint TRAIN/VALID rows. The full source archive contains TEST rows, but the follow-up code never builds a TEST/PBMC scoring input. Source hashing is not scoring.

`diagnose.py:83-85` also predicts the model's own correction from h. This is explicitly a correction-predictability proxy. Its Q² is not biological ground truth, classifier accuracy, proof of global linearity, or proof of invertibility.

## B. Scores, normalization and numerical behavior — PASS with recorded warning

Macro-F1 is the unnormalized mean across the fixed eight labels, with absent-class F1 set to zero (`experiment.py:61-63`). No score is divided by the model's own maximum. Fold 4 VALID has no megakaryocytes, so its fixed-eight Macro-F1 ceiling is 0.875; folds 1 and 5 have one such VALID cell. This makes rare-class changes discrete and potentially influential. `class_supports.csv` discloses the supports.

The augmented classifier fits `[standardized h, eta × standardized z]`. Means, standard deviations, and active-coordinate masks are learned on TRAIN only, eta is applied after scaling, and no later standardization cancels it (`diagnose.py:90-104`). In unscaled z coefficient coordinates, its objective is weighted square loss plus `alpha ||W_h||² + alpha/eta² ||W_z||²`. The weighted mean/intercept and ±1 class targets match the original Ridge convention. Identity wins ties; lower eta and alpha break later ties by loop order.

There are 115,458 stored NumPy matmul RuntimeWarnings in trajectory execution and 3,780 in branch execution, in equal divide-by-zero/overflow/invalid groups. They are not hidden. Finite-score and normal-equation checks passed. The reviewer additionally bypassed NumPy matmul with SciPy dgemm and Cholesky: all 24 trajectory-selected readouts plus six identity instances, each with all 11 alphas, reproduced the selected alpha, predictions and F1; maximum selected-score absolute error was 6.58e-11. Some selected epoch0 instances duplicate identity. Two fold0 augmented branch solutions independently matched, with maximum score errors 2.04e-14 and 2.59e-10. These checks validate the reported bounded results; they do not certify all future executions in this environment.

## C. Artifact existence, immutability and replay — PASS

- All 1,062 entries in the original experiment's `FINAL_MANIFEST.json` matched their hashes. The 63 original source entries and 43 diagnostic source entries also matched.
- All 378 historical candidate rows reproduce their saved best/stop epochs, CE changes, head deltas and Ridge deltas. Gate rejection is exactly 13/18 at epoch0 checkpoint selection, 3/18 at fresh Ridge selection, and 2/18 retained.
- All 24 forty-epoch trajectories and all 12 conditional-readout families completed. There are 984 epoch records, 504 saved trajectory VALID prediction files, and 396 branch candidates.
- Independent row-ID/label and confusion-matrix recomputation matched every saved trajectory and branch F1 exactly. This covers 1,436,550 primary score rows; repeated rows are evaluations of the same historical cells, not additional samples.
- Twelve original selected joint checkpoints match the replay tensors bitwise. The historical prefix VALID F1 matches at every saved historical epoch. Full original model states at every prefix epoch do not exist, so full-prefix bitwise state identity is not claimed.

Evidence: `review_trace/static_checks.json`, `prediction_checks.json`, `selected_readout_checks.json`, `independent_numerical_checks.json`, and `report_number_checks.json`; source replay assertions are at `diagnose.py:66-74`. The numeric output and report mappings were read directly.

## D. Execution and dead-code coverage — PASS

The trajectory, gradient, fixed-head, fresh-Ridge, and branch metrics are actually executed and appear in saved records (`diagnose.py:64-86,98-107`). All 33 candidates in each branch were checked against saved predictions and the selection rule. No reported outcome depends on an uncalled metric. The unused local `prefix_ok` variable is cosmetic; assertions, not that variable, enforce replay. `historical.py` and `summarize.py` reconstruct the historical tables and aggregate the completed results without fitting to TEST/PBMC.

## E. Scope and explanatory controls — WARN

This is a post-hoc diagnostic on the same six historically explored TRAIN/VALID splits after earlier TEST results were known, using only SAE/training seed42 for new trajectories and branches. Each trajectory probes 21 checkpoints and selects the Ridge alpha on the same VALID; the branch chooses among 33 configurations plus identity on that VALID. Maxima are therefore optimistically selected. They are not new held-out generalization estimates, and an identity fallback makes selected gain nonnegative by construction.

The current code establishes several real differences but does not isolate one complete root cause:

- CE training uses class-uniform, within-class donor-balanced replacement sampling (`experiment.py:116-124`); fresh Ridge uses class-balanced cell weights without donor balancing (`experiment.py:169-187`). The difference is real. Its contribution to performance was not separately ablated.
- Joint versus frozen head is a controlled trainability comparison under the same sampler, learning rate and clipping rule. A modest change after freezing does not prove head co-adaptation was the sole cause.
- The orthogonally rotated decoder retains the learned z, dictionary Gram matrix and parameter count. It tests decoder alignment under this residual formulation; it is not a complete random-feature null.
- The random TopK branch matches nominal 2048 coordinates, K32 and search budget. Actual active coordinates differ (learned 1993–2004, random 2013–2036), and activation frequencies/correlations, effective rank and multiple random seeds are not matched (`diagnose.py:21,87-104`). A learned-versus-random difference cannot be assigned uniquely to SAE semantics.
- The expanded readout has independent weights on up to 2,560 features. The original residual classifier constrains the z weights through the decoder and scalar gates. These hypothesis classes are different. z is a deterministic function of h: any extra readout signal is improved accessibility/nonlinear transformation, not new information independent of h.
- Penalty-gradient ratios refer to one fixed balanced TRAIN batch per probed checkpoint, with beta0.1. They are not a complete pathwise causal attribution under Adam. The historical beta0 control provides separate corroboration against the simple explanation that a large penalty alone forced identity.
- The clarified `before_stop_better_than_original_joint_folds` field for frozen trajectories compares a newly frozen-head trajectory with the old joint checkpoint. Only its joint rows can describe a missed candidate along the replayed original training prefix. The final method text makes this distinction explicitly.

## F. Evaluation-type classification — PASS

Classification F1: `real_gt`, with exploratory validation selection. Residual predictability Q²: `synthetic_proxy`, explicitly labeled. Gradient norms, residual norms and hashes: optimization/numerical diagnostics. None is evidence of causal biological mechanisms.

## Action and claim impact

Retain the bounded results and all warnings; no additional large experiment is required to complete this diagnosis. Corrected reporting already calls the historical norm column `max_batch_RMS_relative_residual`: it is the maximum batch RMS ratio, not the largest individual-cell ratio. Future independent performance claims require a newly frozen protocol, comparable control budgets, and genuinely untouched biological units. Avoid repeatedly tuning these same six VALID donors.

The supported result is a qualified explanation of the original selection funnel and a demonstration that the follow-up changes do not establish an SAE-specific performance advantage. Exclusive root-cause claims and intrinsic-uselessness claims remain unsupported. See `CLAIMS_FROM_RESULTS.md`.

Report status/index refreshes and the executor's final package manifest may follow this review. Reviewed prose snapshots and all analyzed code/data/result hashes are retained in `review_trace`; no further seal-and-review loop is required for those presentation-only updates.
