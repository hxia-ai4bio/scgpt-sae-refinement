from pathlib import Path
import json,hashlib,datetime,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/sae_residual_diagnosis';O=ROOT/'outputs/sae_residual_diagnosis';T=W/'review_trace'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
meta=read(T/'run.meta.json')
for n in ['RESULTS_ZH.txt','REPORT.html','EXPERIMENT_TRACKER.json']:
    shutil.copy2(O/n,T/(n+'.reviewed_snapshot'))
inputs=set()
for p in [W/'frozen_sources.json',ROOT/'work/sae_residual/frozen_sources.json']:
    inputs.add(p);inputs.update(Path(k) for k in read(p))
inputs.update(p for p in W.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt','.py'] and T not in p.parents and '__pycache__' not in p.parts and p.name!='write_audit.py')
inputs.update(p for p in O.iterdir() if p.is_file() and p.name not in ['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md','CLAIMS_FROM_RESULTS.json','findings.md','RESULTS_ZH.txt','REPORT.html','EXPERIMENT_TRACKER.json','FINAL_MANIFEST.json'])
inputs.update(T/n for n in ['static_checks.json','prediction_checks.json','independent_numerical_checks.json','gradient_checks.json','selected_readout_checks.json','report_number_checks.json','verified_trajectory_summary.csv','evidence_claims.json','evidence_precheck.json','RESULTS_ZH.txt.reviewed_snapshot','REPORT.html.reviewed_snapshot','EXPERIMENT_TRACKER.json.reviewed_snapshot'])
hashes={str(p):'sha256:'+sha(p) for p in sorted(inputs)};save(T/'audited_input_hashes.json',hashes)
common=dict(executor_model='unknown',executor_family='openai',reviewer_model='gpt-6-astra',reviewer_family='openai',reviewer_reasoning='ultra',model_identity_basis='requested fresh reviewer route; exact backend identity not independently exposed',review_independence='same-family',acceptance_status='provisional',agent_id='/root/residual_diagnosis_audit',verdict_id='2026-10-19_sae_residual_diagnosis',generated_at=now,date='2026-10-19',trace_path=str(T),audited_input_hashes=hashes)
audit_md='''# SAE residual diagnosis: experiment integrity audit

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
'''
claims_md='''# SAE residual diagnosis: result-to-claim assessment

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
'''
checks={
'gt_provenance':{'status':'pass','details':'All 9577 dataset labels matched original h5ad annotations; TRAIN/VALID masks independently verified.','evidence':['diagnose.py:30-37','prepare.py:16-23,40-46','review_trace/static_checks.json']},
'score_normalization':{'status':'pass','details':'Fixed8 macroF1; TRAIN-only scaling; no prediction-statistic normalization. Independent Cholesky checks passed despite recorded matmul warnings.','evidence':['experiment.py:61-63,169-187','diagnose.py:90-104','review_trace/selected_readout_checks.json']},
'result_existence':{'status':'pass','details':'1062 old manifest entries unchanged; 24 trajectories/12 branches; 504+396 prediction files; 12 selected states bitwise replayed.','evidence':['review_trace/static_checks.json','review_trace/prediction_checks.json']},
'dead_code':{'status':'pass','details':'All reported diagnostics execute and have saved artifacts.','evidence':['diagnose.py:64-86,98-107','summarize.py:13-37']},
'scope':{'status':'warn','details':'Seed42 posthoc historically explored validation; same VALID selects epochs and alphas; controls partly matched; no TEST/PBMC scoring.','evidence':['PROTOCOL.json','diagnose.py:12-22','summarize.py:45-49']},
'eval_type':{'classification':'real_gt','correction_predictability':'synthetic_proxy','gradient_norms':'optimization_diagnostic'}}
audit={**common,'audit_skill':'experiment-audit','verdict':'WARN','overall_verdict':'warn','integrity_status':'warn','reason_code':'exploratory_selection_and_bounded_mechanism_attribution','summary':'Descriptive diagnostics and bounded numerical checks pass; no SAE-specific generalization or exclusive root cause is established.','checks':checks,'unresolved_material_computation_errors':[],'permitted_post_review_refreshes':['RESULTS_ZH.txt review-status only','REPORT.html review-status/index only','EXPERIMENT_TRACKER.json status only','executor final package manifest'],'claims':[{'id':'C1','impact':'supported','text':'Historical rejection funnel and nonzero updates.'},{'id':'C2','impact':'needs_qualifier','text':'Checkpoint/readout mismatch can discard higher VALID Ridge checkpoints; no expected new-data benefit implied.'},{'id':'C3','impact':'unsupported','text':'SAE features intrinsically useless or a unique root cause established.'},{'id':'C4','impact':'unsupported','text':'New generalization gain or SAE semantic specificity.'}]}
claims={**common,'claim_skill':'result-to-claim','claim_supported':'partial','integrity_status':'warn','confidence':'medium','confidence_by_scope':{'descriptive':'high','bounded_mechanism':'medium','generalization':'low'},'what_results_support':'Rejection is mostly at epoch0 checkpoint selection; updates occur; same-VALID CE-head/Ridge checkpoint preferences differ in a subset; bounded alternatives do not establish learned-SAE specificity.','what_results_dont_support':['Intrinsic uselessness of SAE features','A unique causal failure mechanism','New TEST/PBMC or independent generalization improvement','Biological semantic specificity'],'missing_evidence':['Untouched biological evaluation after freezing the rule','Isolated alignment of weighting/objective/checkpoint selection','Matched effective feature statistics and multiple random seeds'],'suggested_claim_revision':'The current pipeline discards most residuals early and exhibits limited exploratory readout mismatch; the diagnostic neither establishes SAE uselessness nor reliable SAE-specific benefit.','next_experiments_needed':[],'optional_future_evidence':'Only if this direction is pursued: preregister/freeze a bounded aligned-readout test with matched controls on untouched biological units; no automatic additional run.','routing_action':'narrow_claim_and_complete_diagnosis','performance_claim_advance':False}
(O/'EXPERIMENT_AUDIT.md').write_text(audit_md);save(O/'EXPERIMENT_AUDIT.json',audit)
(O/'CLAIMS_FROM_RESULTS.md').write_text(claims_md);save(O/'CLAIMS_FROM_RESULTS.json',claims)
(O/'findings.md').write_text('''# Research findings — frozen SAE residual diagnosis

- Experiment integrity: WARN; no unresolved material computation error found in independent bounded checks. Same-family/provisional.
- Result-to-claim: partial. Historical rejection stage and optimizer movement are established; exclusive cause, SAE intrinsic uselessness, and new generalization gains are not.
- Gate rejection funnel: epoch0 13/18, fresh Ridge 3/18, retained 2/18. Beta0 also selects epoch0 in 13/18 cases.
- In seed42 joint Gate replay, three folds contain a better probed Ridge VALID checkpoint before old stopping, but all comparisons reuse historically explored VALID labels.
- More training/frozen head yields small selected VALID maxima; learned decoder is not consistently better than rotation. Expanded learned and random TopK readouts both show selected gains under a more flexible classifier.
- Keep the current experiment immutable. Do not retune the same six VALID donors to claim generalization. Do not merge expanded-head diagnostics with the original residual's performance protocol.
- Recorded matmul warnings require preservation. All selected trajectory/identity readout selections were independently reproduced with dgemm/Cholesky; this bounded success is not certification of the environment for future experiments.
- Routing: finish this explanatory task with qualified claims. Further experiments are optional future work, not automatically authorized by the partial verdict. See CLAIMS_FROM_RESULTS.md for the precise evidence gap.
''')
(T/'001-review.response.md').write_text(audit_md+'\n\n'+claims_md)
save(T/'001-review.parsed.json',{'integrity_verdict':'WARN','claim_supported':'partial','confidence':'medium','integrity_status':'warn','review_independence':'same-family','acceptance_status':'provisional','unresolved_material_computation_errors':[]})
save(T/'001-review.routing.json',{'routing_action':'narrow_claim_and_complete_diagnosis','additional_training':False,'performance_claim_advance':False,'pending_external_review':True,'permitted_post_review_refreshes':audit['permitted_post_review_refreshes']})
save(T/'001-review.meta.json',{k:v for k,v in common.items() if k!='audited_input_hashes'}|{'call_number':1,'purpose':'experiment-integrity-and-claim-review','status':'ok'})
snap=T/'final_review_artifact_hashes.json';save(snap,{str(O/n):sha(O/n) for n in ['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md','CLAIMS_FROM_RESULTS.json','findings.md']})
event={'event':'review_trace','skill':'experiment-audit+result-to-claim','purpose':'experiment-integrity-and-claim-review','agent_id':'/root/residual_diagnosis_audit','trace_path':str(T),'status':'ok','verdict':'WARN','claim_supported':'partial'}
ep=ROOT/'.aris/meta/events.jsonl';ep.parent.mkdir(parents=True,exist_ok=True)
with ep.open('a') as f:f.write(json.dumps(event)+'\n')
print('AUDIT_WRITTEN',len(hashes),'audited input hashes')
