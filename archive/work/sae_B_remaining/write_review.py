"""Serialize the fresh reviewer's final judgment and forensic trace."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd

R=Path(__file__).resolve().parents[2];W=R/'work/sae_B_remaining';O=R/'outputs/sae_B_remaining';T=W/'review_trace'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return 'sha256:'+h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

e=json.loads((T/'final_evidence_check.json').read_text());assert e['status']=='PASS'
pre=json.loads((T/'evidence_precheck.json').read_text());assert pre['summary']['verified']==9
fids=[137,370,146,1373,1125,1409]
summary=pd.read_csv(O/'feature_summary.csv')
gate=pd.read_csv(O/'TopK_gate_diagnostics.csv')
assert gate.SAE_encoder_unequal.sum()==0 and gate.TopK_exits.sum()==0
files=[W/n for n in ['run.py','analyze.py','pairs_qc.py','verify.py','pathways.py','verify_pathways.py','report.py','context.py','expression_bcells.py','pipeline.py','finish_pipeline.py','preflight.py','benchmark.py','backend_benchmark.json','frozen_sources.json','frozen_inference_sources.json','recovery/amendment.json','review_independent.py','review_final_check.py']]
files += [R/'work/sae_interpretability'/n for n in ['common.py','01_atlas.py','02_forward.py','04_expression.py','primary_activations.npz','qc.npz','pca.npz']]
files += [R/'outputs/sae_interpretability/input_manifest.json',R/'outputs/B_cell_feature_ranking.csv']
files += sorted((R/'work/sae_pathways_v3/gene_sets').glob('*.gmt'))
files += [R/'work/sae_pathways_v3/gene_sets/c5.go.bp.v2025.1.Hs.json']
exclude={'EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.json','EXPERIMENT_TRACKER.json','FINAL_MANIFEST.json'}
files += [p for p in sorted(O.rglob('*')) if p.suffix in ['.csv','.json'] and p.name not in exclude]
files += [T/n for n in ['independent_check.json','final_evidence_check.json','evidence_precheck.json','evidence_claims.json']]
files += [R/'outputs/sae_gene_attribution_v2/gene_evidence_combined.csv']+[R/f'outputs/sae_B898_1387/F{x}/gene_evidence_combined.csv' for x in [898,1387]]
hashes={str(p.relative_to(R)):sha(p) for p in dict.fromkeys(files)}
snapshots={str((O/n).relative_to(R)):sha(O/n) for n in ['REPORT.html','RESULTS_ZH.txt','EXPERIMENT_TRACKER.json']}
common={'agent_id':'/root/b_remaining_audit','verdict_id':'/root/b_remaining_audit/2026-10-19-final','executor_model':'unknown','executor_family':'openai','reviewer_model':'gpt-6-astra','reviewer_reasoning':'ultra','reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','backend':'collaboration.spawn_agent','fork_turns':'none','model_metadata_source':'requested route recorded from actual spawn metadata supplied by parent; executor exact model unavailable','generated_at':now.isoformat(),'date':now.date().isoformat(),'trace_path':str(T.relative_to(R)),'audited_input_hashes':hashes,'report_snapshot_hashes':snapshots,'snapshot_policy':'Report/text/tracker hashes are the reviewed snapshot with pending review state. Only review-status/link/index refresh is permitted after this verdict; final package is bound by executor FINAL_MANIFEST. No new seal loop is required.','pending_external_review':False,'external_review_status':'not performed or scheduled; semantic acceptance remains provisional'}
checks={
 'gt_provenance':{'status':'pass','evidence':['work/sae_interpretability/common.py:56-61','/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/prepare.py:16-23','work/sae_B_remaining/run.py:37-48','work/sae_B_remaining/review_trace/independent_check.json'], 'details':'AUROC uses dataset-provided cell-type labels; raw H5 label mapping and donor-disjoint split independently verified. Attribution and same-RNA associations are declared proxies, not manufactured biological ground truth. Gene sets supply annotation membership only.'},
 'score_normalization':{'status':'pass','evidence':['work/sae_B_remaining/run.py:64-73','work/sae_B_remaining/analyze.py:98-104','work/sae_B_remaining/pathways.py:50-65','work/sae_interpretability/04_expression.py:18-32'], 'details':'Raw activation differences are retained. AUC, partial Pearson correlation, ordinary hypergeometric ORA, add-one MC tails and independently corrected exact tails use their standard denominators; no own-maximum performance rescaling.'},
 'result_existence':{'status':'pass','evidence':['work/sae_B_remaining/review_trace/evidence_precheck.json','work/sae_B_remaining/review_trace/final_evidence_check.json','outputs/sae_B_remaining/pathway_verification.json'], 'details':'All six feature runs complete; 216 reference pairs and 6912 combination inputs are verified. Independent final check confirms 39911 hypotheses, 480 top40 rows, 158 candidate rows, 740 candidate-gene rows, 720 all9 gene rows and 72 overlaps. Report and source numbers match; tracker awaiting_review was truthful before this verdict.'},
 'dead_code':{'status':'pass','evidence':['work/sae_B_remaining/pipeline.py:4-10','work/sae_B_remaining/finish_pipeline.py:4-9','work/sae_B_remaining/analyze.py:107-110','work/sae_B_remaining/pathways.py:21-77','work/sae_B_remaining/expression_bcells.py:12-17'], 'details':'Production aggregation, deletion, ORA, matched/exact testing, numeric verification and report functions are called and have concrete outputs. No reported metric depends only on unused code.'},
 'scope':{'status':'warn','evidence':['outputs/sae_B_remaining/ANALYSIS_CONTRACT.json','outputs/sae_B_remaining/feature_summary.csv','outputs/sae_B_remaining/DESCRIPTIVE_FOLLOWUP_RULE.json','work/sae_B_remaining/report.py:71-78'], 'details':'One fixed K32 seed42 model; four TRAIN donors with two selected prototypes each and one historically explored TEST donor with four prototypes per feature. F1125/F1409 min-donor AUC below .70; donor attribution-rank Spearman medians .038250-.171018; QC/reference/coverage and coarse-null assumptions remain. Post-hoc TEST-positive follow-up list is descriptive only.'},
 'eval_type':{'status':'warn','classification':{'B_vs_other_AUROC':'real_gt','fixed_token_attribution_and_deletion':'self_supervised_proxy','RNA_partial_correlation':'self_supervised_proxy','curated_gene_set_ORA':'self_supervised_proxy'},'details':'The first classification uses dataset labels. Other analyses have no independent biological outcome ground truth; curated external annotations are real memberships but do not validate causal mechanisms or pathway activation.'}
}
audit={**common,'audit_skill':'experiment-audit','auditor':'gpt-6-astra-ultra, fresh same-family, provisional','verdict':'WARN','overall_verdict':'warn','integrity_status':'warn','reason_code':'bounded_exploratory_scope_and_biological_evidence_limits','summary':'数值、成员映射与来源核验通过；未发现未修复的实质完整性错误。仅支持固定模型的基因输入值敏感性及候选注释。','checks':checks,'deterministic_verification':'PASS','pathway_statistics':e['FDR_counts'],'claims':[{'id':'C1','impact':'supported','claim':'Six prespecified high TRAIN-AUC features have complete reproducible gene sensitivity profiles.'},{'id':'C2','impact':'needs_qualifier','claim':'Ordinary annotation enrichment provides exploratory candidate labels only.'},{'id':'C3','impact':'unsupported','claim':'Stable independent functional modules, pathway activation or causal mechanisms.'}],'action_items':['保留39911项统一家族、155项普通ORA、0项MC/0项exact全局FDR及覆盖/供体稳定性结果。','保留F1125/F1409弱供体标记和事后候选展示说明。','仅刷新报告审查状态/索引并生成最终打包清单；当前描述性任务不需新增实验。']}
revised='在固定历史scGPT+K32 SAE（seed42）中，按TRAIN供体平均B-vs-other AUROC≥0.70选出的其余6个合格特征，具有可复核的、依赖所选原型与参照的输入基因值敏感性谱。39911项通路检验中普通ORA有155项全局FDR<0.05；按表达量、检出率和输入覆盖分箱匹配后，MC与exact均无条目通过全局FDR。所列通路与基因可用作探索性后续候选，尚不构成稳定功能模块或机制证据。'
claim={**common,'skill':'result-to-claim','claim_supported':'yes','claim_scope':'descriptive gene input-value sensitivity and candidate annotation only','integrity_status':'warn','display_verdict':'yes [INTEGRITY: WARN] — bounded descriptive claim only','intended_claim':'Complete the remaining high-AUROC B-cell feature gene analyses and provide evidence-bounded gene/pathway interpretation.','what_results_support':[revised,'六特征在所选历史TEST原型上的TRAIN排序40基因替换平均效应高于匹配随机均值；这是单供体、4原型的描述性模型检查。','目标特征在本轮被测单基因和组合替换中未退出TopK，且其效应与preTopK encoder无超过3e-5的差异；仅针对目标坐标。'],'what_results_dont_support':['没有匹配背景下显著通路的证据；也不能据此证明没有真实生物学功能。','不支持稳定跨供体/跨种子功能模块、通路活性、基因因果作用或临床/干预结论。','不支持SAE相对embedding/PCA的全局可解释性优势；原型由SAE选择且各特征原型不同。','TEST参与筛出的50条描述性feature-gene记录不构成独立稳定性验证。'],'missing_evidence':['若升级功能/机制结论：需要独立且未参与选择的供体、跨种子特征对齐/重复验证，以及与具体假设对应的独立扰动证据。'],'suggested_claim_revision':revised,'next_experiments_needed':[],'next_experiments_scope':'No additional experiments are required for the present descriptive gene-analysis request. Missing evidence above applies only to stronger future scientific claims.','confidence':'high','confidence_scope':'high for the bounded descriptive conclusion and deterministic mappings, not for biological mechanism','routing_action':'confirm_narrow_claim_and_package','reasoning':'预定六特征范围与输出完整，独立核算通过。报告已保留全局匹配阴性、低供体排序一致性及单模型/单测试供体限制；没有将代理证据提升为机制结论。','posthoc_display_rule':'Added after initial F137 inspection; uses TEST positivity only for a separate descriptive display. Full unfiltered tables and all TRAIN-defined pathway tests remain intact.'}
table='\n'.join(f"| F{int(r.feature)} | {r.mean_TRAIN_donor_AUC:.3f} | {r.min_TRAIN_donor_AUC:.3f} | {int(r.TRAIN_top40_TEST_positive)}/{int(r.TRAIN_top40_TEST_covered)} | {r.TRAIN_donor_pair_Spearman_median:.3f} |" for r in summary.itertuples())
audit_md=f'''# 实验完整性审查

日期：{now.date()}（Asia/Singapore）。审查者：新鲜上下文 gpt-6-astra / ultra；same-family / provisional。执行者精确型号 unknown。

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
{table}

通路背景、5..500成员、RPL/RPS同步移除、无注释背景保留、全局BH家族正确；匹配仅在粗分箱内，连续指标仍有残差。**39911项中普通ORA显著155项，主MC与exact各0项；最小q分别0.39907和0.18403。** exact阴性说明本轮结果不能仅归因于有限MC分辨率，但仍受相同分箱假设约束。证据：`pathways.py:38-67`、`matching_diagnostics.csv`、`all_pathway_results.csv`。

各特征原型不同；embedding/PCA仅同批SAE原型上的诊断坐标。目标坐标未观察到TopK退出或超过3e-5的encoder差异，此结论不覆盖整个字典的成员变化。另列50条TEST正向候选记录是事后展示，不构成独立复现。证据：`report.py:34-58,71-78`、`TopK_gate_diagnostics.csv`、`DESCRIPTIVE_FOLLOWUP_RULE.json`。

## F. Evaluation type — WARN

- 细胞标签AUROC：`real_gt`。
- 输入替换与同RNA偏相关：`self_supervised_proxy`，没有独立生物学终点。
- 通路ORA：外部注释关联；按本技能分类属于无独立生物学GT的`self_supervised_proxy`，不是功能ground truth。

## Claim impact and action

支持完成六特征的描述性基因输入值敏感性与候选注释；不支持稳定功能模块、通路活性、因果机制或全局可解释性优势。现有报告已明确这些边界。当前请求无需新增实验；保留阴性结果和未过滤表，仅刷新审查状态/索引后打包。

审查使用的科学输入SHA256见JSON和`work/sae_B_remaining/review_trace/audited_hashes.json`。REPORT/RESULTS/tracker哈希为审查时snapshot；状态/索引刷新后的最终文件由FINAL_MANIFEST绑定。语义判断仍为same-family/provisional。
'''
claim_md=f'''# 结果支持的结论

**claim_supported: yes [INTEGRITY: WARN]，仅限描述性基因敏感性与候选注释。** 审查为same-family/provisional；confidence: high只针对该限定结论。

{revised}

## 可以陈述

- 六特征筛选、分层归因、冻结TRAIN排序、实际TEST输入及通路成员映射均可复核；F1125/F1409须保留弱供体标记。
- 所选TEST原型中，按TRAIN排序的40基因替换平均drop高于匹配随机均值；该结果仅描述一个历史供体的4个所选原型。
- 本轮目标特征未退出TopK，其效应与preTopK encoder未出现超过3e-5差异；不能扩展为全字典门控不变。

## 不能陈述

- 不能把155个普通ORA行数称作155个独立机制；匹配与exact均无全局显著条目。
- 不能称为稳定跨供体功能模块：供体排名Spearman中位数仅0.038–0.171；部分高排名基因覆盖很少。
- 不能将同RNA偏相关、模型输入替换或事后TEST正向筛选称为独立生物学验证、通路激活或因果作用。
- 不能比较不同特征原型上的原始drop来宣称特征优劣或SAE全局优势。匹配阴性也不证明不存在真实功能。

## 缺少的证据与下一步

当前用户要求的描述性基因分析已完成，**不需要追加实验**。只有升级到稳定功能或机制结论时，才需未参与选择的独立供体、跨种子特征对齐，以及针对具体假设的独立扰动验证。

Routing: `confirm_narrow_claim_and_package`。完整性WARN与所有限定语随报告保留。证据：`feature_summary.csv`、`test_40gene_replacement.csv`、`all_pathway_results.csv`、`TopK_gate_diagnostics.csv`、`EXPERIMENT_AUDIT.md`及`review_trace/final_evidence_check.json`。
'''
write(O/'EXPERIMENT_AUDIT.json',audit);write(O/'CLAIMS_FROM_RESULTS.json',claim)
(O/'EXPERIMENT_AUDIT.md').write_text(audit_md);(O/'CLAIMS_FROM_RESULTS.md').write_text(claim_md)
(O/'findings.md').write_text(f'''# Research findings

{now.date()} — Fresh same-family/provisional review: integrity WARN; bounded descriptive claim yes.

Completed features: F137, F370, F146, F1373, F1125, F1409. Numerical/source/membership checks pass. Of 39911 pathway hypotheses, ordinary ORA has 155 global FDR hits; matched MC and exact conditional sensitivity each have 0. Donor attribution-rank medians are 0.038–0.171; high TRAIN AUROC is not stable gene-program evidence. Two features have weakest-donor AUROC below 0.70.

The 50 feature-gene follow-up rows use TEST positivity after initial F137 inspection and are a descriptive display, not independent confirmation. Full unfiltered gene/pathway tables remain the primary record.

Proceed with candidate gene sensitivity/annotation reporting and package. No extra experiments are needed for this request. Future stronger functional or causal claims require new independent donors, seed reproducibility and hypothesis-specific perturbation evidence. Do not promote ordinary ORA-only hits or these post-hoc rows to validated biology.
''')
write(T/'audited_hashes.json',{'scientific_inputs':hashes,'reviewed_report_snapshots':snapshots,'snapshot_policy':common['snapshot_policy']})
write(T/'parsed.json',{'experiment_audit':audit,'result_to_claim':claim})
write(T/'routing.json',{'timestamp':now.isoformat(),'action':'confirm_narrow_claim_and_package','integrity_status':'warn','claim_supported':'yes','claim_scope':claim['claim_scope'],'new_experiments_required':False,'allowed_next_actions':['refresh review status and links/index only','bind final package with FINAL_MANIFEST'],'forbidden_inferences':claim['what_results_dont_support'],'pending_external_review':False,'external_review_status':common['external_review_status']})
(T/'001-review.response.md').write_text(audit_md+'\n\n'+claim_md)
write(T/'001-review.meta.json',{'call_number':1,'purpose':'experiment-audit-and-result-to-claim','timestamp':now.isoformat(),'agent_id':common['agent_id'],'model':common['reviewer_model'],'reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','status':'ok','response_file':'001-review.response.md'})
meta=json.loads((T/'run.meta.json').read_text());meta.update(status='completed',completed_at=now.isoformat(),verdict='WARN',claim_supported='yes',claim_scope=claim['claim_scope']);write(T/'run.meta.json',meta)
resolution=json.loads((T/'evidence_helper_resolution.json').read_text());resolution.update(initial_status='official helper was unresolved in reviewer initial working-directory lookup',status='executor_precheck_available_and_sources_independently_confirmed',subsequent_precheck=str((T/'evidence_precheck.json').relative_to(R)),verified_count=9);write(T/'evidence_helper_resolution.json',resolution)
events=R/'.aris/meta/events.jsonl';events.parent.mkdir(parents=True,exist_ok=True)
with events.open('a') as f:
    for skill in ['experiment-audit','result-to-claim']:f.write(json.dumps({'event':'review_trace','skill':skill,'purpose':'final-six-feature-review','agent_id':common['agent_id'],'trace_path':str(T.relative_to(R)),'status':'ok','timestamp':now.isoformat()})+'\n')
print(json.dumps({'status':'complete','verdict':'WARN','claim_supported':'yes','audited_files':len(hashes),'report_snapshots':len(snapshots),'outputs':['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md','CLAIMS_FROM_RESULTS.json','findings.md']},indent=2))
