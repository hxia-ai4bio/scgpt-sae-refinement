from pathlib import Path
import json,hashlib,zipfile,sys,platform,importlib.metadata
W=Path(__file__).resolve().parent;R=W.parents[1];O=R/'outputs/sae_B_remaining'
(O/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','statsmodels','h5py','psutil','matplotlib']}},indent=2))
(O/'REPRODUCIBILITY.txt').write_text('''Remaining B-cell high-AUROC feature gene analysis
Features137,370,146,1373,1125,1409: all eligible TRAIN mean B-vs-other donor AUC>=0.70, excluding the three already analyzed. The last two have min TRAIN donor AUC<0.70 and are labeled separately.
Read REPORT.html, top40_gene_contributions.csv, candidate_pathway_genes.csv. All scores and negative effects remain in each feature's full gene_rankings_train/test.csv. all9 files compare existing feature-selected prototypes descriptively; old3features are not new matched experiments.

Full inference requires external original datasets/checkpoints/project modules identified by work/sae_interpretability/common.py and historical manifests. These are not bundled. Source paths/hashes and frozen activations are retained. Full production used MPS float32 batch16, same model and fixed tokens as prior CPU analysis; CPU slow replay of each reference pair largest-effect input and maximal combination is recorded. A normal sandbox may hide the MPS device; genuine device access is needed. CPU-only verification of cached outputs remains possible.
Python versions: ENVIRONMENT.json.
From extracted root:
 python work/sae_B_remaining/context.py
 python work/sae_B_remaining/expression_bcells.py
 For each feature, set SAE_FEATURE to its integer ID and run work/sae_B_remaining/run.py -> analyze.py -> pairs_qc.py -> verify.py (or pipeline.py on GPU).
 python work/sae_B_remaining/pathways.py
 python work/sae_B_remaining/verify_pathways.py
 python work/sae_B_remaining/report.py
Use OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=8 for model stages; code sets Torch threads8 and max16perturbed cells perbatch.
Saved reference caches bind original inputs, checkpoint/activation/producer hashes and protocol. recovery/ records a metadata-path-only fix after the first F137 run; no inference or selection change. The first F137 cache was validated and reused after that fix. frozen_sources.json includes the corrected analyzer; frozen_inference_sources.json preserves the original snapshot and recovery/amendment.json explains the difference.

GO BP/Reactome/Hallmark MSigDB2025.1.Hs. Primary global BH covers all6 new features x2lists x3libraries. Ordinary-only rows are not matched discoveries.10,000random sets per list; exact stratum-hypergeom convolution is separate prespecified sensitivity. Strata/indices are saved in null_*.npz; replay them directly instead of rebuilding bins from rounded CSV metrics.
DESCRIPTIVE_FOLLOWUP_RULE.json documents posthoc display flags, which do not alter rankings or pathway tests and are not independent validation.
Original models/raw data and imported historical project modules remain external; archive contains local helper sources, fixed GMT/GO JSON, source frozen activations, PCA, prior3gene tables, current caches/results, and reviewer evidence. Standalone reports are readable without model dependencies. Numerical/audit checks are not biological mechanism validation.
''')
if '--prepare-only' in sys.argv:sys.exit(0)
paths=[p for base in [W,O,R/'work/sae_pathways_v3/gene_sets'] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_MANIFEST.json']
paths += [R/'work/sae_interpretability'/n for n in ['common.py','02_forward.py','04_expression.py','primary_activations.npz','qc.npz','pca.npz']]
paths += [R/'outputs/sae_interpretability'/n for n in ['candidates.json','baseline_directions.json','input_manifest.json']]
paths += [R/'outputs/sae_gene_attribution_v2/gene_evidence_combined.csv']+[R/f'outputs/sae_B898_1387/F{fid}/gene_evidence_combined.csv' for fid in [898,1387]]
paths=sorted(set(paths));(O/'FINAL_MANIFEST.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2));paths.append(O/'FINAL_MANIFEST.json');z=R/'outputs/SAE_Bcells_remaining_high_AUROC_gene_analysis.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in paths:f.write(p,p.relative_to(R))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
print(json.dumps({'archive':str(z),'files':len(paths),'bytes':z.stat().st_size,'CRC':'PASS'}),flush=True)
