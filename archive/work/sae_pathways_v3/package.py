from pathlib import Path
import json,hashlib,zipfile,platform,sys,importlib.metadata
R=Path(__file__).resolve().parents[2];O=R/'outputs/sae_pathways_v3';W=R/'work/sae_pathways_v3'
env={'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','h5py','psutil','matplotlib']}}
(O/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
(O/'REPRODUCIBILITY.txt').write_text('''F160 GO/pathway analysis — 2026-10-18
Read REPORT.html for conclusions and candidate_pathway_genes.csv for gene mappings.
This is positive-attribution ORA, not GSEA. All 6462 hypotheses and both nulls are retained.
significant_pathways.csv includes ordinary-ORA-only discoveries; it does NOT imply matched-null significance.

Archive layout preserves work/ and outputs/. Included: source scripts, fixed-version GMTs, official GO metadata, saved 20000 matched-null draws, the upstream gene attribution table, reports and audit evidence.
Full source-data rerun requires external kang.h5ad, historical data_manifest.json and kang_embeddings.npz at paths recorded in input_manifest.json. These large/private source files are not bundled. Adjust the OLD path in run.py and the dataset path in the external manifest when relocating. verify.py reads absolute paths from the regenerated input_manifest.json. No checkpoint/model inference is needed for this stage.
Python dependencies and versions: ENVIRONMENT.json.
From the extracted archive root run:
  python work/sae_pathways_v3/run.py
  python work/sae_pathways_v3/verify.py
  python work/sae_pathways_v3/annotate_go.py
  python work/sae_pathways_v3/report.py
The original numerical verification records the original source hashes; editing paths changes script hashes. Source data and upstream attribution are required for fresh validation. Reports and saved result tables can be read without those files.
To recompute the existing matched-null tests, use the strata and indices saved in NPZ; do not recreate bins only from rounded CSV metrics. CSV floating-point round trips alter 8 and 5 boundary assignments, without affecting any sampled pool in this run.
GO IDs were attached only after exact membership equality against official GO JSON was verified.
Audit is an independent-agent, same-family/provisional scientific review; it is not biological validation.
''')
paths=[]
for base in [W,O,R/'work/review_trace/sae_pathways_v3']:
 if base.exists():paths += [p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_MANIFEST.json']
paths += [R/'outputs/sae_gene_attribution_v2/gene_evidence_combined.csv']
paths=sorted(set(paths));manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(O/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));paths.append(O/'FINAL_MANIFEST.json')
z=R/'outputs/SAE_F160_GO_pathways_20261018.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in paths:f.write(p,p.relative_to(R))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
print(json.dumps({'archive':str(z),'files':len(paths),'bytes':z.stat().st_size,'crc':'PASS'}))
