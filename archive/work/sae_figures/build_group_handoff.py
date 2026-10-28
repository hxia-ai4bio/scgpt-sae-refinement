"""Collect small, shareable evidence files; do not copy RNA or model checkpoints."""
from pathlib import Path
import shutil, json, hashlib, csv
ROOT=Path("/Users/apple/Documents/Codex/2026-10-18/wo")
OUT=ROOT/"outputs/SAE_group_handoff"
OUT.mkdir(parents=True,exist_ok=True)
manifest=[]
def cp(src,dest):
    src=Path(src); dest=OUT/dest
    assert src.is_file(),src
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
    manifest.append({"relative_path":str(dest.relative_to(OUT)),"original_path":str(src),
                     "bytes":src.stat().st_size,"sha256":hashlib.sha256(src.read_bytes()).hexdigest()})
figs=ROOT/"outputs/sae_figures"
for folder,target in [("overview_all_celltypes","figure3"),("Figure4_complete","figure4"),("performance","performance")]:
    for p in (figs/folder).iterdir():
        if p.suffix in [".csv",".json",".txt"]:
            cp(p,Path("data")/target/p.name)
for src,name in [
    ("overview_all_celltypes/SAE_celltype_overview_all","Figure3_celltype_associations"),
    ("Figure4_complete/Figure4_Bcell_F160","Figure4_Bcell"),
    ("performance/Secondary_performance_three_panels","Figure5_secondary_performance"),
    ("performance/Classification_and_lambda","Table1_classification_lambda"),
]:
    for ext in [".pdf",".svg"]:
        p=figs/(src+ext)
        if p.exists(): cp(p,Path("figures")/(name+ext))
desktop=Path("/Users/apple/Desktop/6204 Figure")
for src,name in [
    ("Figure1模型流程图.png","Figure1_model.png"),
    ("Figure2可解释性流程图.png","Figure2_interpretability.png"),
    ("Figure3 全细胞的激活值分析.png","Figure3_celltype_associations.png"),
    ("Figure4 Bcell的一些生物学分析.png","Figure4_Bcell.png"),
    ("Figure5 其他性能的比较.png","Figure5_secondary_performance.png"),
    ("Table1.png","Table1_classification_lambda.png"),
]:
    cp(desktop/src,Path("figures")/name)
folders=["sae_lambda","sae_extension","sae_evaluation","sae_masked_ac","sae_expression_probe",
         "sae_checkpoint_selection","sae_checkpoint_extension","sae_gene_attribution_v2","sae_interpretability"]
names={"PROTOCOL.json","PROTOCOL_NOTES.json","DATA_MANIFEST.json","REPRODUCIBILITY.txt",
       "EXPERIMENT_AUDIT.md","CLAIMS_FROM_RESULTS.md","findings.md","summary.csv","seed_metrics.csv",
       "all_metrics.csv","clustering_all.csv","knn_metrics.csv","combined_with_ctrl_metrics.csv",
       "deletion_per_prototype.csv","deletion_summary.csv","gene_rankings_train.csv"}
for folder in folders:
    for p in (ROOT/"outputs"/folder).iterdir():
        if p.is_file() and p.name in names:
            cp(p,Path("evidence")/folder/p.name)
old=Path("/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014")
cp(old/"data_manifest.json","evidence/original/data_manifest.json")
for name in ["sae.py","prepare.py"]:
    cp(old/name,Path("source_reference/original")/name)
for folder,files in {
    "sae_figures":["compute_all_celltypes.py","draw_all_celltypes.py","draw_figure4_complete.py","draw_performance_table.py","draw_secondary_performance.py"],
    "sae_relation":["run.py"],"sae_refine":["run.py"],"sae_complementarity":["core.py"],
    "sae_lambda":["run.py"],"sae_extension":["run.py","prepare.py"],
    "sae_masked_ac":["pipeline.py","prepare.py"],
    "sae_checkpoint_selection":["run.py","execute.py"],
    "sae_checkpoint_extension":["run.py","execute.py"],
    "sae_interpretability":["common.py","01_atlas.py","05_all_folds.py"],
}.items():
    for name in files:
        cp(ROOT/"work"/folder/name,Path("source_reference")/folder/name)
(OUT/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print("Copied",len(manifest),"files;",round(sum(r["bytes"] for r in manifest)/1e6,2),"MB")
