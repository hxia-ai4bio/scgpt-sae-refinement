from pathlib import Path
import sys,json,hashlib,zipfile,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/sae_gene_attribution_v2';OUT=ROOT/'outputs/sae_gene_attribution_v2'
text='''B细胞 SAE F160 多参照归因分析包

先阅读REPORT.html或RESULTS_ZH.txt。
gene_rankings_train.csv / gene_rankings_test.csv：各输出坐标在实际覆盖的输入基因范围内的完整归因，包含负贡献、输入覆盖次数、供体正向数和条件均值。
gene_evidence_combined.csv：训练/测试归因、上一轮表达偏相关的合并表；二者均来自同源RNA，不是独立验证。
per_prototype_gene_attribution.csv：每原型平均参照后的归因。
reference_sensitivity.csv / train_donor_consistency.csv：换参照、换供体的描述性一致性。
heldout_deletion.csv：锁定TRAIN排序后的逐对、逐步原始激活和下降。
pairs_with_qc.csv：配对细胞身份及技术差异。
verification.json：输入、数值、原始文件hash与前向重放核验。
EXPERIMENT_AUDIT.* / CLAIMS_FROM_RESULTS.*：独立fresh-agent审查，同模型家族暂定判断。

本包不是独立数据/模型发布。原始RNA、scGPT/SAE权重、历史项目代码仍需保留在原机器路径。相关文件和hash见input_precheck.json。不要将选定原型的输入敏感性理解成真实细胞基因敲除效果。

目录布局必须保留work/和outputs/的相对关系。附带上一轮依赖脚本、缓存和结果，原始大数据/权重不复制。
原解释器：/opt/anaconda3/envs/scgpt/bin/python。
运行顺序（在独立副本中运行，避免覆盖本轮结果）：run.py → pairs_qc.py → analyze.py → verify.py → report.py。
建议环境：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1；MPLCONFIGDIR指向可写目录。
run.py会检查已存在缓存的输入与源代码绑定。改变输入/生产代码后应使用新运行目录，不能跳过验证复用缓存。
不要通过脚本重新生成科研审查结论；审查是单独的人工/代理检查步骤。
FINAL_MANIFEST.json列出封装时所有文件的SHA256。原始依赖的绝对路径仍是原机器路径，移动机器需显式迁移并重新做输入核验。
'''
(OUT/'REPRODUCIBILITY.txt').write_text(text)
vers={'python':sys.version,'packages':{}}
for name in ['numpy','pandas','scipy','torch','torchtext','scikit-learn','matplotlib','h5py','anndata','scanpy','numba']:
 try:vers['packages'][name]=md.version(name)
 except md.PackageNotFoundError:vers['packages'][name]=None
(OUT/'ENVIRONMENT.json').write_text(json.dumps(vers,indent=2))
if '--archive' in sys.argv:
 paths=[]
 for d in ['work/sae_gene_attribution_v2','outputs/sae_gene_attribution_v2','work/sae_interpretability','outputs/sae_interpretability']:
  for p in sorted((ROOT/d).rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['__pycache__','matplotlib_cache','numba_cache']):continue
   if p==OUT/'FINAL_MANIFEST.json':continue
   paths.append(p)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};(OUT/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));paths.append(OUT/'FINAL_MANIFEST.json')
 target=ROOT/'outputs/SAE_Bcell_gene_attribution_v2_20261018.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print(json.dumps({'archive':str(target),'files':len(paths),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
