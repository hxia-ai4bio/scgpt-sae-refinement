from pathlib import Path
import json,hashlib
import pandas as pd
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_pathways_v3'
p=W/'gene_sets/c5.go.bp.v2025.1.Hs.json';data=json.loads(p.read_text());rows=[]
for line in (W/'gene_sets/c5.go.bp.v2025.1.Hs.symbols.gmt').read_text().splitlines():
 a=line.split('\t');v=data[a[0]];assert set(a[2:])==set(v['geneSymbols']);rows.append({'term':a[0],'GO_ID':v['exactSource'],'GO_URL':v['externalDetailsURL'],'MSigDB_URL':v['msigdbURL']})
m=pd.DataFrame(rows);assert m.term.is_unique;m.to_csv(O/'go_term_metadata.csv',index=False)
d=pd.read_csv(O/'all_pathway_results.csv').merge(m,on='term',how='left');d.to_csv(O/'all_pathways_with_GO_ID.csv',index=False)
(O/'go_metadata_provenance.json').write_text(json.dumps({'file':p.name,'url':'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'GO_terms':len(m),'all_GMT_memberships_equal_JSON':True},indent=2))
print('GO_IDS_VERIFIED',len(m))
