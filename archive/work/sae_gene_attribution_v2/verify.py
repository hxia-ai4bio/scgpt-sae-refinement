"""Independent saved-input, aggregation, source-hash and baseline checks."""
import run as r
from run import *

def main():
 f,pairs,sources=r.setup();source_manifest=json.loads((r.OLD.parents[1]/'outputs/sae_interpretability/input_manifest.json').read_text())
 for path,h in source_manifest['sha256'].items():assert sha(path)==h,path
 dm=f.manifest
 for path,h in dm['sources'].items():assert sha(path)==h,path
 z=load_npz(OLD/'primary_activations.npz')['kang']; baseline_err=[];replays=[];group={};exact=0
 ci=f.manifest['classes'].index('B cells')
 for p in pairs:
  dest=BASE/'cache'/('pair_%02d.npz'%p['pair_id']);a=load_npz(dest);binding=json.loads(dest.with_suffix('.json').read_text());assert binding['sha256']==sha(dest);assert binding['identity']['sources']==sources
  pi,ri=p['prototype_index'],p['reference_index'];assert f.k['labels'][pi]==ci and f.k['labels'][ri]==ci;assert f.k['donors'][pi]==f.k['donors'][ri]==p['donor'];assert z[pi,160]>0 and z[ri,160]==0
  allowed=set(dm['splits'][0]['indices'][p['role']]);assert pi in allowed and ri in allowed
  b=f.c.tokenize([p['prototype_raw_row']]);pos=np.flatnonzero(~b['padding'].numpy()[0]);cols=np.array([f.vocab_to_col[int(b['genes'][0,j])] for j in pos]);ref=f.full_bins(p['reference_raw_row'])[cols]
  assert np.array_equal(pos,a['positions']);assert np.array_equal(cols,a['columns']);assert np.array_equal(b['values'].numpy()[0,pos],a['original_bins']);assert np.array_equal(ref,a['reference_bins']);assert np.array_equal(np.flatnonzero(a['original_bins']!=ref),a['changed'])
  baseline_err.append(abs(float(a['baseline'][0])-float(z[pi,160])));replays.append(float(a['replay_error']));assert baseline_err[-1]<3e-5
  dense=np.zeros((len(f.genes),4));dense[cols]=a['effects'];group.setdefault(p['role'],{}).setdefault(p['donor'],{}).setdefault(pi,[]).append(dense);exact+=1
 errors=[]
 for role,donors in group.items():
  expected=np.mean([np.mean([np.mean(refs,0) for refs in pro.values()],0) for pro in donors.values()],0)
  df=pd.read_csv(OUT/f'gene_rankings_{role}.csv');idx=[f.gene_col[g] for g in df.gene]
  errors.append(float(np.max(np.abs(df[NAMES].values-expected[idx]))));assert errors[-1]<1e-10
 allrows=pd.read_csv(OUT/'heldout_deletion.csv');deletion_err=[];stratum_checks=0;deletion_inputs=0
 for p in pairs:
  if p['role']!='test':continue
  dest=BASE/'deletion'/('pair_%02d.npz'%p['pair_id']);d=load_npz(dest);meta=json.loads(dest.with_suffix('.json').read_text());assert sha(dest)==meta['sha256'];assert meta['rankings_sha256']==sha(OUT/'gene_rankings_train.csv')
  a=load_npz(BASE/'cache'/('pair_%02d.npz'%p['pair_id']));b=f.c.tokenize([p['prototype_raw_row']]);orig=b['values'].numpy()[0];posmap={int(v):j for j,v in enumerate(a['positions'])}
  for i,(order,n,actual,positions) in enumerate(zip(d['orders'],d['budgets'],d['actual'],meta['chosen_positions'])):
   assert len(positions)==actual and len(set(positions))==actual
   values=orig.copy()
   for pos in positions:values[pos]=a['reference_bins'][posmap[pos]]
   assert np.array_equal(values,d['values'][i]);deletion_inputs+=1
   for j,name in enumerate(NAMES):
    row=allrows[(allrows.pair_id==p['pair_id'])&(allrows.order==order)&(allrows.requested_genes==n)&(allrows.score==name)];assert len(row)==1
    deletion_err.append(abs(float(row['drop'].iloc[0])-float(d['baseline'][j]-d['scores'][i,j])))
   if str(order).startswith('matched_random'):
    source_i=next(q for q,(o,nn) in enumerate(zip(d['orders'],d['budgets'])) if o=='sae_F160' and nn==n);source=meta['chosen_positions'][source_i];assert len(source)==len(positions)
    def key(pos):
     j=posmap[pos];delta=a['reference_bins'][j]-a['original_bins'][j];return (int(a['original_bins'][j])//10,int(np.sign(delta)),int(abs(delta))//5)
    assert [key(pos) for pos in source]==[key(pos) for pos in positions];stratum_checks+=1
 assert max(deletion_err)<1e-6
 js(OUT/'verification.json',{'status':'PASS','source_hashes_unchanged':True,'exact_occlusion_inputs_checked':exact,'baseline_cache_max_error':max(baseline_err),'slow_backend_replays_from_run':len(replays),'slow_backend_max_error':max(replays),'gene_aggregation_max_error':max(errors),'deletion_inputs_checked':deletion_inputs,'joint_random_stratum_sets_checked':stratum_checks,'deletion_csv_max_error':max(deletion_err),'scope':'numerical/input/provenance verification, not biological validation'})
 print('VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
