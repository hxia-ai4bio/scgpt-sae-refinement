"""Donor-balanced gene rankings and locked TEST deletion; descriptive, no cell p values."""
import run as r
from run import *
from scipy.stats import spearmanr

def aggregate(f,pairs):
 G=len(f.genes);perproto={};refstats=[];rows=[]
 for p in pairs:perproto.setdefault(p['prototype_index'],[]).append(p)
 proto_arrays={};proto_cover={};proto_meta={}
 for pi,pp in perproto.items():
  dense=[];cover=np.zeros(G,bool)
  for p in pp:
   a=load_npz(BASE/'cache'/('pair_%02d.npz'%p['pair_id']));v=np.zeros((G,4));v[a['columns']]=a['effects'];dense.append(v);cover[a['columns']]=True
  proto_arrays[pi]=np.mean(dense,0);proto_cover[pi]=cover;proto_meta[pi]=pp[0]
  for q in range(len(pp)):
   for s in range(q+1,len(pp)):
    for ci,name in enumerate(NAMES):
     aa=dense[q][cover,ci];bb=dense[s][cover,ci];rho=spearmanr(aa,bb).statistic
     ia=np.flatnonzero(cover);ta=set(ia[np.argsort(-aa)[:20]]);tb=set(ia[np.argsort(-bb)[:20]])
     refstats.append({'prototype_index':pi,'role':pp[0]['role'],'donor':pp[0]['donor'],'score':name,'pair_a':pp[q]['pair_id'],'pair_b':pp[s]['pair_id'],'spearman':float(rho) if np.isfinite(rho) else None,'top20_jaccard':len(ta&tb)/len(ta|tb)})
  for col in np.flatnonzero(cover):
   rows.append({'prototype_index':pi,'role':pp[0]['role'],'donor':pp[0]['donor'],'gene':f.genes[col],**{name:proto_arrays[pi][col,j] for j,name in enumerate(NAMES)}})
 pd.DataFrame(rows).to_csv(OUT/'per_prototype_gene_attribution.csv',index=False)
 pd.DataFrame(refstats).to_csv(OUT/'reference_sensitivity.csv',index=False)
 rankings={};ribo=[];donor_arrays={};consistency=[]
 for role in ['train','test']:
  pis=[pi for pi,pp in proto_meta.items() if pp['role']==role];donors=sorted(set(proto_meta[pi]['donor'] for pi in pis))
  da=[];dc=[]
  for donor in donors:
   ids=[pi for pi in pis if proto_meta[pi]['donor']==donor];da.append(np.mean([proto_arrays[pi] for pi in ids],0));dc.append(np.any([proto_cover[pi] for pi in ids],0));donor_arrays[(role,donor)]=da[-1]
  da=np.array(da);dc=np.array(dc);mean=da.mean(0);coverage=np.sum([proto_cover[pi] for pi in pis],axis=0);used=coverage>0
  df=pd.DataFrame({'gene':f.genes,'input_prototypes':coverage,'total_prototypes':len(pis),'input_donors':dc.sum(0),'total_donors':len(donors)})
  for j,name in enumerate(NAMES):
   df[name]=mean[:,j];df[name+'_positive_donors']=(da[:,:,j]>0).sum(0)
   # conditional mean excludes prototypes where gene was not selected; distinct estimand
   df[name+'_conditional_mean']=np.divide(np.sum([proto_arrays[pi][:,j] for pi in pis],axis=0),coverage,out=np.zeros(G),where=used)
   df[name+'_donor_sd']=da[:,:,j].std(0,ddof=1) if len(donors)>1 else np.nan
   top=np.flatnonzero(used & (mean[:,j]>0));top=sorted(top,key=lambda q:(-mean[q,j],q))[:40]
   ribo.append({'role':role,'score':name,'positive_top_n':len(top),'ribosomal_top_n':sum(str(f.genes[q]).startswith(('RPL','RPS')) for q in top)})
  df=df[used].sort_values(NAMES[0],ascending=False,kind='stable');df.to_csv(OUT/f'gene_rankings_{role}.csv',index=False);rankings[role]=mean
  if role=='train':
   for i in range(len(donors)):
    for j in range(i+1,len(donors)):
     common=dc[i]&dc[j]
     for ci,name in enumerate(NAMES):
      rho=spearmanr(da[i,common,ci],da[j,common,ci]).statistic
      consistency.append({'donor_a':donors[i],'donor_b':donors[j],'score':name,'common_genes':int(common.sum()),'spearman':float(rho) if np.isfinite(rho) else None})
 pd.DataFrame(consistency).to_csv(OUT/'train_donor_consistency.csv',index=False)
 pd.DataFrame(ribo).to_csv(OUT/'ribosomal_summary.csv',index=False)
 # Fixed TRAIN top40 evaluated in TEST, no re-selection for this transfer metric.
 transfer=[]
 for j,name in enumerate(NAMES):
  eligible=np.flatnonzero(rankings['train'][:,j]>0);top=np.array(sorted(eligible,key=lambda c:(-rankings['train'][c,j],c))[:40]);testcov=np.any([proto_cover[pi] for pi in proto_meta if proto_meta[pi]['role']=='test'],0);present=top[testcov[top]]
  transfer.append({'score':name,'train_top_n':len(top),'test_present_n':len(present),'test_positive_n':int((rankings['test'][present,j]>0).sum()),'test_mean_raw_effect_present':float(rankings['test'][present,j].mean()),'train_top_genes':';'.join(f.genes[top])})
 pd.DataFrame(transfer).to_csv(OUT/'train_top40_transfer.csv',index=False)
 np.savez_compressed(BASE/'aggregated.npz',genes=f.genes,train=rankings['train'],test=rankings['test'])
 return rankings

def expr_diff(f,pairs):
 def norm(row):
  v=f.x[row].toarray().ravel().astype(float);return np.log1p(1e4*v/max(v.sum(),1))
 donors={}
 for p in pairs:
  if p['role']!='train':continue
  donors.setdefault(p['donor'],{}).setdefault(p['prototype_index'],[]).append(norm(p['prototype_raw_row'])-norm(p['reference_raw_row']))
 return np.mean([np.mean([np.mean(v,0) for v in pro.values()],0) for pro in donors.values()],0)

def deletion(f,pairs,rankings):
 start=time.time();dd=expr_diff(f,pairs);rows=[];matches=[];inputs_count=0;replays=[]
 (BASE/'deletion').mkdir(exist_ok=True)
 for p in [p for p in pairs if p['role']=='test']:
  a=load_npz(BASE/'cache'/('pair_%02d.npz'%p['pair_id']));b=f.c.tokenize([p['prototype_raw_row']]);original=b['values'].numpy()[0]
  train_exposed=set(pd.read_csv(OUT/'gene_rankings_train.csv').gene)
  eligible=np.array([j for j,col in enumerate(a['columns']) if f.genes[col] in train_exposed and a['original_bins'][j]!=a['reference_bins'][j]],int)
  def sortby(s,idx=eligible):return np.array(sorted(idx,key=lambda j:(-s[a['columns'][j]],int(a['columns'][j]))),int)
  orders={name:sortby(rankings['train'][:,j]) for j,name in enumerate(NAMES)};orders['expression_difference']=sortby(np.abs(dd))
  nonr=np.array([j for j in eligible if not str(f.genes[a['columns'][j]]).startswith(('RPL','RPS'))],int);orders['sae_nonribosomal']=sortby(rankings['train'][:,0],nonr)
  delta=a['reference_bins']-a['original_bins'];strata=[(int(v)//10,int(np.sign(d)),int(abs(d))//5) for v,d in zip(a['original_bins'],delta)]
  for rep in range(10):
   rng=np.random.default_rng([20261019,p['pair_id'],rep]);left=set(eligible.tolist());order=[]
   for source in orders[NAMES[0]][:40]:
    pool=sorted(j for j in left if strata[j]==strata[source]);assert pool
    selected=int(rng.choice(pool));left.remove(selected);order.append(selected)
   orders[f'matched_random_{rep}']=np.array(order,int)
   matches.append({'pair_id':p['pair_id'],'prototype_index':p['prototype_index'],'random_rep':rep,'actual_genes':len(order),'overlap_sae40':len(set(order)&set(orders[NAMES[0]][:40])),'mean_abs_change_random':float(np.abs(delta[order]).mean()),'mean_abs_change_sae':float(np.abs(delta[orders[NAMES[0]][:40]]).mean())})
  values=[];meta=[];chosen_positions=[]
  for name,order in orders.items():
   for n in [0,1,5,10,20,40]:
    ii=order[:n];v=original.copy();v[a['positions'][ii]]=a['reference_bins'][ii];values.append(v);meta.append((name,n,len(ii)));chosen_positions.append(a['positions'][ii].tolist())
  values=np.array(values);ss=np.concatenate([scores(f,f.batch(b,values[i:i+16])) for i in range(0,len(values),16)])
  # Slow backend parity at maximal SAE deletion.
  j=next(i for i,v in enumerate(meta) if v[0]==NAMES[0] and v[1]==40)
  slow=cpu_scores(f,f.batch(b,values[j:j+1]))[0]
  err=float(np.abs(slow-ss[j]).max());assert err<3e-5;replays.append(err)
  dest=BASE/'deletion'/('pair_%02d.npz'%p['pair_id']);np.savez_compressed(dest,values=values,scores=ss,baseline=a['baseline'],orders=np.array([m[0] for m in meta]),budgets=np.array([m[1] for m in meta]),actual=np.array([m[2] for m in meta]))
  js(dest.with_suffix('.json'),{'pair':p,'sha256':sha(dest),'chosen_positions':chosen_positions,'analysis_script_sha256':sha(Path(__file__)),'rankings_sha256':sha(OUT/'gene_rankings_train.csv'),'run_script_sha256':sha(WORK/'run.py')})
  for (name,n,actual),s in zip(meta,ss):
   for ci,score in enumerate(NAMES):rows.append({'pair_id':p['pair_id'],'prototype_index':p['prototype_index'],'donor':p['donor'],'order':name,'requested_genes':n,'actual_genes':actual,'score':score,'baseline':float(a['baseline'][ci]),'activation':float(s[ci]),'drop':float(a['baseline'][ci]-s[ci])})
  inputs_count+=len(values);print('DELETION',p['pair_id'],inputs_count,'elapsed',round(time.time()-start),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'heldout_deletion.csv',index=False);pd.DataFrame(matches).to_csv(OUT/'matched_random_diagnostics.csv',index=False)
 frame['method']=frame.order.str.replace(r'_\d+$','',regex=True)
 # Average random repetitions and refs within each prototype, then average prototypes.
 prot=frame.groupby(['prototype_index','method','requested_genes','score'])[['baseline','activation','drop']].mean().reset_index();prot.to_csv(OUT/'deletion_per_prototype.csv',index=False)
 summary=prot.groupby(['method','requested_genes','score'])[['baseline','activation','drop']].mean().reset_index();summary.to_csv(OUT/'deletion_summary.csv',index=False)
 js(OUT/'analysis_completion.json',{'status':'complete','deletion_forwards':inputs_count,'deletion_seconds':time.time()-start,'slow_backend_replays':len(replays),'max_replay_error':max(replays),'scope':'one historical test donor;4 selected B cells;3 reused references; raw scores not comparable across representations'})

def main():
 assert json.loads((OUT/'occlusion_completion.json').read_text())['status']=='complete'
 f,pairs,sources=r.setup();rankings=aggregate(f,pairs);deletion(f,pairs,rankings)
if __name__=='__main__':main()
