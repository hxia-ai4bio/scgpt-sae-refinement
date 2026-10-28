"""Model-input occlusion with locked train/test prototype selection and raw outputs.

Streaming batches use O(B*s*d) activation memory. N selected cells require at
most N*s forwards. All held-out deletion gene orders come from training cells.
"""
import importlib.util,time,json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from common import *
spec=importlib.util.spec_from_file_location('forward_impl',HERE/'02_forward.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def normalized_row(f,row):
    raw=f.x[row].toarray().ravel().astype(np.float64)
    return np.log1p(raw/max(raw.sum(),1)*1e4)

def select_pairs(f,z):
    split=f.manifest['splits'][0]; classes=f.manifest['classes']; k=f.k
    pairs=[]
    for cidx,c in enumerate(f.candidates):
      feature=c['feature']; target=c['target']
      for role in ['train','test']:
        indices=np.asarray(split['indices'][role]); groups=split['train'] if role=='train' else [split['test']]
        for donor in groups:
          pool=indices[k['donors'][indices]==donor]
          if target!='all':pool=pool[k['labels'][pool]==classes.index(target)]
          pool=pool[z[pool,feature]>0]
          proto=sorted(pool,key=lambda i:(-float(z[i,feature]),int(i)))[:1 if role=='train' else 2]
          for pi in proto:
            full=indices[(k['donors'][indices]==donor)&(z[indices,feature]==0)]
            same=full[k['labels'][full]==k['labels'][pi]]
            cfpool=same if len(same) else full
            if not len(cfpool): raise RuntimeError('No inactive same-donor counterfactual')
            delta=k['embedding'][cfpool].astype(np.float64)-k['embedding'][pi]
            dist=np.einsum('ij,ij->i',delta,delta); ci=int(cfpool[np.argmin(dist)])
            pairs.append({'pair_id':len(pairs),'candidate_index':cidx,'feature':feature,'target':target,
              'role':role,'donor':donor,'prototype_index':int(pi),'reference_index':ci,
              'prototype_raw_row':int(k['rows'][pi]),'reference_raw_row':int(k['rows'][ci]),
              'prototype_cell_id':str(k['cell_ids'][pi]),'reference_cell_id':str(k['cell_ids'][ci]),
              'prototype_type':classes[int(k['labels'][pi])],'reference_type':classes[int(k['labels'][ci])],
              'same_type_reference':bool(len(same)),'reference_distance':float(np.sqrt(dist.min())),
              'cached_prototype_activation':float(z[pi,feature]),'cached_reference_activation':float(z[ci,feature])})
    return pairs

def main():
    start=time.time(); f=mod.Forward(); torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(True)
    z=load_npz(HERE/'primary_activations.npz')['kang']; pairs=select_pairs(f,z)
    if (OUT/'prototype_pairs.json').exists():
      if json.loads((OUT/'prototype_pairs.json').read_text())!=pairs:
        raise RuntimeError('Frozen prototype pairs changed; use a new output directory')
    else:js(OUT/'prototype_pairs.json',pairs)
    resultdir=HERE/'occlusion';resultdir.mkdir(exist_ok=True)
    # Backend parity on real selected prototypes, before any substantive attribution.
    pb=f.c.tokenize(np.array([p['prototype_raw_row'] for p in pairs[:4]]))
    torch.backends.mha.set_fastpath_enabled(False); slow=f.scores(pb)
    torch.backends.mha.set_fastpath_enabled(True); fast=f.scores(pb)
    parity=float(np.abs(slow-fast).max()); assert parity<3e-5
    js(OUT/'occlusion_backend.json',{'threads':8,'batch_size':16,'fastpath':True,'real_prototype_score_max_diff':parity})
    for pair in pairs:
      dest=resultdir/('pair_%02d.npz'%pair['pair_id'])
      if dest.exists():
        binding=dest.with_suffix('.binding.json')
        if not binding.exists():
          raise RuntimeError('Unbound legacy cache: run 09_verify.py before reuse')
        bound=json.loads(binding.read_text())
        expected={'pair':pair,'sae_sha256':sha(f.sae_path),'protocol_sha256':sha(OUT/'protocol.json'),
                  'source_manifest_sha256':sha(OUT/'input_manifest.json'),'cache_sha256':sha(dest),
                  'consumer_script_sha256':sha(Path(__file__)),'common_script_sha256':sha(HERE/'common.py'),
                  'forward_script_sha256':sha(HERE/'02_forward.py')}
        if any(bound.get(key)!=val for key,val in expected.items()):
          raise RuntimeError('Cache identity/hash mismatch: '+str(dest))
        continue
      t=time.time(); base=f.c.tokenize([pair['prototype_raw_row']]); orig=base['values'].numpy()[0]
      pos=np.flatnonzero(~base['padding'].numpy()[0]); cols=np.array([f.vocab_to_col[int(base['genes'][0,j])] for j in pos])
      cf=f.full_bins(pair['reference_raw_row'])[cols]; baseline=f.scores(base)[0]
      assert abs(float(baseline[pair['candidate_index']])-pair['cached_prototype_activation'])<3e-5
      changed=np.flatnonzero(orig[pos]!=cf); effects=np.zeros((len(pos),len(baseline)),np.float32)
      for offset in range(0,len(changed),16):
        ci=changed[offset:offset+16]; vals=np.repeat(orig[None,:],len(ci),axis=0)
        vals[np.arange(len(ci)),pos[ci]]=cf[ci]
        effects[ci]=baseline-f.scores(f.batch(base,vals))
      np.savez_compressed(dest,columns=cols,positions=pos,reference_bins=cf,original_bins=orig[pos],
        effects=effects,baseline=baseline,changed=changed)
      js(dest.with_suffix('.binding.json'),{'pair':pair,'sae_sha256':sha(f.sae_path),
          'protocol_sha256':sha(OUT/'protocol.json'),'source_manifest_sha256':sha(OUT/'input_manifest.json'),
          'cache_sha256':sha(dest),'producer_script_sha256':sha(Path(__file__)),
          'consumer_script_sha256':sha(Path(__file__)),'common_script_sha256':sha(HERE/'common.py'),
          'forward_script_sha256':sha(HERE/'02_forward.py')})
      print('OCCLUSION',pair['pair_id'],pair['feature'],pair['role'],'changed',len(changed),
            'sec',round(time.time()-t,1),'elapsed',round(time.time()-start,1),flush=True)
    # Aggregate over all prototypes; a gene absent from a prototype input contributes zero.
    ranks={}; tables=[]; agreement=[]
    for ci,c in enumerate(f.candidates):
      for role in ['train','test']:
        pp=[p for p in pairs if p['candidate_index']==ci and p['role']==role]
        effects=np.zeros((len(pp),len(f.genes))); exposures=np.zeros_like(effects,dtype=np.int8)
        deg=np.zeros(len(f.genes))
        for j,pair in enumerate(pp):
          a=load_npz(resultdir/('pair_%02d.npz'%pair['pair_id']))
          effects[j,a['columns']]=a['effects'][:,ci];exposures[j,a['columns']]=1
          deg+=normalized_row(f,pair['prototype_raw_row'])-normalized_row(f,pair['reference_raw_row'])
        avg=effects.mean(0); count=exposures.sum(0); deg/=len(pp)
        df=pd.DataFrame({'feature':c['feature'],'role':role,'gene':f.genes,'mean_attribution':avg,
            'input_exposures':count,'n_prototypes':len(pp),'mean_logexpr_difference':deg,
            'positive_prototypes':(effects>0).sum(0),'negative_prototypes':(effects<0).sum(0)})
        df=df[count>0].sort_values('mean_attribution',ascending=False,kind='stable')
        df.to_csv(OUT/('gene_attribution_%s_%s.csv'%(c['feature'],role)),index=False)
        tables.append(df)
        if role=='train':ranks[c['feature']]={'attr':avg,'deg':np.abs(deg),'exposed':count>0}
        # Per-prototype consistency only over co-observed genes; descriptive, not independent replicates.
        for i in range(len(pp)):
          for j in range(i+1,len(pp)):
            common=(exposures[i]>0)&(exposures[j]>0)
            rho=spearmanr(effects[i,common],effects[j,common]).statistic if common.sum()>2 else np.nan
            agreement.append({'feature':c['feature'],'role':role,'pair_i':pp[i]['pair_id'],'pair_j':pp[j]['pair_id'],'shared_genes':int(common.sum()),'spearman':float(rho) if np.isfinite(rho) else None})
    pd.DataFrame(agreement).to_csv(OUT/'attribution_consistency.csv',index=False)
    # Frozen TRAIN gene orders applied to TEST cells; ten matched-random orders.
    curves=[]
    for pair in [p for p in pairs if p['role']=='test']:
      ci=pair['candidate_index']; feature=pair['feature']; a=load_npz(resultdir/('pair_%02d.npz'%pair['pair_id']))
      base=f.c.tokenize([pair['prototype_raw_row']]); orig=base['values'].numpy()[0]
      # All arms share eligible genes and only change values that actually differ.
      eligible=np.array([i for i,col in enumerate(a['columns']) if ranks[feature]['exposed'][col] and a['original_bins'][i]!=a['reference_bins'][i]])
      orders={}
      orders['attribution']=np.array(sorted(eligible,key=lambda i:(-ranks[feature]['attr'][a['columns'][i]],int(a['columns'][i]))))
      orders['expression_difference']=np.array(sorted(eligible,key=lambda i:(-ranks[feature]['deg'][a['columns'][i]],int(a['columns'][i]))))
      for rep in range(10):
        rng=np.random.default_rng([20261018,pair['pair_id'],rep]); left=set(eligible.tolist()); order=[]
        for i in orders['attribution'][:40]:
          matched=sorted(j for j in left if int(a['original_bins'][j])//10==int(a['original_bins'][i])//10)
          if not matched:matched=sorted(left)
          if not matched:break
          j=int(rng.choice(matched));left.remove(j);order.append(j)
        orders['matched_random_'+str(rep)]=np.array(order,dtype=int)
      inputs=[]; meta=[]
      for name,order in orders.items():
        for n in [0,1,5,10,20,40]:
          chosen=order[:min(n,len(order))]; vals=orig.copy()
          vals[a['positions'][chosen]]=a['reference_bins'][chosen]
          inputs.append(vals);meta.append((name,n,len(chosen)))
      scores=np.concatenate([f.scores(f.batch(base,np.array(inputs[i:i+16]))) for i in range(0,len(inputs),16)])
      for (name,n,actual),score in zip(meta,scores):
        curves.append({'feature':feature,'pair_id':pair['pair_id'],'donor':pair['donor'],
          'order':name,'requested_genes':n,'changed_genes':actual,'raw_activation':float(score[ci]),
          'raw_baseline':float(a['baseline'][ci]),'raw_drop':float(a['baseline'][ci]-score[ci]),
          'scope':'TEST prototypes, TRAIN fixed ranking; within one donor descriptive'})
      print('DELETION_DONE',feature,pair['pair_id'],flush=True)
    pd.DataFrame(curves).to_csv(OUT/'heldout_deletion_curves.csv',index=False)
    # Same-cell baseline sensitivities are exported but are not global interpretability comparisons.
    base_records=[]
    for pair in pairs:
      a=load_npz(resultdir/('pair_%02d.npz'%pair['pair_id']))
      for j,b in enumerate(f.baselines):
        if b['feature']!=pair['feature']:continue
        effect=a['effects'][:,4+j]; order=np.argsort(-effect,kind='stable')[:20]
        for rank,idx in enumerate(order):
          base_records.append({'pair_id':pair['pair_id'],'feature':pair['feature'],'role':pair['role'],
              'baseline_kind':b['kind'],'dimension':b['dimension'],'gene':f.genes[a['columns'][idx]],
              'rank':rank+1,'raw_attribution':float(effect[idx])})
    pd.DataFrame(base_records).to_csv(OUT/'baseline_samecell_attributions.csv',index=False)
    js(OUT/'occlusion_completion.json',{'status':'complete','pairs':len(pairs),'features':len(f.candidates),
      'single_gene_changed_forwards':int(sum(len(load_npz(p)['changed']) for p in resultdir.glob('pair_*.npz'))),
      'deletion_forwards':len(curves),'seconds':time.time()-start,
      'independent_test_donors':1,'references_per_prototype':1,
      'not_paper_exact':'four train donor-stratified prototypes, two test prototypes, one matched reference; fixed token/bin perturbation; ORA annotation instead of GSEA'})
    print('OCCLUSION_COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
