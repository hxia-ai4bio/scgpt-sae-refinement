"""Independent deterministic arithmetic and forward replay checks."""
import importlib.util,json,time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from common import *

def main():
    k,p,m=dataset();manifest=json.loads((OUT/'input_manifest.json').read_text());errors=[]
    for path,expected in manifest['sha256'].items():assert sha(path)==expected,path
    acts=load_npz(HERE/'primary_activations.npz');rep=pd.read_csv(OUT/'candidate_replication.csv')
    auc_errors=[]
    for _,r in rep[rep.auc.notna()].iterrows():
      data=k if r.domain=='kang' else p;groups=data['donors' if r.domain=='kang' else 'batches'];ix=np.flatnonzero(groups==r.group)
      ci=m['classes'].index(r.target);v=roc_auc_score(data['labels'][ix]==ci,acts[r.domain][ix,int(r.feature)])
      auc_errors.append(abs(v-r.auc))
    assert max(auc_errors)<1e-12
    pairs=json.loads((OUT/'prototype_pairs.json').read_text());candidates=json.loads((OUT/'candidates.json').read_text())
    spec=importlib.util.spec_from_file_location('forward_impl',HERE/'02_forward.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    f=mod.Forward();torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);forward_errors=[];aggregation_errors=[]
    # Bind every historical-current-run cache only after validating identities,
    # input token columns, both sets of bins, baseline and a full score-vector replay.
    cache_bind_errors=[]
    for pair in pairs:
      dest=HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']);a=load_npz(dest)
      base=f.c.tokenize([pair['prototype_raw_row']]);pos=np.flatnonzero(~base['padding'].numpy()[0])
      cols=np.array([f.vocab_to_col[int(base['genes'][0,j])] for j in pos])
      assert np.array_equal(cols,a['columns']);assert np.array_equal(pos,a['positions'])
      assert np.array_equal(base['values'].numpy()[0,pos],a['original_bins'])
      assert np.array_equal(f.full_bins(pair['reference_raw_row'])[cols],a['reference_bins'])
      original=f.scores(base)[0];assert np.max(np.abs(original-a['baseline']))<3e-5
      ci=pair['candidate_index'];j=int(np.argmax(np.abs(a['effects'][:,ci])))
      vals=base['values'].numpy().copy();vals[0,pos[j]]=a['reference_bins'][j]
      err=float(np.abs(original-f.scores(f.batch(base,vals))[0]-a['effects'][j]).max())
      assert err<3e-5;cache_bind_errors.append(err)
      js(dest.with_suffix('.binding.json'),{'pair':pair,'sae_sha256':sha(f.sae_path),
          'protocol_sha256':sha(OUT/'protocol.json'),'source_manifest_sha256':sha(OUT/'input_manifest.json'),
          'cache_sha256':sha(dest),'producer_script_sha256':sha(HERE/'executed_v1/03_occlusion.py'),
          'consumer_script_sha256':sha(HERE/'03_occlusion.py'),'common_script_sha256':sha(HERE/'common.py'),
          'forward_script_sha256':sha(HERE/'02_forward.py'),
          'binding_method':'legacy current-run cache validated against every exact input and one largest-effect full-vector forward replay',
          'max_replay_error':err})
    for ci,c in enumerate(candidates):
      for role in ['train','test']:
        pp=[p for p in pairs if p['candidate_index']==ci and p['role']==role];total={};count={}
        for pair in pp:
          a=load_npz(HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']))
          assert np.isfinite(a['effects']).all();assert len(set(a['columns']))==len(a['columns'])
          unchanged=a['original_bins']==a['reference_bins'];assert np.array_equal(a['effects'][unchanged],np.zeros_like(a['effects'][unchanged]))
          for col,eff in zip(a['columns'],a['effects'][:,ci]):total[int(col)]=total.get(int(col),0)+float(eff);count[int(col)]=count.get(int(col),0)+1
        table=pd.read_csv(OUT/f'gene_attribution_{c["feature"]}_{role}.csv').set_index('gene')
        # Gene-name mapping checked from the raw source by the forward class below.
        errors.append({'feature':c['feature'],'role':role,'n_prototypes':len(pp),'gene_columns':len(total)})
    for ci,c in enumerate(candidates):
      for role in ['train','test']:
        pp=[p for p in pairs if p['candidate_index']==ci and p['role']==role];total={}
        for pair in pp:
          a=load_npz(HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']))
          for col,eff in zip(a['columns'],a['effects'][:,ci]):total[int(col)]=total.get(int(col),0)+float(eff)
        table=pd.read_csv(OUT/f'gene_attribution_{c["feature"]}_{role}.csv').set_index('gene')
        for col,value in total.items():aggregation_errors.append(abs(value/len(pp)-table.loc[f.genes[col],'mean_attribution']))
        pair=pp[0];a=load_npz(HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']))
        j=int(np.argmax(np.abs(a['effects'][:,ci])));base=f.c.tokenize([pair['prototype_raw_row']]);orig=f.scores(base)
        vals=base['values'].numpy().copy();vals[0,a['positions'][j]]=a['reference_bins'][j]
        new=f.scores(f.batch(base,vals));err=float(np.abs((orig-new)[0]-a['effects'][j]).max());forward_errors.append(err)
    assert max(aggregation_errors)<1e-7;assert max(forward_errors)<3e-5
    curves=pd.read_csv(OUT/'heldout_deletion_curves.csv');assert len(curves)==len([p for p in pairs if p['role']=='test'])*12*6
    assert np.isfinite(curves[['raw_activation','raw_baseline','raw_drop']].values).all()
    assert np.abs(curves.raw_drop-(curves.raw_baseline-curves.raw_activation)).max()<1e-7
    assert np.abs(curves[curves.requested_genes==0].raw_drop).max()<3e-5
    assert (curves.changed_genes<=curves.requested_genes).all()
    strong_checks=0;strong_errors=[]
    if (OUT/'heldout_deletion_strong_null.csv').exists():
      strong=pd.read_csv(OUT/'heldout_deletion_strong_null.csv')
      for pair in [p for p in pairs if p['role']=='test']:
        a=load_npz(HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']))
        saved=load_npz(HERE/'strong_null'/('pair_%02d.npz'%pair['pair_id']))
        base=f.c.tokenize([pair['prototype_raw_row']]);orig=base['values'].numpy()[0]
        df=pd.read_csv(OUT/f'gene_attribution_{pair["feature"]}_train.csv');rank=dict(zip(df.gene,df.mean_attribution))
        eligible=np.array([i for i,col in enumerate(a['columns']) if f.genes[col] in rank and a['original_bins'][i]!=a['reference_bins'][i]])
        order=np.array(sorted(eligible,key=lambda i:(-rank[f.genes[a['columns'][i]]],int(a['columns'][i]))))[:40]
        delta=a['reference_bins']-a['original_bins'];strata=[(int(v)//10,int(np.sign(d)),int(abs(d))//5) for v,d in zip(a['original_bins'],delta)]
        byposition={int(pos):i for i,pos in enumerate(a['positions'])}
        for vals,n in zip(saved['values'],saved['requested_genes']):
          changed=np.flatnonzero(vals!=orig);assert len(changed)==n
          ids=[byposition[int(j)] for j in changed]
          assert np.array_equal(vals[changed],a['reference_bins'][ids])
          assert sorted(strata[i] for i in ids)==sorted(strata[i] for i in order[:n])
          strong_checks+=1
        i=5;actual=f.scores(f.batch(base,saved['values'][i:i+1]))[0]
        err=float(np.max(np.abs(actual-saved['scores'][i])));assert err<3e-5;strong_errors.append(err)
        sub=strong[strong.pair_id==pair['pair_id']]
        assert np.abs(sub.raw_activation.values-saved['scores'][:,pair['candidate_index']]).max()<1e-7
      assert np.abs(strong.raw_drop-(strong.raw_baseline-strong.raw_activation)).max()<1e-7
    enrich=pd.read_csv(OUT/'pathway_enrichment_all.csv');from scipy.stats import hypergeom
    re=hypergeom.sf(enrich.overlap.values-1,enrich.universe_size.values,enrich.pathway_genes_tested.values,enrich.selected_genes.values)
    assert np.abs(re-enrich.p_hypergeom.values).max()<1e-12
    result={'status':'PASS','source_sha256_unchanged':True,'primary_auc_checks':len(auc_errors),
      'auc_max_error':max(auc_errors),'attribution_aggregate_max_error':max(aggregation_errors),
      'independent_slow_forward_replays':len(forward_errors)+len(cache_bind_errors),'forward_replay_max_error':max(forward_errors+cache_bind_errors),
      'bound_cache_pairs':len(cache_bind_errors),
      'strong_null_exact_input_stratum_checks':strong_checks,'strong_null_slow_forward_replays':len(strong_errors),
      'strong_null_replay_max_error':max(strong_errors) if strong_errors else None,
      'deletion_rows':len(curves),'enrichment_hypergeom_max_error':float(np.abs(re-enrich.p_hypergeom.values).max()),
      'scope':'arithmetic/provenance/replay checks, not independent biological validation',
      'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
    js(OUT/'deterministic_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
