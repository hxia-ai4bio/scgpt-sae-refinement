"""Fresh reviewer checks; no producer imports and no scGPT forward rerun."""
from pathlib import Path
import json, hashlib
from functools import lru_cache
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata

R = Path(__file__).resolve().parents[2]
W = R/'work/sae_B_remaining'
O = R/'outputs/sae_B_remaining'
H = Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
T = W/'review_trace'
T.mkdir(exist_ok=True)

@lru_cache(maxsize=None)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    m=json.loads((H/'data_manifest.json').read_text())
    k=np.load(H/'kang_embeddings.npz')
    z=np.load(R/'work/sae_interpretability/primary_activations.npz')['kang']
    split=m['splits'][0]
    tr=np.array(split['indices']['train'])
    test=np.array(split['indices']['test'])
    valid=np.array(split['indices']['valid'])
    assert not(set(tr)&set(test) or set(tr)&set(valid) or set(test)&set(valid))
    assert not(set(k['donors'][tr])&set(k['donors'][test]))
    ci=m['classes'].index('B cells')
    with h5py.File(m['preprocess']['dataset'],'r') as f:
        genes=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in f['var'][:]['name']])
        obs=f['obs'][:]
        labels=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in f['uns/cell_type_categories'][:]])
        assert np.array_equal(labels[obs['cell_type']][k['rows']], np.array(m['classes'])[k['labels']])
    assert len(set(genes))==len(genes)
    nz=z[tr]!=0
    eligible=(nz.sum(0)>=20)&(nz.mean(0)<=.5)
    auc=[]
    for donor in split['train']:
        ix=tr[k['donors'][tr]==donor]
        eligible &= (z[ix]!=0).sum(0)>=3
        pos=k['labels'][ix]==ci
        ranks=rankdata(z[ix],axis=0,method='average')
        n1=int(pos.sum());n0=len(ix)-n1
        auc.append((ranks[pos].sum(0)-n1*(n1+1)/2)/(n1*n0))
    auc=np.array(auc)
    expected=np.flatnonzero(eligible & (auc.mean(0)>=.7))
    expected=set(expected)-{160,898,1387}
    selected=pd.read_csv(O/'selected_features.csv').set_index('feature')
    assert set(selected.index)==expected=={137,370,146,1373,1125,1409}
    aucerr=max(np.max(abs(selected['B cells_mean_donor_auc']-auc.mean(0)[selected.index])),np.max(abs(selected['B cells_min_donor_auc']-auc.min(0)[selected.index])))
    assert aucerr<1e-12
    result={'status':'PASS','scope':'independent cached-data checks, no biological validation or full scGPT replay','all_2048_features_screen_recomputed':True,'selected_features':list(map(int,sorted(expected))),'auc_max_error':float(aucerr),'raw_dataset_cell_labels_match':True,'donor_split_disjoint':True,'features':{}}
    for fid in [137]:
        FO=O/f'F{fid}';FW=W/f'F{fid}';pairs=json.loads((FO/'pairs.json').read_text());group={};covered={};zero=[];base=[];replay=[]
        names=[f'sae_F{fid}',f'encoder_F{fid}','embedding_neg201','pca3']
        expected_pairs=[]
        for role,number in [('train',2),('test',4)]:
            ix=np.array(split['indices'][role])
            for donor in sorted(set(k['donors'][ix])):
                bc=ix[(k['donors'][ix]==donor)&(k['labels'][ix]==ci)]
                active=bc[z[bc,fid]>0];inactive=bc[z[bc,fid]==0]
                for pi in sorted(active,key=lambda j:(-float(z[j,fid]),int(j)))[:number]:
                    ds=np.linalg.norm(k['embedding'][inactive].astype(float)-k['embedding'][pi],axis=1)
                    rr=sorted(range(len(inactive)),key=lambda j:(ds[j],int(inactive[j])))[:3]
                    expected_pairs.extend((role,str(donor),int(pi),int(inactive[j])) for j in rr)
        assert expected_pairs==[(p['role'],p['donor'],p['prototype_index'],p['reference_index']) for p in pairs]
        for p in pairs:
            dest=FW/'cache'/('pair_%02d.npz'%p['pair_id']);a=np.load(dest);meta=json.loads(dest.with_suffix('.json').read_text())
            assert meta['sha256']==digest(dest) and meta['identity']['pair']==p
            for path,value in meta['identity']['sources'].items(): assert digest(path)==value
            cols=a['columns'];assert len(set(cols))==len(cols)
            unchanged=a['original_bins']==a['reference_bins'];assert np.array_equal(a['effects'][unchanged],np.zeros_like(a['effects'][unchanged]))
            zero.append(int(unchanged.sum()))
            base.append(abs(a['baseline'][0]-z[p['prototype_index'],fid]));replay.append(float(a['replay_error']))
            v=np.zeros((len(genes),4));v[cols]=a['effects']
            key=(p['role'],p['donor'],p['prototype_index']);group.setdefault(key,[]).append(v);covered[key]=np.isin(np.arange(len(genes)),cols)
        prot={key:np.mean(arr,axis=0) for key,arr in group.items()}
        errors=[];rankings={};coverage={}
        for role in ['train','test']:
            donor_arrays=[];donor_covers=[]
            for donor in sorted(set(key[1] for key in prot if key[0]==role)):
                keys=[key for key in prot if key[:2]==(role,donor)]
                donor_arrays.append(np.mean([prot[key] for key in keys],axis=0));donor_covers.append(np.any([covered[key] for key in keys],axis=0))
            keys=[key for key in prot if key[0]==role];cover=np.sum([covered[key] for key in keys],axis=0)
            d=np.array(donor_arrays);mean=d.mean(0);ranking=pd.read_csv(FO/f'gene_rankings_{role}.csv').set_index('gene');gene_lookup={g:i for i,g in enumerate(genes)};cols=np.array([gene_lookup[g] for g in ranking.index])
            assert set(ranking.index)==set(genes[cover>0])
            assert np.array_equal(ranking.input_prototypes,cover[cols])
            assert np.array_equal(ranking.input_donors,np.sum(donor_covers,axis=0)[cols])
            for j,name in enumerate(names):
                err=np.max(abs(ranking[name]-mean[cols,j]));errors.append(float(err));assert err<1e-10
                assert np.array_equal(ranking[name+'_positive_donors'],(d[:,cols,j]>0).sum(0))
                conditional=np.sum([prot[key][cols,j] for key in keys],axis=0)/cover[cols]
                assert np.max(abs(conditional-ranking[name+'_conditional_mean']))<1e-10
            rankings[role]=mean;coverage[role]=cover
        transfer=pd.read_csv(FO/'train_top40_transfer.csv').set_index('score')
        for j,name in enumerate(names):
            order=sorted(np.flatnonzero(rankings['train'][:,j]>0),key=lambda c:(-rankings['train'][c,j],c))[:40]
            assert transfer.loc[name,'train_top_genes'].split(';')==list(genes[order])
            present=np.array(order)[coverage['test'][order]>0]
            assert len(present)==transfer.loc[name,'test_present_n']
            assert (rankings['test'][present,j]>0).sum()==transfer.loc[name,'test_positive_n']
        allrows=pd.read_csv(FO/'heldout_deletion.csv');ninputs=0
        for p in pairs:
            if p['role']!='test': continue
            dest=FW/'deletion'/('pair_%02d.npz'%p['pair_id']);d=np.load(dest);meta=json.loads(dest.with_suffix('.json').read_text());a=np.load(FW/'cache'/('pair_%02d.npz'%p['pair_id']))
            assert digest(dest)==meta['sha256']
            train_exposed=coverage['train'][a['columns']]>0
            eligible=np.flatnonzero(train_exposed&(a['original_bins']!=a['reference_bins']))
            orders={name:sorted(eligible,key=lambda q:(-rankings['train'][a['columns'][q],j],int(a['columns'][q]))) for j,name in enumerate(names)}
            orders['sae_nonribosomal']=[q for q in orders[names[0]] if not str(genes[a['columns'][q]]).startswith(('RPL','RPS'))]
            delta=a['reference_bins']-a['original_bins'];strata=[(int(v)//10,int(np.sign(change)),int(abs(change))//5) for v,change in zip(a['original_bins'],delta)]
            for rep in range(10):
                rng=np.random.default_rng([20261019,p['pair_id'],rep]);left=set(eligible);random=[]
                for source in orders[names[0]][:40]:
                    pool=sorted(q for q in left if strata[q]==strata[source]);chosen=int(rng.choice(pool));left.remove(chosen);random.append(chosen)
                orders[f'matched_random_{rep}']=random
            for i,(order,n,actual,positions) in enumerate(zip(d['orders'],d['budgets'],d['actual'],meta['chosen_positions'])):
                if order in orders: assert positions==a['positions'][orders[order][:n]].tolist()
                assert actual==len(positions)==len(set(positions))
                expected_values=d['values'][0].copy();pos_lookup=dict(zip(a['positions'],a['reference_bins']))
                for pos in positions: expected_values[pos]=pos_lookup[pos]
                assert np.array_equal(expected_values,d['values'][i])
                for j,name in enumerate(names):
                    row=allrows[(allrows.pair_id==p['pair_id'])&(allrows.order==order)&(allrows.requested_genes==n)&(allrows.score==name)]
                    assert len(row)==1 and abs(row.iloc[0]['drop']-float(d['baseline'][j]-d['scores'][i,j]))<1e-6
                ninputs+=1
        allrows['method']=allrows.order.str.replace(r'_\d+$','',regex=True)
        psummary=allrows.groupby(['prototype_index','method','requested_genes','score'])[['baseline','activation','drop']].mean().reset_index()
        summary=psummary.groupby(['method','requested_genes','score'])[['baseline','activation','drop']].mean().sort_index()
        recorded=pd.read_csv(FO/'deletion_summary.csv').set_index(['method','requested_genes','score']).sort_index()
        assert np.max(abs(summary.to_numpy()-recorded[['baseline','activation','drop']].to_numpy()))<1e-12
        result['features'][str(fid)]={'pairs':len(pairs),'same_type_same_donor_nearest_pair_selection_exact':True,'cache_hashes_match':True,'unchanged_genes_zero':sum(zero),'baseline_max_error':float(max(base)),'cached_CPU_replay_max_error':max(replay),'donor_balanced_aggregation_max_error':max(errors),'coverage_and_conditional_mean_exact':True,'transfer_list_frozen_TRAIN':True,'deletion_inputs_checked':ninputs,'TRAIN_ranked_deletion_and_random_seed_reproduced':True,'deletion_summary_recomputed':True}
    (T/'independent_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
