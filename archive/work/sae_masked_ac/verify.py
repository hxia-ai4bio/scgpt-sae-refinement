"""Verify model/readout provenance and replay saved predictions against raw labels."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='4'
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score,r2_score,mean_absolute_error
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];W=ROOT/'work/sae_masked_ac';O=ROOT/'outputs/sae_masked_ac'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 for name in ['sources.json']:
  for q,h in read(W/name).items():assert sha(q)==h,q
 for q,h in read(O/'SUPPLEMENT_BINDINGS.json').items():assert sha(q)==h,q
 lock=read(O/'SELECTION_LOCK.json')
 for q,h in lock['hashes'].items():assert sha(q)==h,q
 spec=importlib.util.spec_from_file_location('masked_verification_runtime',W/'pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.torch.set_num_threads(4);m.torch.use_deterministic_algorithms(True)
 df=pd.read_csv(O/'all_metrics.csv');st=pd.read_csv(O/'IFN_all.csv');ex=pd.read_csv(O/'expression_all.csv');assert len(df)==364
 models=list((W/'networks').glob('*/*/*/*/training.json'));assert len(models)==126
 masks=0;state_replays=0;gene_replays=0;neighbor_replays=0;panel_replays=0;parameter_checks=0;source_counts={};scaler_checks=0;alpha_checks=0;normal_equation_checks=0;score_checks=0;embedding_checks=0;head_checks=0
 for ds in ['ctrl','stim','pbmc']:
  a=m.canonical(ds);rawpath=Path('/Users/apple/Desktop/Llava-Cell/dataset')/('PBMC368k_raw.h5ad' if ds=='pbmc' else 'kang.h5ad');xx,gn,meta=m.p.prep.read(rawpath,ds!='pbmc');np.testing.assert_array_equal(a['cell_ids'],meta['cell_ids'][a['rows']]);np.testing.assert_array_equal(a['groups'],meta['replicate'][a['rows']]);np.testing.assert_array_equal(a['condition'],meta['label'][a['rows']]);np.testing.assert_array_equal(np.array(m.core.CFG['classes'])[a['labels']],meta['cell_type'][a['rows']]);source_counts[ds]=len(a['rows'])
  # Replay token-level targets on a fixed row sample against raw count normalization.
  sample=np.random.default_rng(20261021).choice(len(a['rows']),min(64,len(a['rows'])),replace=False)
  for ai in sample:
   r=int(a['rows'][ai]);cols=a['columns'][ai];valid=cols>=0;total=float(xx[r].astype('float64').sum());target=np.log1p(xx[r,cols[valid]].toarray().ravel().astype('float64')/total*1e4);np.testing.assert_allclose(target,a['target_log_expression'][ai,valid],rtol=0,atol=1e-6);assert a['hidden'][ai].sum()==max(1,int(np.floor(.25*valid.sum())));masks+=1
  with np.load(ROOT/'work/sae_evaluation'/f'{ds}_targets.npz') as q:ifn={k:q[k].copy() for k in q.files}
  byrow={int(r):i for i,r in enumerate(ifn['rows'])};lab=dict(zip(a['rows'].tolist(),a['labels'].tolist()))
  for fi in range(2 if ds=='pbmc' else 6):
   d,f,ix=m.fold_data(a,ds,fi);pn=m.panel(ds,fi,a,ix);x=sparse.load_npz(W/'data'/ds/'normalized_expression.npz')[ix['train']];s1=np.asarray(x.sum(0)).ravel();s2=np.asarray(x.power(2).sum(0)).ravel();nnz=np.asarray((x>0).sum(0)).ravel();mu=s1/np.maximum(nnz,1);sd=np.sqrt(np.maximum(s2/np.maximum(nnz,1)-mu**2,0)).clip(.1);v=s2/len(ix['train'])-(s1/len(ix['train']))**2;voc=read(Path('/Users/apple/Desktop/scGPT/scGPT_human/vocab.json'));eligible=(nnz>=20)&np.array([g in voc for g in a['gene_names']])&(v>0);cols=np.flatnonzero(eligible);expected=np.sort(cols[np.argsort(-v[cols],kind='stable')[:2048]]);np.testing.assert_array_equal(pn['columns'],expected);np.testing.assert_allclose(pn['mean'],mu[expected],rtol=0,atol=1e-10);np.testing.assert_allclose(pn['std'],sd[expected],rtol=0,atol=1e-10);np.testing.assert_array_equal(pn['train_rows'],d['train']['rows']);panel_replays+=1
   for arm in ['original_h','clean_mse']+m.ARMS:
    for seed in (['shared'] if arm=='original_h' else m.SEEDS):
     if arm in m.ARMS:
      path=W/'networks'/ds/f'fold_{fi}'/f'seed_{seed}'/arm;t=read(path/'training.json');assert t['steps']==t['selected_step']==1000;assert sha(path/'model.pt')==t['checkpoint_sha'];assert m.arraysha(d['train']['h'])==t['train_clean_sha'];assert m.arraysha(a['masked_embedding'][ix['train']])==t['train_masked_sha'];assert sha(m.source_warm(ds,fi,seed))==t['warmup_sha'];state=m.torch.load(path/'model.pt',map_location='cpu',weights_only=False);assert all(m.torch.isfinite(v).all() for v in state.values());
      if arm!='A_dense':np.testing.assert_allclose(np.linalg.norm(state['network.decoder.weight'].numpy().astype('float64'),axis=0),1,rtol=0,atol=2e-6)
      assert all(np.isfinite(z) for gg in t['gradients'] for z in gg['norms'].values());parameter_checks+=1
     model=m.restore(ds,fi,seed,arm);fc=m.features(model,a,{'train':ix['train'],'valid':ix['valid']},'clean');ft=fc['train'];fv=fc['valid'];probe=W/'probes'/ds/f'fold_{fi}'/f'seed_{seed}'/arm
     with np.load(probe/'readout.npz') as rr:readout={key:rr[key].copy() for key in rr.files}
     mu0=ft.mean(0);sd0=ft.std(0);sd0[sd0<1e-12]=1
     np.testing.assert_allclose(readout['mean'],mu0,rtol=0,atol=1e-12);np.testing.assert_allclose(readout['scale'],sd0,rtol=0,atol=1e-12);scaler_checks+=1
     ridge=m.core.Ridge(ft,d['train']['y']);best=-1.;alpha0=None;selection=read(probe/'selection.json')
     for ri,alpha in enumerate(m.core.ALPHAS):
      ridge.fit(alpha);value=m.core.metrics(d['valid']['y'],ridge.scores(fv))['macro_f1'];assert abs(value-selection['grid'][ri]['valid_macro_f1'])<1e-12
      if value>best:best=value;alpha0=alpha
     assert float(readout['alpha'])==selection['alpha']==alpha0;alpha_checks+=1
     yy=2*np.eye(8)[d['train']['y']]-1.;ww=m.core.weights(d['train']['y']);ss=(ft-mu0)/sd0;xm=np.average(ss,axis=0,weights=ww);ym=np.average(yy,axis=0,weights=ww);aa=(ss-xm)*np.sqrt(ww[:,None]);bb=(yy-ym)*np.sqrt(ww[:,None]);rhs=m.core.mm(aa,bb,ta=True);grad=m.core.mm(m.core.mm(aa,aa,ta=True),readout['coef'])+float(readout['alpha'])*readout['coef']-rhs;assert np.abs(grad).max()/max(1.,np.abs(rhs).max())<1e-7;np.testing.assert_allclose(readout['intercept'],ym-np.einsum('i,ij->j',xm,readout['coef']),rtol=0,atol=1e-10);normal_equation_checks+=1
     for view in ['clean','masked']:
      dest=O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm/view
      with np.load(dest/'test.npz') as q:
       np.testing.assert_array_equal(q['rows'],d['test']['rows']);np.testing.assert_array_equal(q['labels'],d['test']['y']);y=q['labels'];direct=np.einsum('ni,ij->nj',(q['embedding']-readout['mean'])/readout['scale'],readout['coef'],optimize=True)+readout['intercept'];np.testing.assert_allclose(direct,q['ridge_scores'],rtol=1e-9,atol=1e-7);score_checks+=1;subset=ix['test'][:512];inputh=a['embedding'][subset] if view=='clean' else a['masked_embedding'][subset];expected=inputh.astype('float64') if model is None else m.inference(model,inputh);np.testing.assert_allclose(q['embedding'][:len(subset)],expected,rtol=0,atol=1e-6);embedding_checks+=1;ridge=q['ridge_scores'].argmax(1);nr=q['neighbor_train_rows'];assert not set(q['rows'])&set(nr.ravel());nlab=np.array([[lab[int(r)] for r in rr] for rr in nr]);kn=np.stack([(nlab==c).sum(1) for c in range(8)],1).argmax(1);np.testing.assert_array_equal(kn,q['knn_prediction']);z=df[(df.dataset==ds)&(df.fold==fi)&(df.method==arm)&(df.seed.astype(str)==str(seed))&(df.view==view)];assert len(z)==1;z=z.iloc[0]
       assert abs(f1_score(y,ridge,labels=np.arange(8),average='macro',zero_division=0)-z.ridge_macro_f1)<1e-12;assert abs(f1_score(y,kn,labels=np.arange(8),average='macro',zero_division=0)-z.knn_macro_f1)<1e-12
       # Replay eight exact NN queries with saved transformed TEST and same clean TRAIN refiner.
       center=d['train']['h'].astype('float64').mean(0);nn,_=m.ev.top(m.ev.norm(q['embedding'][:8]-center),m.ev.norm(ft-center),15);np.testing.assert_array_equal(d['train']['rows'][nn],nr[:8]);neighbor_replays+=1
      for q0 in dest.glob('IFN_*.npz'):
       c=int(q0.stem.split('_')[1]);variant=q0.stem.split('_',2)[2]
       with np.load(q0) as q:
        y=q['target'];pp=q['prediction'];np.testing.assert_allclose(y,[ifn['score'][byrow[int(r)]] for r in q['rows']],rtol=0,atol=1e-12);r=st[(st.dataset==ds)&(st.fold==fi)&(st.method==arm)&(st.seed.astype(str)==str(seed))&(st.view==view)&(st.cell_type==m.core.CFG['classes'][c])&(st.variant==variant)].iloc[0];assert pd.isna(r.r2) and np.var(y)<=1e-15 or abs(r2_score(y,pp)-r.r2)<1e-12;assert abs(mean_absolute_error(y,pp)-r.mae)<1e-12
        if pd.isna(r.spearman):assert np.std(y)<=1e-12 or np.std(pp)<=1e-12
        else:assert abs(spearmanr(y,pp).statistic-r.spearman)<1e-12
        state_replays+=1
     if arm=='C_sae':
      dest=O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm
      with np.load(dest/'masked_expression_predictions.npz') as q:
       names={str(v):i for i,v in enumerate(gn)};rows=q['rows'];col=np.array([names[str(v)] for v in q['gene_names']]);ii=np.random.default_rng(20261021).choice(len(rows),min(128,len(rows)),replace=False)
       for j in ii:
        r=int(rows[j]);target=float(np.log1p(float(xx[r,col[j]])/float(xx[r].astype('float64').sum())*1e4));assert abs(target-q['target'][j])<1e-6
       batch=ix['test'][:256];rmap={int(r):j for j,r in enumerate(a['rows'][batch])};gmap={str(g):j for j,g in enumerate(pn['gene_names'])};take=np.array([j for j,r in enumerate(q['rows'][:128]) if int(r) in rmap]);assert len(take)>0
       with m.torch.no_grad():rec=model(m.torch.from_numpy(a['masked_embedding'][batch].astype('float32')));inp=((rec-model.mean)/model.sd).numpy()
       hw=model.head.weight.detach().numpy();hb=model.head.bias.detach().numpy();gp=np.einsum('ni,ji->nj',inp,hw,optimize=True)+hb;rr=np.array([rmap[int(r)] for r in q['rows'][take]]);cc=np.array([gmap[str(g)] for g in q['gene_names'][take]]);pred=gp[rr,cc]*pn['std'][cc]+pn['mean'][cc];np.testing.assert_allclose(pred,q['prediction'][take],rtol=0,atol=1e-4);head_checks+=1
       e=ex[(ex.dataset==ds)&(ex.fold==fi)&(ex.seed.astype(str)==str(seed))].iloc[0];assert abs(np.abs(q['prediction']-q['target']).mean()-e.MAE)<1e-12;assert abs(np.mean(((q['prediction']-q['target'])/q['std'])**2)-e.normalized_MSE)<1e-12;gene_replays+=1
 # Original clean embedding baseline reproduces previously sealed Ridge results.
 prior=pd.read_csv(ROOT/'outputs/sae_lambda/all_metrics.csv');orig=df[(df.dataset=='ctrl')&(df.method=='original_h')&(df.view=='clean')].sort_values('fold');old=prior[prior.method=='original_h'].sort_values('fold');np.testing.assert_allclose(orig.ridge_macro_f1.to_numpy(),old.macro_f1.to_numpy(),rtol=0,atol=1e-12)
 out=dict(status='PASS',readout_views=364,new_model_checks=parameter_checks,raw_input_counts=source_counts,raw_target_row_replays=masks,TRAIN_gene_panel_replays=panel_replays,classification_and_NN_metric_replays=364,exact_neighbor_query_block_replays=neighbor_replays,IFN_metric_replays=state_replays,expression_metric_replays=gene_replays,original_ctrl_Ridge_reproduces_sealed_baseline=True,no_new_TEST_selection=True,TRAIN_scaler_checks=scaler_checks,VALID_alpha_grid_replays=alpha_checks,weighted_normal_equation_checks=normal_equation_checks,readout_score_replays=score_checks,checkpoint_embedding_replays=embedding_checks,bounded_gene_head_prediction_replays=head_checks,IFN_prediction_replay_scope='Saved IFN targets and metric arithmetic fully replayed; fitted IFN predictions not independently refit in this script, see independent reviewer bounded checks.')
 (O/'verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
