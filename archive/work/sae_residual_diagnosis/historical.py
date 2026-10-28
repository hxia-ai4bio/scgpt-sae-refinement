"""Replay historical TRAIN/VALID selection diagnostics; no held-out scoring."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import diagnose as d
m=d.m;O=d.O
def main():
 m.check_sources();rows=[];funnel=[]
 for sp in sorted(m.W.glob('fold_*/seed_*/*/selection.json')):
  s=m.check_selection(sp)
  metas=[json.loads(Path(t['metadata']).read_text()) for t in s['trials']]
  if s['arm']=='gate':funnel.append({'fold':s['fold'],'seed':s['seed'],'all_beta_checkpoints_epoch0':all(x['best_epoch']==0 for x in metas),'identity_selected':s['identity_selected'],'rejection_stage':'checkpoint_epoch0' if all(x['best_epoch']==0 for x in metas) else ('fresh_Ridge' if s['identity_selected'] else 'retained')})
  for t,meta in zip(s['trials'],metas):
   h=meta['history'];v=np.array([a['valid_head_macro_f1'] for a in h]);train=[a for a in h if 'next_train_CE' in a];first,last=train[0],train[-1]
   rows.append({'fold':s['fold'],'seed':s['seed'],'arm':s['arm'],'beta':t['beta'],'best_epoch':meta['best_epoch'],'last_epoch':meta['last_epoch'],'head_delta_pp':100*(meta['valid_head_macro_f1']-v[0]),'ridge_delta_pp':100*(meta['fresh_ridge']['valid_macro_f1']-s['identity_valid_macro_f1']),'head_initial_minus_Ridge_pp':100*(v[0]-s['identity_valid_macro_f1']),'train_CE_first':first['next_train_CE'],'train_CE_last':last['next_train_CE'],'train_CE_drop':first['next_train_CE']-last['next_train_CE'],'train_penalty_last':t['beta']*last['next_train_residual_ratio_squared'],'train_SupCon_last':last['next_train_SupCon'],'max_batch_RMS_relative_residual':np.sqrt(meta['max_train_residual_ratio_squared']),'head_ever_changes':bool(np.max(np.abs(v-v[0]))>1e-12),'identity_selected':s['identity_selected']})
 pd.DataFrame(rows).to_csv(O/'historical_candidate_diagnostics.csv',index=False);m.save(O/'gate_rejection_funnel.json',funnel);scales=[];weights=[];supports=[]
 for fi in range(6):
  data,_,fold,_,_=d.data_for(fi);mu,sd=m.get_stats(data);h=data['train']['h'];norm=h.norm(dim=1);cn=(h-mu).norm(dim=1);scales.append({'fold':fi,'raw_norm_median':float(norm.median()),'centered_norm_median':float(cn.median()),'raw_to_centered_norm_median':float((norm/cn.clamp_min(1e-12)).median()),'TRAIN_donors':len(data['train']['donor'].unique()),'TRAIN_cells':len(h),'VALID_cells':len(data['valid']['h'])})
  y=data['train']['y'].numpy();don=data['train']['donor'].numpy()
  for c in range(8):
   ds=np.unique(don[y==c]);N=int((y==c).sum());supports.append({'fold':fi,'class_index':c,'TRAIN_cells':N,'VALID_cells':int((data['valid']['y'].numpy()==c).sum())})
   for dd in ds:
    n=int(((y==c)&(don==dd)).sum());weights.append({'fold':fi,'class_index':c,'TRAIN_donor_index':int(dd),'cells':n,'class_cells':N,'donors_with_class':len(ds),'CE_expected_cell_weight':1/(8*len(ds)*n),'Ridge_expected_cell_weight':1/(8*N),'CE_to_Ridge_cell_weight_ratio':N/(len(ds)*n)})
 m.save(O/'input_scale_diagnostics.json',scales);pd.DataFrame(weights).to_csv(O/'sampling_weight_mismatch.csv',index=False);pd.DataFrame(supports).to_csv(O/'class_supports.csv',index=False);print('HISTORICAL_DIAGNOSTICS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':torch.set_num_threads(4);main()
