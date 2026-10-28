"""Compare cached overlapping VALID grids with old published local results.

No fit, no new test prediction, no new selection. This explains readout changes.
"""
import numpy as np
from core import *

def main():
    oldgrid=read(R/'protocol.json')['config']['probe_alphas'];records=[]
    for fi in CFG['folds']:
        for seed in SEEDS:
            src=src_for(fi,seed)/'sparse/result.json';old=read(src);newdir=W/'models'/f'fold_{fi}'/f'seed_{seed}'/'sparse_z';new=read(newdir/'selection.json');trials=read(newdir/'trials.json')
            overlap={r['alpha']:r['VALID_F1'] for r in trials if r['alpha'] in oldgrid}
            err=max(abs(r['valid']['macro_f1']-overlap[r['alpha']]) for r in old['trials']);assert err<1e-12,(fi,seed,err)
            records.append({'fold':fi,'seed':seed,'old_alpha':old['selected_alpha'],'new_alpha':new['alpha'],
                            'old_VALID_F1':old['validation_best'],'new_VALID_F1':new['VALID_F1'],'overlap_VALID_F1_max_error':err,
                            'historical_old_grid_TEST_F1':old['test']['macro_f1'],'historical_old_grid_PBMC_F1':old['pbmc']['macro_f1'],
                            'old_result_path':str(src),'old_result_sha':sha(src)})
    result={'scope':'cached overlap/old result audit only; no new fit/test prediction or retuning','old_grid':oldgrid,'new_grid':ALPHAS,'selected_above_old_max_runs':sum(r['new_alpha']>max(oldgrid) for r in records),
            'historical_old_grid_TEST_percent':100*float(np.mean([r['historical_old_grid_TEST_F1'] for r in records])),
            'historical_old_grid_PBMC_percent':100*float(np.mean([r['historical_old_grid_PBMC_F1'] for r in records])),
            'same_alpha_overlap_max_VALID_F1_error':max(r['overlap_VALID_F1_max_error'] for r in records),'records':records}
    save(O/'regularization_diagnostics.json',result);print({k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':main()
