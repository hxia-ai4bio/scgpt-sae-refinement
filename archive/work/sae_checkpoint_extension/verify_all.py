"""Check global gate, two independent dataset replay reports, and aggregation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_extension'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    for p,h in read(O/'GLOBAL_INPUT_BINDINGS.json').items():assert sha(Path(p))==h,p
    lock=read(O/'GLOBAL_SELECTION_LOCK.json');assert lock['all_new_datasets_selected_before_TEST'] and lock['source_sha']==sha(W/'run.py')
    reports={}
    for ds,n in [('stim',18),('pbmc',6)]:
        assert sha(O/ds/'SELECTION_LOCK.json')==lock['dataset_lock_hashes'][ds]
        v=read(O/ds/'verification.json');assert v['status']=='PASS'
        assert v['body_trajectories']==n and v['checkpoint_VALID_objective_replays']==7*n and v['all_step1000_exactly_equal_old']
        reports[ds]=dict(sha256=sha(O/ds/'verification.json'),body_trajectories=n,checkpoint_VALID_objective_replays=7*n)
    d=pd.read_csv(O/'all_metrics.csv');kn=pd.read_csv(O/'knn_metrics.csv');assert len(d)==72 and len(kn)==144
    assert not d.duplicated(['dataset','fold','seed','method']).any()
    assert not kn.duplicated(['dataset','fold','seed','method','view']).any()
    # Independent summary arithmetic with explicitly labeled prior ctrl context.
    ctrl=pd.read_csv(ROOT/'outputs/sae_checkpoint_selection/all_metrics.csv');allrows=pd.concat([d,ctrl],ignore_index=True)
    summary=read(O/'SUMMARY.json');s=pd.read_csv(O/'summary.csv')
    for row in s.to_dict('records'):
        x=allrows[(allrows.dataset==row['dataset'])&(allrows.method==row['method'])]
        assert len(x)==(6 if row['dataset']=='pbmc' else 18)
        for key in ['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE']:
            value=x.groupby('fold')[key].mean().mean();assert abs(value-row[key])<1e-12
    for row in summary['contrasts']:
        x=allrows[allrows.dataset==row['dataset']];a=x[x.method=='C_selected'];b=x[x.method==row['reference']]
        pair=a.merge(b,on=['dataset','fold','group','seed'],suffixes=('_a','_b'),validate='one_to_one');delta=pair.MAE_a-pair.MAE_b
        group=pair.assign(delta=delta).groupby('fold')[['MAE_a','MAE_b','delta']].mean()
        assert abs(group.MAE_a.mean()-row['selected_MAE'])<1e-12
        assert abs(group.MAE_b.mean()-row['reference_MAE'])<1e-12
        assert int((delta<0).sum())==row['improved_pairs'] and len(pair)==row['pairs']
        assert int((group.delta<0).sum())==row['improved_groups'] and len(group)==row['groups']
        assert abs((1-group.MAE_a.mean()/group.MAE_b.mean())*100-row['MAE_reduction_pct'])<1e-10
    out=dict(status='PASS',new_body_trajectories=24,new_fresh_heads=24,checkpoint_VALID_replays=168,new_expression_rows=72,new_knn_rows=144,
        dataset_verification=reports,global_sources_rechecked=True,both_datasets_locked_before_new_TEST=True,root_aggregation_independently_recomputed=True,
        ctrl_context='Old ctrl source metrics are explicit context; not counted in24newpairs or72newexpressionrows.')
    (O/'verification.json').write_text(json.dumps(out,indent=2));print('GLOBAL_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
