"""Independent final source-number and mapping checks; no model execution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

R=Path(__file__).resolve().parents[2];W=R/'work/sae_B_remaining';O=R/'outputs/sae_B_remaining';T=W/'review_trace';F=[137,370,146,1373,1125,1409]
def bh(x):
    x=np.array(x,float);i=np.argsort(x);out=np.empty(len(x));out[i]=np.minimum.accumulate((x[i]*len(x)/np.arange(1,len(x)+1))[::-1])[::-1]
    return np.minimum(out,1)
def main():
    for fid in F:
        for name in ['verification','input_precheck']:
            assert json.loads((O/f'F{fid}/{name}.json').read_text())['status']=='PASS'
        for name in ['occlusion_completion','analysis_completion']:
            assert json.loads((O/f'F{fid}/{name}.json').read_text())['status']=='complete'
    assert json.loads((O/'pathway_verification.json').read_text())['status']=='PASS'
    p=pd.read_csv(O/'all_pathway_results.csv')
    assert set(p.feature)==set(F) and p.groupby('feature').analysis.nunique().eq(2).all()
    assert p.groupby(['feature','analysis']).collection.nunique().eq(3).all()
    assert not p.duplicated(['feature','analysis','collection','term']).any()
    qerr={q:float(np.max(abs(bh(p[raw])-p[q]))) for raw,q in [('p_ora','q_ora_global'),('p_matched','q_matched_global'),('p_matched_exact','q_matched_exact_global')]}
    assert max(qerr.values())<1e-12
    assert np.max(abs(hypergeom.sf(p.overlap-1,p.background_n,p.pathway_background_n,p.selected_n)-p.p_ora))<1e-12
    assert np.max(abs((p.matched_tail_count+1)/10001-p.p_matched))<1e-12
    top=pd.read_csv(O/'top40_gene_contributions.csv');candidate=pd.read_csv(O/'candidate_pathway_summary.csv');links=pd.read_csv(O/'candidate_pathway_genes.csv')
    gene_maps={};test_maps={};stats=[]
    for fid in F:
        g=pd.read_csv(O/f'F{fid}/gene_rankings_train.csv').sort_values([f'sae_F{fid}','gene'],ascending=[False,True]).set_index('gene');test=pd.read_csv(O/f'F{fid}/gene_rankings_test.csv').set_index('gene');gene_maps[fid]=g;test_maps[fid]=test
        for analysis in ['all_input_top40','nonribosomal_top40']:
            a=g if analysis=='all_input_top40' else g[~g.index.str.startswith(('RPL','RPS'))]
            selected=a[a[f'sae_F{fid}']>0].head(40)
            tt=top[(top.feature==fid)&(top.analysis==analysis)].set_index('gene')
            assert list(tt.index)==list(selected.index)
            assert np.max(abs(tt.train_contribution-selected[f'sae_F{fid}']))<1e-12
            for gene,v in tt.iterrows():
                if gene in test.index: assert abs(v.test_contribution-test.loc[gene,f'sae_F{fid}'])<1e-12
                else: assert pd.isna(v.test_contribution)
            bg=pd.read_csv(O/f'F{fid}/background_{analysis}.csv')
            assert set(bg.gene)==set(a.index)
        stats.append({'feature':fid,'TRAIN_covered_genes':len(g),'TRAIN_top40_test_covered':int(top[(top.feature==fid)&(top.analysis=='all_input_top40')].test_contribution.notna().sum()),'TRAIN_top40_test_positive':int((top[(top.feature==fid)&(top.analysis=='all_input_top40')].test_contribution>0).sum())})
    pmap=p.set_index(['feature','analysis','collection','term']);cmap=candidate.set_index(['feature','analysis','collection','term'])
    for key,c in cmap.iterrows():
        row=pmap.loc[key]
        for col in ['overlap','p_ora','p_matched','q_ora_global','q_matched_global','q_matched_exact_global']:
            assert abs(c[col]-row[col])<1e-12
        members=set(str(row.overlap_genes).split(';')) if pd.notna(row.overlap_genes) else set()
        seen=links[(links.feature==key[0])&(links.analysis==key[1])&(links.collection==key[2])&(links.term==key[3])]
        assert set(seen.gene)==members and len(seen)==len(members)
        g=gene_maps[key[0]]
        for v in seen.itertuples():
            assert abs(v.train_contribution-g.loc[v.gene,f'sae_F{key[0]}'])<1e-12
            test=test_maps[key[0]]
            if v.gene in test.index: assert abs(v.test_contribution-test.loc[v.gene,f'sae_F{key[0]}'])<1e-12
            else: assert pd.isna(v.test_contribution)
    assert not links.duplicated(['feature','analysis','collection','term','gene']).any()
    follow=pd.read_csv(O/'descriptive_followup_genes.csv')
    expected=top[(top.analysis=='nonribosomal_top40')&(top.positive_TRAIN_donors>=3)&(top.input_TRAIN_donors>=3)&(top.test_contribution>0)]
    assert set(zip(follow.feature,follow.gene))==set(zip(expected.feature,expected.gene))
    alltop=pd.read_csv(O/'all9_top40_gene_contributions.csv');overlap=pd.read_csv(O/'all9_feature_gene_overlap.csv');topsets={}
    for fid in [160,898,1387]+F:
        source=O.parent/'sae_gene_attribution_v2' if fid==160 else O.parent/'sae_B898_1387'/f'F{fid}' if fid in [898,1387] else O/f'F{fid}'
        g=pd.read_csv(source/'gene_evidence_combined.csv').sort_values([f'sae_F{fid}_train','gene'],ascending=[False,True]).set_index('gene')
        for mode in ['all','non_RPL_RPS']:
            a=g if mode=='all' else g[~g.index.str.startswith(('RPL','RPS'))];a=a[a[f'sae_F{fid}_train']>0].head(40)
            d=alltop[(alltop.feature==fid)&(alltop['list']==mode)].set_index('gene');assert list(d.index)==list(a.index)
            assert np.max(abs(d.train_contribution-a[f'sae_F{fid}_train']))<1e-12;topsets[(fid,mode)]=set(a.index)
    for row in overlap.itertuples():
        a=topsets[(row.feature_a,row.list)];b=topsets[(row.feature_b,row.list)]
        assert row.top40_overlap==len(a&b) and abs(row.top40_Jaccard-len(a&b)/len(a|b))<1e-12
    for fid in F:
        fsummary=pd.read_csv(O/'feature_summary.csv').set_index('feature').loc[fid]
        s=next(x for x in stats if x['feature']==fid)
        for a,b in [('TRAIN_covered_genes','TRAIN_covered_genes'),('TRAIN_top40_test_covered','TRAIN_top40_TEST_covered'),('TRAIN_top40_test_positive','TRAIN_top40_TEST_positive')]: assert s[a]==fsummary[b]
    counts={q:int((p[q]<.05).sum()) for q in qerr}
    minq={q:float(p[q].min()) for q in qerr}
    result={'status':'PASS','scope':'final source-number/mapping checks, not independent biological validation','hypotheses':len(p),'contexts':int(p.groupby(['feature','analysis','collection']).ngroups),'FDR_counts':counts,'minimum_q':minq,'BH_max_error':qerr,'top40_rows':len(top),'candidate_rows':len(candidate),'candidate_gene_rows':len(links),'descriptive_followup_rows':len(follow),'all9_top_rows':len(alltop),'all9_overlap_rows':len(overlap),'complete_feature_runs':len(F),'feature_stats':stats,'all_claimed_sources_exist':True,'track_status_at_review':json.loads((O/'EXPERIMENT_TRACKER.json').read_text())['status']}
    (T/'final_evidence_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
