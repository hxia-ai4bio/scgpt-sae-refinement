"""Create isolated dataset-generalized copies; sealed ctrl sources stay untouched."""
from pathlib import Path
root=Path(__file__).resolve().parents[2];w=root/'work/sae_checkpoint_extension';p=root/'work/sae_checkpoint_selection'
s=(p/'run.py').read_text()
s=s.replace('Kang ctrl checkpoint-selection diagnosis','Stim/PBMC checkpoint-selection replication')
s=s.replace("W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_selection'", "BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]\nDS=sys.argv[2] if len(sys.argv)>2 else os.environ.get('SAE_CHECKPOINT_DATASET','stim')\nassert DS in ['stim','pbmc'],DS\nNF=6 if DS=='stim' else 2;N=NF*3\nW=BASE/DS;O=ROOT/'outputs/sae_checkpoint_extension'/DS")
s=s.replace("'ctrl'",'DS').replace('range(6)','range(NF)').replace('else 6','else NF')
s=s.replace("m.W/'networks/ctrl'", "m.W/'networks'/DS").replace("q.O/'predictions/ctrl'", "q.O/'predictions'/DS")
s=s.replace('dict(dataset=DS,folds=6,','dict(dataset=DS,folds=NF,')
s=s.replace('new_body_trajectories=18,new_fresh_heads=18','new_body_trajectories=N,new_fresh_heads=N')
s=s.replace('total=18','total=N').replace('==18','==N').replace('len(records)==54 and len(knn)==108','len(records)==N*3 and len(knn)==N*6')
s=s.replace('new_bodies=18,new_heads=18,expression_results=54,knn_results=108','new_bodies=N,new_heads=N,expression_results=N*3,knn_results=N*6')
s=s.replace("baseline_reuse='Reuse sealed Kang ctrl original_h/C_1000 fresh-linear-probe models,predictions,metrics. New step1000 body must exactly match old state before using these references.'", "baseline_reuse='Reuse same-dataset sealed original_h/C_1000 fresh-linear-probe models,predictions,metrics. New step1000 body must exactly match old state before using these references.'")
s=s.replace("aggregation='Average3seeds within heldout donor then equal six donors; record all18pairs. No TEST choice among checkpoint,method,seed.'", "aggregation='Average3seeds within heldout group then equal groups: stim six historical Kang donors, PBMC two batch directions. No TEST choice among checkpoint,method,seed.'")
s=s.replace("scope='Diagnostic on historical repeatedly explored Kang ctrl donors, not blind confirmation; one fixed25pct mask on originally selected positive tokens. Only checkpoint selection changes, not sparsity/architecture/loss. Finite optimization and coarse checkpoint grid; upstream VALID reused for readout selection.'", "scope='Same fixed rule transferred from ctrl, not new tuning: historical stim shares Kang donors with ctrl; PBMC is two batch directions, not verified donor generalization. One fixed25pct mask on originally selected positive tokens; no zero-expression/all-gene imputation. Same architecture/loss/budget; finite readout optimization and upstream VALID reused for head selection. No TEST-based adjustment.'")
s=s.replace("    print('PREFLIGHT_PASS',flush=True)","    print('PREFLIGHT_PASS',DS,flush=True)")
s=s.replace("print('BODY_COMPLETE',fi", "print('BODY_COMPLETE',DS,fi").replace("print('PROBE_COMPLETE',fi", "print('PROBE_COMPLETE',DS,fi").replace("print('TEST_COMPLETE',fi", "print('TEST_COMPLETE',DS,fi")
# Bind every additional split/cache/raw provenance input before any training.
needle="    a=m.canonical(DS);checks=[]"
replacement="""    provenance=read(m.W/'data'/DS/'binding.json')
    assert sha(provenance['raw_path'])==provenance['raw_sha']
    assert sha(provenance['canonical_path'])==provenance['canonical_sha']
    for path,h in provenance['sources'].items():assert sha(path)==h,path
    extras=[Path(provenance['raw_path']),Path(provenance['canonical_path']),m.W/'data'/DS/'binding.json',m.O/f'{DS}_DATA_MANIFEST.json',m.p.ext.dsout(DS)/'SPLITS.json',ROOT/'work/sae_checkpoint_selection/run.py',ROOT/'outputs/sae_checkpoint_selection/FINAL_MANIFEST.json']
    for path in extras+list(map(Path,provenance['sources'])):bindings[str(path)]=sha(path)
    a=m.canonical(DS)
    with np.load(provenance['canonical_path']) as cache:
        for key in ['rows','embedding','labels','cell_ids','condition','groups']:np.testing.assert_array_equal(a[key],cache[key])
    checks=[]"""
assert needle in s;s=s.replace(needle,replacement)
# Add role and saved-baseline-TRAIN identities without touching selection rules.
needle="        d,f,ix=m.fold_data(a,DS,fi);pn=m.panel(DS,fi,a,ix)\n        for seed in SEEDS:"
replacement="""        d,f,ix=m.fold_data(a,DS,fi);pn=m.panel(DS,fi,a,ix)
        assert not set(d['train']['rows'])&set(d['valid']['rows'])
        assert not set(d['test']['donor'])&(set(d['train']['donor'])|set(d['valid']['donor']))
        if DS=='stim':assert not set(d['train']['donor'])&set(d['valid']['donor'])
        for seed in SEEDS:"""
assert needle in s;s=s.replace(needle,replacement)
needle="                home=q.dest(DS,fi,seed,arm)"
replacement="""                home=q.dest(DS,fi,seed,arm)
                with np.load(home/'features.npz') as oldfeat:
                    for role in ['train','valid']:np.testing.assert_array_equal(oldfeat[role+'_rows'],d[role]['rows'])"""
assert needle in s;s=s.replace(needle,replacement)
# Enforce the all-dataset lock even for direct evaluate invocations.
needle="    check_inputs();lock=read(O/'SELECTION_LOCK.json')\n    for p,h in lock['hashes'].items():assert sha(p)==h,p"
replacement="""    check_inputs();global_lock=read(O.parent/'GLOBAL_SELECTION_LOCK.json')
    assert global_lock['source_sha']==sha(__file__)
    assert global_lock['all_new_datasets_selected_before_TEST']
    assert set(global_lock['dataset_lock_hashes'])=={'stim','pbmc'}
    for ds,h in global_lock['dataset_lock_hashes'].items():assert sha(O.parent/ds/'SELECTION_LOCK.json')==h,ds
    lock=read(O/'SELECTION_LOCK.json')
    for p,h in lock['hashes'].items():assert sha(p)==h,p"""
assert needle in s;s=s.replace(needle,replacement)
(w/'run.py').write_text(s)
# Verifier remains a per-dataset replay with the identical scientific checks.
v=(p/'verify.py').read_text()
v=v.replace('import importlib.util','import importlib.util,os,sys',1)
v=v.replace('W=Path(__file__).resolve().parent', "W=Path(__file__).resolve().parent\nos.environ['SAE_CHECKPOINT_DATASET']=sys.argv[1]")
v=v.replace("    for p,h in r.read(r.O/'SUPPLEMENT_BINDINGS.json').items():assert r.sha(p)==h,p", "    global_lock=r.read(r.O.parent/'GLOBAL_SELECTION_LOCK.json')\n    assert r.sha(r.O/'SELECTION_LOCK.json')==global_lock['dataset_lock_hashes'][r.DS]")
v=v.replace("a=r.m.canonical('ctrl')","a=r.m.canonical(r.DS)")
start=v.index("    with np.load(r.m.core.R/'kang_embeddings.npz') as z:")
end=v.index('    for fi in range(6):',start)
v=v[:start]+"""    provenance=r.read(r.m.W/'data'/r.DS/'binding.json')
    with np.load(provenance['canonical_path']) as z:
        for key in ['rows','embedding','labels','cell_ids','condition','groups']:np.testing.assert_array_equal(a[key],z[key])
    assert r.sha(provenance['raw_path'])==provenance['raw_sha']
    assert r.sha(provenance['canonical_path'])==provenance['canonical_sha']
    for p,h in provenance['sources'].items():assert r.sha(p)==h,p
"""+v[end:]
v=v.replace("'ctrl'",'r.DS').replace('range(6)','range(r.NF)')
v=v.replace("r.m.W/'networks/ctrl'", "r.m.W/'networks'/r.DS").replace("r.q.O/'predictions/ctrl'", "r.q.O/'predictions'/r.DS")
v=v.replace('checks==18 and checkpoints==126','checks==r.N and checkpoints==r.N*7')
v=v.replace('len(df)==54 and len(kn)==108','len(df)==r.N*3 and len(kn)==r.N*6')
v=v.replace('all18_step1000_exactly_equal_old=True','all_step1000_exactly_equal_old=True').replace('paired_initialization_and_batch_checks=18','paired_initialization_and_batch_checks=r.N').replace('bounded_checkpoint_forward_roles=54','bounded_checkpoint_forward_roles=r.N*3')
# Path of per-dataset body/head working caches is in r.W, not verifier directory.
v=v.replace("source_sha']==r.sha(W/'run.py')", "source_sha']==r.sha(W/'run.py')")
(w/'verify.py').write_text(v)
print('WROTE generalized run.py and verify.py')
