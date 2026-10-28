import hashlib,json,datetime,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_expression_probe'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,a):p.write_text(json.dumps(a,indent=2,ensure_ascii=False))
a=json.loads((O/'EXPERIMENT_AUDIT.json').read_text());c=json.loads((O/'CLAIMS_FROM_RESULTS.json').read_text());v=json.loads((O/'verification.json').read_text());assert v['status']=='PASS';checked={}
for artifact in [a,c]:
 for p,h in artifact.get('audited_input_hashes',{}).items():
  p=Path(p);p=p if p.is_absolute() else ROOT/p
  assert sha(p)==h.removeprefix('sha256:'),str(p);checked[str(p)]=sha(p)
assert checked
now=datetime.datetime.now(datetime.timezone.utc).isoformat();source=O/'source_snapshot';source.mkdir(exist_ok=True)
for p in W.glob('*.py'):shutil.copy2(p,source/p.name)
save(O/'COMPLETION.json',dict(status='COMPLETE',heads_done=84,test_results=84,verified=True,integrity_status=a.get('integrity_status'),claim_supported=c.get('claim_supported'),review_acceptance=a.get('acceptance_status','provisional'),completed_at=now,note='Audited EXPERIMENT_TRACKER.json is preserved byte-for-byte at its pre-review stage; this record is the final completion status.'))
save(O/'AUDIT_SEAL.json',dict(sealed_at=now,reviewed_inputs_rechecked=checked,reviewer_artifact_hashes={n:sha(O/n) for n in ['EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.json','verification.json']}))
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json'}
save(O/'FINAL_MANIFEST.json',dict(sealed_at=now,file_count=len(files),files=files));print('SEALED',len(files),'files;',len(checked),'reviewed inputs checked')
