# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,datetime,shutil
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/sae_masked_ac'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
a=json.loads((O/'EXPERIMENT_AUDIT.json').read_text());c=json.loads((O/'CLAIMS_FROM_RESULTS.json').read_text());v=json.loads((O/'verification.json').read_text());assert v['status']=='PASS'
checked={}
for artifact in [a,c]:
 for p,expected in artifact.get('audited_input_hashes',{}).items():
  pp=Path(p);pp=pp if pp.is_absolute() else ROOT/pp
  actual=digest(pp);assert actual==expected.removeprefix('sha256:'),(str(pp),actual,expected);checked[str(pp)]=actual
assert checked,'Reviewer must bind audited files'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Working code is retained at its original executable paths; copy an archival source snapshot.
source=O/'source_snapshot';source.mkdir(exist_ok=True)
for name in ['prepare.py','pipeline.py','execute.py','verify.py','report.py','finalize.py']:
 shutil.copy2(ROOT/'work/sae_masked_ac'/name,source/name)
tracker=json.loads((O/'EXPERIMENT_TRACKER.json').read_text());tracker.update(status='COMPLETE',completed_at=now,verification_status=v['status'],integrity_status=a.get('integrity_status'),claim_supported=c.get('claim_supported'),review_acceptance=a.get('acceptance_status','provisional'));save(O/'EXPERIMENT_TRACKER.json',tracker)
save(O/'AUDIT_SEAL.json',dict(sealed_at=now,reviewer_input_hashes_rechecked=checked,audit_sha256=digest(O/'EXPERIMENT_AUDIT.json'),claims_sha256=digest(O/'CLAIMS_FROM_RESULTS.json'),verification_sha256=digest(O/'verification.json'),note='Same-family provisional semantic review; deterministic replay separately PASS. Seal records evidence and does not imply general method efficacy.'))
files={str(p.relative_to(ROOT)):digest(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json'}
save(O/'FINAL_MANIFEST.json',dict(sealed_at=now,files=files,file_count=len(files)));print('SEALED',len(files),'files;',len(checked),'reviewed inputs rechecked')
