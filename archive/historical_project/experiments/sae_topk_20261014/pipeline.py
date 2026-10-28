"""One-shot authorized workflow. Wait for current frozen CPU extraction, then finish all stages."""
from pathlib import Path
import subprocess,os,time,json,sys
R=Path(__file__).resolve().parent
while not (R/'data_manifest.json').exists():time.sleep(2)
for name in ['run.py','finish.py']:
 with (R/(name.replace('.py','')+'_pipeline.log')).open('a') as log:
  p=subprocess.run([sys.executable,str(R/name)],stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:
  (R/'PIPELINE_FAILED.json').write_text(json.dumps(dict(stage=name,returncode=p.returncode)));sys.exit(p.returncode)
print('ALL_SAE_COMPUTATION_COMPLETE',flush=True)
