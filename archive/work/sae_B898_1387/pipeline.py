from pathlib import Path
import subprocess,sys,time,json
W=Path(__file__).resolve().parent;P=sys.executable
first=W.parents[1]/'outputs/sae_B898_1387/F898/occlusion_completion.json'
while not first.exists():time.sleep(5)
for fid in [898,1387]:
 for stage in (['analyze','pairs_qc','verify'] if fid==898 else ['run','analyze','pairs_qc','verify']):
  f=W/f'F{fid}'/f'{stage}.py';log=f.with_suffix('.log')
  print('START',fid,stage,flush=True)
  with log.open('w') as out:subprocess.run([P,'-u',str(f)],stdout=out,stderr=subprocess.STDOUT,check=True)
  print('DONE',fid,stage,flush=True)
print('ATTRIBUTION_PIPELINE_COMPLETE',flush=True)
