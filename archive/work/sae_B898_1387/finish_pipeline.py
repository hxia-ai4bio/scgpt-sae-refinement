from pathlib import Path
import subprocess,sys,time
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B898_1387'
while not all((O/f'F{fid}/verification.json').exists() for fid in [898,1387]):time.sleep(5)
for stage in ['pathways','verify_pathways','report']:
 print('START',stage,flush=True)
 with (W/f'{stage}.log').open('w') as out:subprocess.run([sys.executable,'-u',str(W/f'{stage}.py')],stdout=out,stderr=subprocess.STDOUT,check=True)
 print('DONE',stage,flush=True)
print('RESULTS_READY_FOR_AUDIT',flush=True)
