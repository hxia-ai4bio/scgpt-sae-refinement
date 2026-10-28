from pathlib import Path
import sys,subprocess,time
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';F=[137,370,146,1373,1125,1409]
while not all((O/f'F{x}/verification.json').exists() for x in F):time.sleep(5)
for stage in ['pathways','verify_pathways','report']:
 print('START',stage,flush=True)
 with (W/f'{stage}.log').open('w') as log:subprocess.run([sys.executable,'-u',str(W/f'{stage}.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',stage,flush=True)
print('RESULTS_READY_FOR_AUDIT',flush=True)
