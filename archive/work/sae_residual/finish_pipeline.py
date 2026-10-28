"""Wait for TRAIN/VALID development, then fixed evaluation/verification/report."""
from pathlib import Path
import os,subprocess,sys,time,json
W=Path(__file__).resolve().parent
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
start=time.time()
while len(list(W.glob('fold_*/seed_*/*/selection.json')))<126:
 log=W/'development.log'
 if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):raise RuntimeError('Development failed; inspect development.log')
 if time.time()-start>7200:raise TimeoutError('Development exceeded bounded wait; inspect state')
 time.sleep(3)
for name,cmd in [('evaluation',[str(W/'fast_evaluate.py')]),('verification',[str(W/'verify.py')]),('report',[str(W/'report.py')])]:
 print('START',name,flush=True)
 with (W/(name+'.log')).open('w') as f:subprocess.run([sys.executable]+cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ)
 print('DONE',name,flush=True)
print('RESULTS_READY_FOR_REVIEW',flush=True)
