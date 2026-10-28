from pathlib import Path
import os,subprocess,sys,json,time
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';features=[137,370,146,1373,1125,1409];start=time.time()
for fid in features:
 (W/f'F{fid}').mkdir(exist_ok=True);env=dict(os.environ,SAE_FEATURE=str(fid))
 for stage in ['run','analyze','pairs_qc','verify']:
  print('START',fid,stage,flush=True)
  with (W/f'F{fid}/{stage}.log').open('w') as log:subprocess.run([sys.executable,'-u',str(W/f'{stage}.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('DONE',fid,stage,'elapsed_s',round(time.time()-start),flush=True)
(O/'attribution_batch_completion.json').write_text(json.dumps({'status':'complete','features':features,'seconds':time.time()-start},indent=2));print('ALL_ATTRIBUTION_COMPLETE',flush=True)
