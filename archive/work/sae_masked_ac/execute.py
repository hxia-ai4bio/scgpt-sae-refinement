"""Run each fixed stage only after verifying supplementary consumed inputs."""
from pathlib import Path
import json,hashlib,subprocess,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_masked_ac'
for stage in ['preflight','develop','evaluate']:
 for path,h in json.loads((O/'SUPPLEMENT_BINDINGS.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 subprocess.run([sys.executable,str(W/'pipeline.py'),stage],check=True)
