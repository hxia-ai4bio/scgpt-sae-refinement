"""Check supplementary consumed inputs before invoking fixed stages."""
import subprocess,sys,json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_selection'
for stage in ['develop','evaluate']:
 for p,h in json.loads((O/'SUPPLEMENT_BINDINGS.json').read_text()).items():
  assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 subprocess.run([sys.executable,str(W/'run.py'),stage],check=True)
