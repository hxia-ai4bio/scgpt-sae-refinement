"""Verified resume entrypoint, including derived RNA targets and dependencies."""
from pathlib import Path
import json,hashlib,subprocess,sys
root=Path(__file__).resolve().parents[2];out=root/'outputs/sae_evaluation'
for name in ['SOURCE_BINDINGS.json','DEPENDENCY_BINDINGS.json','TARGET_BINDINGS.json']:
 for p,h in json.loads((out/name).read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
subprocess.run([sys.executable,str(root/'work/sae_evaluation/run.py'),sys.argv[1] if len(sys.argv)>1 else 'run'],check=True)
