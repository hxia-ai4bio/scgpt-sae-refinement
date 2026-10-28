import subprocess,sys
from pathlib import Path
p=Path(__file__).with_name('run.py')
for stage in ['develop','evaluate']:
 subprocess.run([sys.executable,str(p),stage],check=True)
