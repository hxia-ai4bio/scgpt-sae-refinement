"""Archive complete downstream evidence, excluding original large inputs."""
import json, zipfile
from pathlib import Path
from core import *

def main():
    assert read(O/'verification.json')['status']=='PASS'
    assert (O/'EXPERIMENT_AUDIT.json').exists() and (O/'CLAIMS_FROM_RESULTS.json').exists()
    files=[p for base in [W,O] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_MANIFEST.json' and not p.name.endswith('.tmp')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    save(O/'FINAL_MANIFEST.json',manifest);files.append(O/'FINAL_MANIFEST.json')
    archive=ROOT/'outputs/SAE_complementarity_experiment.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'CRC':'PASS'}),flush=True)
if __name__=='__main__':main()
