"""Standard-library release integrity check; no model training or external access."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['files']
    for r in files:
        p = ROOT/r['path']
        assert p.is_file(), r['path']
        assert p.stat().st_size == r['bytes'], r['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'], r['path']
    source = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['files']
    for r in source:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256'], r['path']
    py = list(ROOT.rglob('*.py'))
    py = [p for p in py if not set(p.relative_to(ROOT).parts) & {'.venv','venv','build','runs','.git'}]
    for p in py:
        ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
    print(json.dumps({'status':'PASS','manifest_files':len(files),'unchanged_source_copies':len(source),'python_syntax_files':len(py)}, indent=2))

if __name__ == '__main__':
    main()
