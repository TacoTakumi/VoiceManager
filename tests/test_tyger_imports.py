"""The server process never imports the Tyger model code.

Tyger's scorer pins its own torch and transformers 5.x and runs only in the
sidecar's virtual environment. Two checks: a scan of every module the server
package ships (all of src/voicemanager except the two sidecar-only scripts,
tyger/server.py and tyger/scorer.py) finds no import of the scorer, of the sidecar
server script or of transformers; and importing the app, the model loaders and
the sidecar manager in a fresh interpreter loads none of them. No other engine in
the server imports transformers today, so the transformers check covers the whole
package and the whole process.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "voicemanager"
TYGER = PACKAGE / "pipeline" / "tyger"
SIDECAR_ONLY = {TYGER / "server.py", TYGER / "scorer.py"}
FORBIDDEN = ("voicemanager.pipeline.tyger.scorer", "voicemanager.pipeline.tyger.server",
             "tyger.scorer", "tyger.server", "transformers")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_names(path: Path) -> set[str]:
    """Every module an import statement in the file names, relative ones resolved."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            names.add(prefix)
            # "from pkg import mod" may import a submodule.
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def _offending(names: set[str]) -> list[str]:
    return sorted(n for n in names
                  if any(n == f or n.startswith(f + ".") for f in FORBIDDEN))


def test_the_scan_sees_the_scorer_import_in_the_sidecar_script():
    # The scanner itself works: the sidecar script does import the scorer.
    assert "voicemanager.pipeline.tyger.scorer" in _offending(
        _imported_names(TYGER / "server.py"))
    assert "transformers" in _offending(_imported_names(TYGER / "scorer.py"))


def test_no_server_module_imports_the_scorer_the_sidecar_script_or_transformers():
    scanned = 0
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path in SIDECAR_ONLY:
            continue
        scanned += 1
        bad = _offending(_imported_names(path))
        if bad:
            found[str(path.relative_to(SRC))] = bad
    assert scanned > 20
    assert (TYGER / "sidecar.py").is_file()
    assert found == {}


def test_the_sidecar_manager_names_the_script_by_path_only():
    source = (TYGER / "sidecar.py").read_text(encoding="utf-8")
    assert "importlib" not in source and "__import__" not in source
    assert '"server.py"' in source


def test_importing_the_server_loads_no_tyger_model_code():
    code = (
        "import json, sys\n"
        "import voicemanager.app, voicemanager.models\n"
        "import voicemanager.pipeline.tyger.sidecar\n"
        "import voicemanager.transport.websocket\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'transformers'\n"
        "    or m.startswith('transformers.') or m in ('tyger',\n"
        "    'voicemanager.pipeline.tyger.scorer', 'voicemanager.pipeline.tyger.server'))))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
