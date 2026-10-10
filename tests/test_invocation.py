import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOCATION_MANIFEST = {
    "src/segment_ditches.py": "operational",
    "src/segment_embankment.py": "operational",
    "src/segment_ground.py": "operational",
}


def test_invocation_manifest_covers_every_main_guard():
    guarded_files = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in ast.walk(tree)
        ):
            guarded_files.add(path.relative_to(PROJECT_ROOT).as_posix())

    assert guarded_files == set(INVOCATION_MANIFEST)


@pytest.mark.parametrize(
    "entry",
    [
        ["src/segment_ground.py"],
        ["-m", "src.segment_ground"],
        ["src/segment_ditches.py"],
        ["-m", "src.segment_ditches"],
        ["src/segment_embankment.py"],
        ["-m", "src.segment_embankment"],
    ],
)
def test_operational_help_works_without_side_effects(entry, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-path" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "module",
    ["segment_ground", "segment_ditches", "segment_embankment"],
)
def test_help_does_not_load_standalone_dependencies_or_connect_database(module):
    code = f"""
import importlib.abc
import sys

class BlockStandaloneImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {{
            'laspy', 'matplotlib', 'open3d', 'pyvista'
        }}:
            raise ImportError(f'Standalone dependency imported: {{fullname}}')

sys.meta_path.insert(0, BlockStandaloneImports())
workflow = __import__('src.{module}', fromlist=['main'])

def fail_connect(*args, **kwargs):
    raise AssertionError('PostgreSQL connection attempted during --help')

if hasattr(workflow, 'psycopg2'):
    workflow.psycopg2.connect = fail_connect

try:
    workflow.main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('--help did not exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-path" in result.stdout
