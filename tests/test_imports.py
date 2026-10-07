import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_utility_package_import_does_not_require_open3d():
    code = """
import builtins

original_import = builtins.__import__

def import_without_open3d(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'open3d':
        raise ImportError('open3d is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_open3d
import src.utils
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ground_segmenter_imports_from_parent_project():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.embankment_segmentation.src.segment_ground import GroundSegmenter; "
            "assert GroundSegmenter.__module__ == "
            "'src.embankment_segmentation.src.segment_ground'",
        ],
        cwd=PROJECT_ROOT.parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
