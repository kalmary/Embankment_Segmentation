import subprocess
import sys
from pathlib import Path

import pytest


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


def test_voxel_subsampling_import_does_not_require_open3d():
    code = """
import builtins
import numpy as np

original_import = builtins.__import__

def import_without_open3d(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'open3d':
        raise ImportError('open3d is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_open3d
from src.utils.pcd_tools import voxel_subsample_vectorized

points = np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [1.0, 1.0, 1.0]])
mask = voxel_subsample_vectorized(points, voxel_size=0.1)
assert mask.dtype == np.bool_
assert mask.tolist() == [False, True, True]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ground_segmenter_imports_from_parent_project():
    code = """
import importlib.abc

class BlockStandaloneImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'laspy', 'open3d'}:
            raise ImportError(f'Standalone dependency imported: {fullname}')

import sys
sys.meta_path.insert(0, BlockStandaloneImports())
from src.embankment_segmentation.src.segment_ground import GroundSegmenter
assert GroundSegmenter.__module__ == 'src.embankment_segmentation.src.segment_ground'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT.parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ground_segmenter_imports_from_own_project_without_standalone_dependencies():
    code = """
import importlib.abc
import sys

class BlockStandaloneImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'laspy', 'matplotlib', 'open3d', 'pyvista'
        }:
            raise ImportError(f'Standalone dependency imported: {fullname}')

sys.meta_path.insert(0, BlockStandaloneImports())
from src.segment_ground import GroundSegmenter

assert GroundSegmenter.__module__ == 'src.segment_ground'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    ["segment_ground", "segment_ditches", "segment_embankment"],
)
def test_reusable_module_imports_do_not_depend_on_execution_context(module):
    code = f"""
import importlib
import inspect

workflow = importlib.import_module('src.{module}')
for name, value in vars(workflow).items():
    if name == '_run_direct_entry_point':
        continue
    if (
        (inspect.isfunction(value) or inspect.isclass(value))
        and value.__module__ == workflow.__name__
    ):
        source = inspect.getsource(value)
        assert 'if __package__' not in source, name
        assert 'sys.path' not in source, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
