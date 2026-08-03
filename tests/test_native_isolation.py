#
#//  tests/test_native_isolation.py
#//  heteroknockoffpy
#//

"""
test_xgbImportances.py and test_xgb_scip.py deliberately import
heteroknockoffpy.xgbImportances/xgbScip directly (bypassing the
torch/xgboost/rpy2 _processIsolation guard -- see their own module
docstrings), so they can unit-test the underlying xgboost logic without
isolation overhead. That's fine on its own, but if the SAME process has also
imported torch (as any test under tests/heteroknockofftorch/ does), xgboost's
OpenMP runtime crashes on macOS once it actually initializes alongside
torch's bundled libomp -- confirmed: merely importing both is fine, the crash
happens once xgboost's native compute path (e.g. .fit()) actually runs.

Fork-based isolation (pytest-forked, pytest-isolate) does NOT fix this: both
fork from the process that already collected/imported torch, and a forked
child inherits whatever native libraries were already mapped into that
process's memory -- confirmed empirically (every xgboost test still crashed
under --forked). Only a genuinely fresh interpreter process -- never a fork,
never a reused pool worker -- avoids it, which is exactly the mechanism
heteroknockoffpy._processIsolation already uses in production via
`multiprocessing.get_context('spawn')`.

conftest.py's pytest_ignore_collect keeps those two files out of the default
`pytest tests/` collection; this test re-runs them here, in a brand-new
`python -m pytest` subprocess (a real process spawn via subprocess.run, not a
fork), so the whole suite still completes correctly from one command.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ISOLATED_TEST_FILES = ("tests/test_xgbImportances.py", "tests/test_xgb_scip.py")


def test_native_bypass_suite_runs_isolated_from_torch():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *_ISOLATED_TEST_FILES, "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    #
    assert result.returncode == 0, (
        f"isolated subprocess for {_ISOLATED_TEST_FILES} exited with "
        f"{result.returncode}; see captured stdout/stderr above"
    )
#/def test_native_bypass_suite_runs_isolated_from_torch
