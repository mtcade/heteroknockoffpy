#
#//  conftest.py
#//  heteroknockoffpy
#//

"""
Keeps the xgboost-direct-bypass test files (see _NATIVE_ISOLATED_FILES below)
out of the default `pytest tests/` collection, so a broad run of the suite
doesn't import xgboost directly into a process that has already (or will
later) import torch. See test_native_isolation.py for why and how those files
are still run -- in a genuinely fresh subprocess -- as part of the same
overall test command.
"""

_NATIVE_ISOLATED_FILES: frozenset[str] = frozenset({
    "test_xgbImportances.py",
    "test_xgb_scip.py",
})


def pytest_ignore_collect(collection_path, config):
    if collection_path.name not in _NATIVE_ISOLATED_FILES:
        return None
    #

    # Always collect normally when the file (or a node within it) was named
    # explicitly on the command line -- direct developer invocation (e.g.
    # `pytest tests/test_xgbImportances.py -k foo`), or test_native_isolation.py's
    # own dedicated subprocess re-invocation, which names these files explicitly.
    # Only an implicit/broad collection (e.g. `pytest tests/`, or bare `pytest`)
    # gets skipped here in favor of that isolated subprocess.
    invocation_args = [str(a) for a in config.invocation_params.args]
    if any(collection_path.name in arg for arg in invocation_args):
        return None
    #

    return True
#/def pytest_ignore_collect
