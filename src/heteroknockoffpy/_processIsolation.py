#
#//  _processIsolation.py
#//  heteroknockoffpy
#//

import sys
import importlib
import multiprocessing


# Native libraries that must never coexist in the same process (OpenMP/BLAS collisions
# crash the process on macOS). Maps each isolated target_module's import path to the
# single native module it depends on, so a target's own dependency being already loaded
# (e.g. from a prior in-process call to the same function) doesn't trigger needless
# self-isolation.
#
# Every target_module passed to run_isolated_if_loaded must be a true leaf module that
# loads exactly one native dependency -- never a dispatcher that merely forwards to
# other modules (some of which may load a different native library, or none at all).
# Keying by module path only works because of this invariant.
_GUARDED_MODULES: dict[ str, str ] = {
    'heteroknockoffpy.xgbImportances': 'xgboost',
    'heteroknockoffpy.xgbScip': 'xgboost',
    'heteroknockoffpy.heteroknockofftorch.prismImportances': 'torch',
    'heteroknockoffpy.heteroknockofftorch.torchKnockoffs': 'torch',
    'heteroknockoffpy.rbridge': 'rpy2',
}


def _resolve_and_call(target_module: str, target_func: str, kwargs: dict):
    module = importlib.import_module( target_module )
    fn = getattr( module, target_func )
    return fn( **kwargs )
#/def _resolve_and_call


def _worker(conn, target_module: str, target_func: str, kwargs: dict):
    try:
        result = _resolve_and_call( target_module, target_func, kwargs )
        conn.send( ( 'ok', result ) )
    except BaseException as exc:
        conn.send( ( 'error', exc ) )
    finally:
        conn.close()
    #
#/def _worker


def run_isolated_if_loaded(target_module: str, target_func: str, /, **kwargs):
    """
    Call target_module.target_func(**kwargs), importing target_module lazily -- calling
    this function must never itself trigger an import of target_module (or its heavy
    dependency) until after the isolation decision below has been made. Pass
    target_module/target_func as strings, not an already-imported callable: importing
    a module like heteroknockoffpy.xgbImportances (which does `import xgboost` at its
    own top level) as a normal Python import statement in the caller would defeat the
    whole point by loading xgboost into this process regardless of which branch below
    is taken. target_module must be a key in _GUARDED_MODULES.

    If no *other* guarded native module (see _GUARDED_MODULES) is already imported in
    this process, resolve and call directly in-process. Otherwise, run in a freshly
    spawned (never forked) child process that resolves and imports target_module fresh,
    so the two native libraries are never loaded into the same process. target_module's
    own native dependency being already loaded (e.g. from a prior in-process call to the
    same function) does not by itself trigger isolation.

    Guards against known macOS crashes from incompatible native libraries coexisting in
    one process: torch's wheel bundles its own libomp.dylib, xgboost's wheel links
    Homebrew's separate copy, and R (via rpy2) embeds its own native runtime -- loading
    any two of these together can crash. A fresh Process is created per isolated call
    (not reused across calls), so the child never accumulates state from a prior
    isolated call.

    Note: since this can spawn a subprocess, callers invoking these functions from a
    top-level script (not a library/test file) should guard their own driver code
    with `if __name__ == "__main__":`, per normal Python multiprocessing requirements.
    """
    own_module = _GUARDED_MODULES[ target_module ]
    conflicting = {
        native_module
        for native_module in _GUARDED_MODULES.values()
        if native_module != own_module and native_module in sys.modules
    }
    if not conflicting:
        return _resolve_and_call( target_module, target_func, kwargs )
    #
    ctx = multiprocessing.get_context( 'spawn' )
    parent_conn, child_conn = ctx.Pipe( duplex=False )
    proc = ctx.Process(
        target = _worker,
        args = ( child_conn, target_module, target_func, kwargs ),
    )
    proc.start()
    child_conn.close()

    try:
        status, payload = parent_conn.recv()
    except EOFError:
        proc.join()
        raise RuntimeError(
            "Isolated subprocess for {}.{} exited without a result "
            "(exitcode={})".format( target_module, target_func, proc.exitcode )
        )
    #
    proc.join()

    if status == 'error':
        raise payload
    #
    return payload
#/def run_isolated_if_loaded
