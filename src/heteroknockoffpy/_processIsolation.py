#
#//  _processIsolation.py
#//  heteroknockoffpy
#//

import sys
import importlib
import multiprocessing


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


def run_isolated_if_loaded(other_module: str, target_module: str, target_func: str, /, **kwargs):
    """
    Call target_module.target_func(**kwargs), importing target_module lazily -- calling
    this function must never itself trigger an import of target_module (or its heavy
    dependency) until after the isolation decision below has been made. Pass
    target_module/target_func as strings, not an already-imported callable: importing
    a module like heteroknockoffpy.xgbImportances (which does `import xgboost` at its
    own top level) as a normal Python import statement in the caller would defeat the
    whole point by loading xgboost into this process regardless of which branch below
    is taken.

    If `other_module` is not yet imported in this process, resolve and call directly
    in-process. Otherwise, run in a freshly spawned (never forked) child process that
    resolves and imports target_module fresh, so `other_module`'s native library and
    target_module's are never loaded into the same process.

    Used to keep torch and xgboost from ever loading into the same process: torch's
    wheel bundles its own libomp.dylib and xgboost's wheel links Homebrew's separate
    copy, and loading both crashes on macOS. A fresh Process is created per isolated
    call (not reused across calls), so the child never accumulates state from a prior
    isolated call.

    Note: since this can spawn a subprocess, callers invoking these functions from a
    top-level script (not a library/test file) should guard their own driver code
    with `if __name__ == "__main__":`, per normal Python multiprocessing requirements.
    """
    if other_module not in sys.modules:
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
