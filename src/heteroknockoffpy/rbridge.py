#
#//  rbridge.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 2/13/26.
#//

from . import utilities
from .utilities import OutcomeDescriptor

import os
os.environ["RPY2_CFFI_MODE"] = "ABI"

import polars as pl
import pandas as pd
import numpy as np

from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import NULL as rNULL
from rpy2.robjects import conversion
from rpy2.rinterface_lib import callbacks as _r_cb

import contextlib
import sys
from typing import Literal

@contextlib.contextmanager
def _r_warnings_to_stdout():
    """
    Context manager: redirect R warnings and messages to stdout, printed immediately.

    Two things happen on entry:
    1. rpy2's consolewrite_warnerror callback is replaced with a stdout writer
       so that warning/message text reaches the terminal.
    2. R's warn option is set to 1 (immediate), overriding the default 0
       (buffered).  With warn=0, R accumulates warnings silently and only
       prints "There were N warnings" at the end of a top-level expression,
       making individual messages invisible.  warn=1 prints each warning as
       it is raised, which is what we want for diagnostics.

    Both are restored on exit.
    """
    _orig_cb = _r_cb.consolewrite_warnerror
    with ( ro.default_converter ).context():
        _orig_warn: int = int( ro.r( 'getOption("warn")' )[0] )
    _r_cb.consolewrite_warnerror = lambda s: (sys.stdout.write(s), sys.stdout.flush())
    with ( ro.default_converter ).context():
        ro.r( 'options(warn=1)' )
    try:
        yield
    finally:
        _r_cb.consolewrite_warnerror = _orig_cb
        with ( ro.default_converter ).context():
            ro.r( f'options(warn={_orig_warn})' )
    #/try/finally
#/def _r_warnings_to_stdout

if not rpackages.isinstalled('knockoff'):
    rutils = rpackages.importr('utils')
    rutils.chooseCRANmirror(ind=1)
    rutils.install_packages(
        'knockoff',
    )
#

if not rpackages.isinstalled('ranger'):
    rutils = rpackages.importr('utils')
    rutils.chooseCRANmirror(ind=1)
    rutils.install_packages(
        'ranger',
    )
#


rKnockoff = rpackages.importr('knockoff')

rRanger = rpackages.importr('ranger')

rStats = rpackages.importr('stats')

def get_ohe_forest_probabilities_np(
    X: pl.DataFrame,
    logit: bool = True,
    drop_first: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        Use r ranger to get probabilities, or log probabilities if logit is selected. If we have `logit` and `drop_first`, subtract the first column.

        The column loop runs entirely in R (forest.ohe_probabilities), loaded once
        per call via rpy2.robjects.packages.STAP.
    """
    # -- Load R script via STAP
    _script_path: str = os.path.normpath(
        os.path.join(
            os.path.dirname( os.path.abspath( __file__ ) ),
            '..', '..', 'scripts',
            'forest.ohe_probabilities.R',
        )
    )
    with open( _script_path ) as _f:
        _r_code: str = _f.read()

    with ro.default_converter.context():
        _ohe_probs = rpackages.STAP( _r_code, "ohe_probs" )

        # -- Convert X to R data.frame
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            X_r = ro.conversion.get_conversion().py2rpy( X.to_pandas() )

        # -- Single R call: returns named list of n×k probability matrices
        with ( _r_warnings_to_stdout() if verbose > 0 else contextlib.nullcontext() ):
            result_r = _ohe_probs.forest_ohe_probabilities(
                X_r,
                **{ k.replace( '_', '.' ): v for k, v in kwargs.items() },
            )

        # -- Extract probability matrices per categorical column
        columns_dict: dict[ str, np.ndarray ] = {}
        for col in list( result_r.names ):
            _r_vec = result_r.rx2( col )
            with (
                ro.default_converter + numpy2ri.converter
            ).context():
                columns_dict[col] = np.asarray( _r_vec )

    # -- Apply logit transform (with zero-handling) if requested
    # (pure Python from here — outside the rpy2 context)
    if logit:
        for col in columns_dict:
            proba: np.ndarray = columns_dict[col]
            zeroMask: np.ndarray = (proba == 0.0)
            if np.any( zeroMask ):
                predictions_infMask: np.ndarray = np.where(
                    zeroMask, np.inf, proba
                )
                proba_min = np.min( predictions_infMask, axis = 0 )
                proba = np.where( zeroMask, proba_min, proba )
                proba = proba / np.sum( proba, axis = 1 )[:,np.newaxis]
            columns_dict[col] = np.log( proba )

    if drop_first:
        if logit:
            # Subtract first column and drop it
            columns_dict = {
                col: val[:,1:] - val[:,0:1]\
                    for col, val in columns_dict.items()
            }
        else:
            # Just drop first
            columns_dict = {
                col: val[:,1:]\
                    for col, val in columns_dict.items()
            }
        #/if logit/else
    #/if drop_first

    X_ohe_probabilities_np: np.ndarray = np.concatenate(
        tuple(
            columns_dict[col] if dtype == pl.Categorical\
                else X[col].to_numpy()[:,np.newaxis]\
                for col, dtype in X.schema.items()
            #/
        ),
        axis = 1,
    )

    assert X_ohe_probabilities_np.shape[0] == X.shape[0]
    assert X_ohe_probabilities_np.shape[1] >= X.shape[1]

    return X_ohe_probabilities_np
#/def get_ohe_forest_probabilities_np

def get_forest_conditional_expectations(
    X: pl.DataFrame,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs
    ) -> pl.DataFrame:
    """
        Uses r ranger::ranger to get conditional expectations for each numeric column of X.

        The column loop runs entirely in R (forest.conditional_expectations), loaded once
        per call via rpy2.robjects.packages.STAP.

        :param kwargs: Passed to r ranger::ranger
        :returns: DataFrame with non-categorical columns of X replaced by their conditional expectations
    """
    # -- Load R script via STAP
    _script_path: str = os.path.normpath(
        os.path.join(
            os.path.dirname( os.path.abspath( __file__ ) ),
            '..', '..', 'scripts',
            'forest.conditional_expectations.R',
        )
    )
    with open( _script_path ) as _f:
        _r_code: str = _f.read()

    with ro.default_converter.context():
        _cond_exp = rpackages.STAP( _r_code, "cond_exp" )

        # -- Convert X to R data.frame
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            X_r = ro.conversion.get_conversion().py2rpy( X.to_pandas() )

        # -- Single R call: returns named list of per-column expectation vectors
        with ( _r_warnings_to_stdout() if verbose > 0 else contextlib.nullcontext() ):
            result_r = _cond_exp.forest_conditional_expectations(
                X_r,
                **{ k.replace( '_', '.' ): v for k, v in kwargs.items() },
            )

        # -- Extract expectation vectors per numeric column
        conditional_expectations_dict: dict[ str, np.ndarray ] = {}
        for col in list( result_r.names ):
            _r_vec = result_r.rx2( col )
            with (
                ro.default_converter + numpy2ri.converter
            ).context():
                conditional_expectations_dict[col] = np.asarray( _r_vec )

    return pl.DataFrame(
        conditional_expectations_dict,
        schema = {
            col: val for col, val in X.schema.items()\
                if val != pl.Categorical
            #/
        }
    )
#/def get_forest_conditional_expectations

# -- Categorical SCIP with numeric knockoffs

def get_knockoffs_with_Xk_numeric(
    X: pl.DataFrame,
    Xk_numeric: np.ndarray,
    rng: np.random.Generator,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Categorical-only SCIP: numeric knockoffs are pre-provided; only
        categorical knockoffs are generated here, sequentially.

        The column loop runs entirely in R (scip.knockoffs.R), loaded once
        per call via rpy2.robjects.packages.STAP.

        :param kwargs: Passed to r ranger::ranger
    """
    numeric_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items() if dtype != pl.Categorical
    )
    assert len( numeric_columns ) == Xk_numeric.shape[1]

    # -- Load R script via STAP
    _script_path: str = os.path.normpath(
        os.path.join(
            os.path.dirname( os.path.abspath( __file__ ) ),
            '..', '..', 'scripts',
            'scip.knockoffs.R',
        )
    )
    with open( _script_path ) as _f:
        _r_code: str = _f.read()

    # -- Draw one integer seed from Python's Generator to seed R's RNG
    _seed: int = int( rng.integers( 1, 2**31 - 1 ) )

    with ro.default_converter.context():
        _scip = rpackages.STAP( _r_code, "scip" )

        # -- Convert X to R data.frame (pandas2ri handles pl.Categorical -> R factor)
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            X_r = ro.conversion.get_conversion().py2rpy( X.to_pandas() )

        # -- Convert Xk_numeric to R matrix (positional, no column names needed)
        with (
            ro.default_converter + numpy2ri.converter
        ).context():
            Xk_numeric_r = ro.conversion.get_conversion().py2rpy(
                Xk_numeric.astype( np.float64 )
            )

        # -- Single R call: returns n x p data.frame of knockoffs
        with ( _r_warnings_to_stdout() if verbose > 0 else contextlib.nullcontext() ):
            Xk_r = _scip.scip_knockoffs_with_numeric(
                X           = X_r,
                Xk_numeric  = Xk_numeric_r,
                seed        = _seed,
                **{ k.replace( '_', '.' ): v for k, v in kwargs.items() },
            )

        # -- Convert R data.frame back to polars via pandas; pin dtypes to X.schema
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            Xk_pd: pd.DataFrame = ro.conversion.get_conversion().rpy2py( Xk_r )

    return pl.from_pandas( Xk_pd ).cast(
        { col: dtype for col, dtype in X.schema.items() }
    )
#/def get_knockoffs_with_Xk_numeric

def get_knockoffs_second_order_np(
    X: np.ndarray,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        :param X: One hot encoded numpy array of X, with drop_first
        :param kwargs: Passed to r knockoff::create_second_order, likely only "shrink"
        
        :returns: Knockoffs using `rKnockoff.create_second_order`, thus with continuous categorical values.
    """
    
    Xk: np.ndarray
    shrink: bool = kwargs.pop( 'shrink', True )
    with (
        ro.default_converter + numpy2ri.converter
    ).context():
        Xk = rKnockoff.create_second_order(
            X,
            shrink = shrink,
            **kwargs
        )
    #/with ( ro.default_converter + numpy2ri.converter )
    
    return Xk
#/def get_knockoffs_second_order_np

def get_knockoffs_SCIP(
    X: pl.DataFrame,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Full sequential SCIP knockoff generation (numeric + categorical).

        Each column's knockoff conditions on all original columns plus all
        previously generated knockoffs. The entire column loop runs in R
        (scip.knockoffs.R), loaded once per call via rpy2.robjects.packages.STAP.

        :param residuals_method: "normal" or "permute" — how to resample numeric residuals
        :param kwargs: Passed to r ranger::ranger
    """
    # -- Load R script via STAP
    _script_path: str = os.path.normpath(
        os.path.join(
            os.path.dirname( os.path.abspath( __file__ ) ),
            '..', '..', 'scripts',
            'scip.knockoffs.R',
        )
    )
    with open( _script_path ) as _f:
        _r_code: str = _f.read()

    # -- Draw one integer seed from Python's Generator to seed R's RNG
    _seed: int = int( rng.integers( 1, 2**31 - 1 ) )

    with ro.default_converter.context():
        _scip = rpackages.STAP( _r_code, "scip" )

        # -- Convert X to R data.frame (pandas2ri handles pl.Categorical -> R factor)
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            X_r = ro.conversion.get_conversion().py2rpy( X.to_pandas() )

        # -- Single R call: returns n x p data.frame of knockoffs
        with ( _r_warnings_to_stdout() if verbose > 0 else contextlib.nullcontext() ):
            Xk_r = _scip.scip_knockoffs(
                X                = X_r,
                residuals_method = residuals_method,
                seed             = _seed,
                **{ k.replace( '_', '.' ): v for k, v in kwargs.items() },
            )

        # -- Convert R data.frame back to polars via pandas; pin dtypes to X.schema
        with (
            ro.default_converter + pandas2ri.converter
        ).context():
            Xk_pd: pd.DataFrame = ro.conversion.get_conversion().rpy2py( Xk_r )

    return pl.from_pandas( Xk_pd ).cast(
        { col: dtype for col, dtype in X.schema.items() }
    )
#/def get_knockoffs_SCIP


def rangerGiniImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        :param kwargs: Passed to r knockoff::stat.random_forest
    """
    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )
    
    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")
    #
    
    # stat_forest_hetero_gini gives only W stats, not importances
    # As a result, concatenate with zeros. This gives the same result.
    
    _categories_override: dict[ str, list[ str ] ] = {
        col: sorted( X[ col ].cast( pl.Utf8 ).unique().drop_nulls().to_list() )
        for col in X.columns
        if X.schema[ col ] == pl.Categorical
    }

    X_ohe: np.ndarray = utilities.get_ohe_np(
        X = X,
        drop_first = True,
    )

    Xk_ohe: np.ndarray = utilities.get_ohe_np(
        X = Xk,
        drop_first = True,
        categories_override = _categories_override,
    )
    
    oheDict: dict[ str, int | tuple[ int,...] ] = utilities.get_oheDict(
        X = X,
        drop_first = True,
    )
    
    w_stats_raw: np.ndarray
    with ( _r_warnings_to_stdout() if verbose > 0 else contextlib.nullcontext() ):
        with ro.default_converter.context():
            if outcomeDescriptor.outcome_type == 'categorical':
                _y_series: pl.Series = y.to_series() if isinstance(y, pl.DataFrame) else y
                y_r = ro.FactorVector(_y_series.cast(pl.Utf8).to_list())
                with (
                    ro.default_converter + numpy2ri.converter
                ).context():
                    w_stats_raw = rKnockoff.stat_random_forest(
                        X = X_ohe,
                        X_k = Xk_ohe,
                        y = y_r,
                        **kwargs,
                    )
            else:
                _y_np: np.ndarray = (
                    y.to_numpy() if isinstance(y, np.ndarray)
                    else y.to_numpy() if isinstance(y, pl.Series)
                    else y.to_numpy().squeeze()
                )
                # rpy2 cannot convert unsigned integer dtypes
                if np.issubdtype(_y_np.dtype, np.unsignedinteger):
                    _y_np = _y_np.astype(np.int64)
                with (
                    ro.default_converter + numpy2ri.converter
                ).context():
                    w_stats_raw = rKnockoff.stat_random_forest(
                        X = X_ohe,
                        X_k = Xk_ohe,
                        y = _y_np,
                        **kwargs,
                    )
            #/if categorical/else
        #/with ro.default_converter
    #/with _r_warnings_to_stdout

    # Take max from categorical OHE columns, others literally
    w_stats: np.ndarray = np.fromiter(
        (
            w_stats_raw[ val ]\
                if isinstance( val, int )\
                else np.max( w_stats_raw[ list(val) ] )\
                for val in oheDict.values()
            #/
        ),
        dtype = float,
    )

    importances = np.concatenate(
        (
            w_stats,
            np.zeros_like( w_stats )
         ),
         axis = 0,
    )
    
    return importances
#/def rangerGiniImportances
