#
#//  rbridge.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 2/13/26.
#//

from . import utilities

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

from typing import Literal

if not rpackages.isinstalled('knockoff'):
    rutils = rpackages.importr('utils')
    rutils.install_packages(
        'knockoff',
    )
#

if not rpackages.isinstalled('ranger'):
    rutils = rpackages.importr('utils')
    rutils.install_packages(
        'knockoff',
    )
#


rKnockoff = rpackages.importr('knockoff')

rRanger = rpackages.importr('ranger')

rStats = rpackages.importr('stats')

# -- Category Choosing

def get_forest_expectations_for_column(
    X: pl.DataFrame,
    col: str,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        Using r ranger, calculate conditional expectations for a variable as a function of all others.
        
        :param kwargs: passed to r ranger::ranger
        :returns: Numpy array of size (n,)
    """
    assert X.schema[col] != pl.Categorical
    
    with (
        ro.default_converter\
            + pandas2ri.converter
        #/
    ).context():
        ro.globalenv['X.df'] = X.to_pandas()
        ro.globalenv['X.df.explanatory'] = X.drop(
            (col,)
        ).to_pandas()
    #
    
    ro.globalenv['.forest'] = rRanger.ranger(
        data = ro.globalenv['X.df'],
        dependent_variable_name = col,
        probability = False,
        respect_unordered_factors = True,
        oob_error = False,
        **kwargs,
    )
    
    ro.globalenv['.predictions'] = rStats.predict(
        ro.globalenv['.forest'],
        data = ro.globalenv['X.df.explanatory'],
        type = 'response',
    )
    
    # Set ro.globalenv['.preidctions.expectation']
    ro.r(
    """
.predictions.expectation <- .predictions$predictions
    """
    )
    
    expectation_predictions: np.ndarray
    with (
        ro.default_converter\
            + numpy2ri.converter
        #/
    ).context():
        expectation_predictions = ro.globalenv['.predictions.expectation']
    #/with (ro.default_converter + ... )
    
    return expectation_predictions
#/def get_forest_expectations_for_column

def get_forest_probabilities_for_column(
    X: pl.DataFrame,
    col: str,
    logit: bool = False,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs
    ) -> np.ndarray:
    """
        Using r ranger, calculate the probabilities for the categorical column `col` from `X` as a function of every other column.
        
        :param kwargs: Passed to r ranger::ranger
        :returns: Numpy array of size (n, k) where k is the number of categories in `X[col]`. If you want to drop first, do it elsewhere
    """
    assert X.schema[col] == pl.Categorical
    
    with (
        ro.default_converter\
            + pandas2ri.converter
        #/
    ).context():
        ro.globalenv['X.df'] = X.to_pandas()
        ro.globalenv['X.df.explanatory'] = X.drop(
            (col,)
        ).to_pandas()
    #

    ro.globalenv['.forest'] = rRanger.ranger(
        data = ro.globalenv['X.df'],
        dependent_variable_name = col,
        probability = True,
        respect_unordered_factors = True,
        oob_error = False,
        **kwargs,
    )
    
    ro.globalenv['.predictions'] = rStats.predict(
        ro.globalenv['.forest'],
        data = ro.globalenv['X.df.explanatory'],
        type = 'response',
    )
    
    # Set ro.globalenv['.preidctions.proba']
    ro.r(
    """
.predictions.proba <- .predictions$predictions
    """
    )

    proba_predictions: np.ndarray
    with (
        ro.default_converter\
            + numpy2ri.converter
        #/
    ).context():
        proba_predictions = ro.globalenv['.predictions.proba']
    #/with (ro.default_converter + ... )
    
    if logit:
        # Replace zero with smallest nonzero values
        #   per column
        # Soft maxing makes this work safely
        zeroMask: np.ndarray = (proba_predictions == 0.0)
        if np.any( zeroMask ):
            predictions_infMask: np.ndarray = np.where(
                zeroMask,
                np.inf,
                proba_predictions
            )
            
            # Smallest nonzeroes
            proba_min = np.min(predictions_infMask, axis=0)
            
            # Replace zeroes with smallest nonzeroes
            proba_predictions = np.where(
                zeroMask,
                proba_min,
                proba_predictions
            )
            
            # Take soft maxes
            proba_predictions = proba_predictions / np.sum(
                proba_predictions, axis = 1
            )[:,np.newaxis]
        #
        
        proba_predictions = np.log( proba_predictions )
    #/if logit
    
    return proba_predictions
#/def get_forest_probabilities_for_column

def get_ohe_forest_probabilities_np(
    X: pl.DataFrame,
    logit: bool = True,
    drop_first: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        Use r ranger to get probabilities, or log probabilities if logit is selected. If we have `logit` and `drop_first`, subtract the first column
    """
    columns_dict: dict[
        str, # categorical column in X
        np.ndarray # probabilities, perhaps with drop_first
    ] = {
        col: get_forest_probabilities_for_column(
            X = X,
            col = col,
            logit = logit,
            **kwargs
        ) for col, dtype in X.schema.items()\
            if dtype == pl.Categorical
        #/
    }
    
    if drop_first:
        if logit:
            # Subtract first column and drop it
            columns_dict = {
                col: val[:,1:] - val[:,0:1]\
                    for col, val in columns_dict.items()
            }
        #
        else:
            # Just drop first
            columns_dict = {
                col: val[:,1:]\
                    for col, val in columns_dict.items()
            }
        #/if drop_first/else
    #/if drop_first/else
    
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
        Uses r ranger::ranger for a series of forests to get conditional expectations for each numeric `X`
        
        :param kwargs: Passed to r ranger::ranger
        :returns: Data Frame, with non categorical columns of X, with conditional expectations
    """
    with (
        ro.default_converter\
            + pandas2ri.converter
        #/
    ).context():
        ro.globalenv['X.df'] = X.to_pandas()
    #
    
    # Build dictionary of col -> expectation
    conditional_expectations_dict: dict[ str, np.ndarray ] = {}
    for col, dtype in X.schema.items():
        if dtype == pl.Categorical:
            continue
        #
        with (
            ro.default_converter\
                + pandas2ri.converter
            #/
        ).context():
            ro.globalenv['X.df.explanatory'] = X.drop(
                (col,)
            ).to_pandas()
        #
        
        ro.globalenv['.forest'] = rRanger.ranger(
            data = ro.globalenv['X.df'],
            dependent_variable_name = col,
            probability = False,
            respect_unordered_factors = True,
            oob_error = False,
            **kwargs,
        )
        
        ro.globalenv['.predictions'] = rStats.predict(
            ro.globalenv['.forest'],
            data = ro.globalenv['X.df.explanatory'],
            type = 'response',
        )
        
        # Set ro.globalenv['.preidctions.proba']
        ro.r(
        """
.predictions.expectation <- .predictions$predictions
        """
        )
        
        # Put the numpy array in the dictionary
        with (
            ro.default_converter\
                + numpy2ri.converter
            #/
        ).context():
            conditional_expectations_dict[col] = ro.globalenv['.predictions.expectation']
        #/with (ro.default_converter + ... )
    #/for col, dtype in X.schema.items()
    
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
        Implements the categorical SCIP method
        
        :param kwargs: Passed to r ranger::ranger
    """
    numeric_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items() if dtype != pl.Categorical
    )
    
    assert len(numeric_columns) == Xk_numeric.shape[1]
    
    # Get knockoffs so far, just the numeric ones
    scip_df: pl.DataFrame = pl.concat(
        (
            X, pl.DataFrame(
                {
                    "{}~".format(numeric_columns[j]): Xk_numeric[:,j]\
                        for j in range( Xk_numeric.shape[1] )
                    #/
                },
                schema = {
                    "{}~".format(col): X.schema[col] for col in numeric_columns
                },
            )
        ),
        how = 'horizontal',
    )
    
    # Add in the categorical columns by calculating predictions
    probabilities: np.ndarray
    for col in X.columns:
        if X.schema[col] != pl.Categorical:
            continue
        #
        
        # Note the categorical variable is already in as `col`
        probabilities = get_forest_probabilities_for_column(
            X = scip_df,
            col = col,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            **kwargs,
        )
        
        # Choose categories
        scip_df = scip_df.with_columns(
            utilities.makeChoices_ohe(
                X = probabilities,
                categories = X[col].cat.get_categories().sort(),
                name = "{}~".format(col),
                method = 'softmax',
                drop_first = False,
                rng = rng,
            )
        )
    #/for col in X.columns
    
    # Now we have all the choices made; extract only the knockoffs
    return scip_df.select(
        **{
            col: pl.col("{}~".format(col))\
                for col in X.columns
            #/
        }
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
    with (
        ro.default_converter + numpy2ri.converter
    ).context():
        Xk = rKnockoff.create_second_order(
            X,
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
        :param method: How do deal with numeric residuals
        :param kwargs: Passed to r ranger::ranger
    """
    
    n: int = X.shape[0]
    
    scip_df: pl.DataFrame = X

    probabilities: np.ndarray
    conditional_expectations: np.ndarray
    conditional_residuals: np.ndarray
    numeric_knockoffs: np.ndarray
    
    for col in X.columns:
        if X.schema[col] == pl.Categorical:
            probabilities = get_forest_probabilities_for_column(
                X = scip_df,
                col = col,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                **kwargs,
            )
            
            # Choose categories
            scip_df = scip_df.with_columns(
                utilities.makeChoices_ohe(
                    X = probabilities,
                    categories = X[col].cat.get_categories().sort(),
                    name = "{}~".format(col),
                    method = 'softmax',
                    drop_first = False,
                    rng = rng,
                )
            )
        #
        else:
            # Numeric
            conditional_expectations = get_forest_expectations_for_column(
                X = scip_df,
                col = col,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                **kwargs,
            )
            conditional_residuals = X[col] - conditional_expectations
            
            if residuals_method == 'normal':
                numeric_knockoffs = conditional_expectations + rng.normal(
                    loc = 0.0,
                    scale = np.std(
                        conditional_residuals,
                        ddof = 1,
                    ),
                    size = n,
                )
            #
            elif residuals_method == 'permute':
                numeric_knockoffs = conditional_expectations\
                    + rng.permutation(
                        conditional_residuals
                    )
                #/
            #
            else:
                raise ValueError("Unreocgnized method={}".format(method))
            #
            
            scip_df = scip_df.with_columns(
                pl.Series(
                    name = "{}~".format(col),
                    values = numeric_knockoffs,
                    dtype = X.schema[col],
                )
            )
        #/switch X.schema[col]
    #/for col in X.columns
    
    # Select knockoffs as original column names
    return scip_df.select(
        **{
            col: pl.col("{}~".format(col))\
                for col in X.columns
            #/
        }
    )
#/def get_knockoffs_SCIP

# -- Importances

def rangerMaldImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    bandwidth: float = 1.0,
    bandwidth_exponent: float = 0.2,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    """
        :param kwargs: Passed to r ranger::ranger
    """
    # FUTURE: categorical outcome
    assert X.schema == Xk.schema
    X_all: pl.DataFrame = pl.concat(
        [
            X,
            Xk.rename({col: "{}~".format(col) for col in X.columns})
        ],
        how = 'horizontal',
    )
    
    probability = False
    with (
        ro.default_converter\
            + pandas2ri.converter\
            + numpy2ri.converter
        #/
    ).context():
        ro.globalenv['X.df'] = X_all.to_pandas()
        ro.globalenv['y'] = y.to_numpy()
        
        #print(ro.globalenv['y'])
        #print(ro.globalenv['y'].shape)
        
    #/with( ro.default_converter + ... )
    
    ro.globalenv['.forest'] = rRanger.ranger(
        x = ro.globalenv['X.df'],
        y = ro.globalenv['y'],
        probability = probability,
        respect_unordered_factors = True,
        oob_error = False,
        **kwargs,
    )
    
    # Get original predictions
    ro.globalenv['.predictions.base'] = rStats.predict(
        ro.globalenv['.forest'],
        data = ro.globalenv['X.df'],
        type = 'response',
    )
    
    ro.r(
    """
.predictions.base.num <- .predictions.base$predictions
    """
    )

    importances_pointwise: np.ndarray = np.zeros(
        shape = X_all.shape,
        dtype = float,
    )
    
    col: str
    categories: pl.Series
    predictions: np.ndarray
    bandwidth_literal: float
    
    for j in range( X_all.shape[1] ):
        col = X_all.columns[j]
        
        if X_all.schema[col] == pl.Categorical:
            categories = X_all[col].cat.get_categories().sort()
            predictions = np.zeros(
                shape = ( X_all.shape[0], len( categories ), )
            )
            
            # Set X[col] to each category, test predictions
            for k in range( len( categories ) ):
                with (
                    ro.default_converter\
                        + pandas2ri.converter
                    #/
                ).context():
                    ro.globalenv['X.modified'] = X_all.with_columns(
                        **{
                            col: pl.lit( categories[k] ).cast(
                                pl.Categorical
                            )
                        }
                    ).to_pandas()
                #/with( ro.default_converter + ... )
                
                ro.globalenv['.predictions.modified'] = rStats.predict(
                    ro.globalenv['.forest'],
                    data = ro.globalenv['X.modified'],
                    type = 'response',
                )
                
                
                ro.r(
                """
.predictions.modified.val <- .predictions.modified$predictions
                """
                )
                
                with (
                    ro.default_converter\
                        + numpy2ri.converter
                    #/
                ).context():
                    predictions[:,k] = ro.globalenv['.predictions.modified.val']
                #/with (ro.default_converter + ... )
                
                importances_pointwise[:,j] = np.max(
                    predictions, axis = 1,
                ) - np.min(
                    predictions, axis = 1
                )
            #/for k in range( len( categories) )
        #
        else:
            # Numeric
            bandwidth_literal = bandwidth*np.std(
                X_all[col].to_numpy()
            )/(X_all.shape[0]**bandwidth_exponent)
            
            # Nudge X, estimate derivative
            with (
                ro.default_converter\
                    + pandas2ri.converter
                #/
            ).context():
                ro.globalenv['X.modified'] = X_all.with_columns(
                    **{
                        col: pl.col( col ) + bandwidth_literal
                    }
                ).to_pandas()
            #/with( ro.default_converter + ... )
            
            ro.globalenv['.predictions.modified'] = rStats.predict(
                ro.globalenv['.forest'],
                data = ro.globalenv['X.modified'],
                type = 'response',
            )
            
            ro.r(
            """
.predictions.modified.val <- .predictions.modified$predictions
            """
            )
            
            with (
                ro.default_converter\
                    + numpy2ri.converter
                #/
            ).context():
                importances_pointwise[:,j] = (ro.globalenv['.predictions.modified.val']\
                    - ro.globalenv['.predictions.base.num']
                #/
                )/bandwidth_literal
            #/with (ro.default_converter + ... )
        #/if X.schema[col] == pl.Categorical/else
    #/for j in range( X.shape[1] )
    
    importances: np.ndarray = np.mean(
        np.abs( importances_pointwise )**exponent,
        axis = 0,
    )
    
    return importances
#/def rangerMaldImportances

def rangerGiniImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        :param kwargs: Passed to r knockoff::stat.random_forest
    """
    
    # stat_forest_hetero_gini gives only W stats, not importances
    # As a result, concatenate with zeros. This gives the same result.
    
    X_ohe: np.ndarray = utilities.get_ohe_np(
        X = X,
        drop_first = True,
    )
    
    Xk_ohe: np.ndarray = utilities.get_ohe_np(
        X = Xk,
        drop_first = True,
    )
    
    oheDict: dict[ str, int | tuple[ int,...] ] = utilities.get_oheDict(
        X = X,
        drop_first = True,
    )
    
    w_stats_raw: np.ndarray
    with (
        ro.default_converter + numpy2ri.converter
    ).context():
        w_stats_raw = rKnockoff.stat_random_forest(
            X = X_ohe,
            X_k = Xk_ohe,
            y = y.to_numpy(),
            **kwargs,
        )
    #/with (r.default_converter + ... )
    
    # Take max from categorical, others literally
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
