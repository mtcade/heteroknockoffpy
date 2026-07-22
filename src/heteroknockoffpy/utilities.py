#
#//  utilities.py
#//  rangerknockoffpy
#//
#//  Created by Evan Mason on 2/10/26.
#//

import os

import polars as pl
import numpy as np

from typing import Literal, Self
from dataclasses import dataclass

DataFrameLike = pl.DataFrame | str | os.PathLike
SeriesOrDataFrameLike = pl.Series | pl.DataFrame | str | os.PathLike

def _resolve_df(x: DataFrameLike) -> pl.DataFrame:
    if isinstance(x, pl.DataFrame):
        return x
    return pl.read_parquet(x)

def _resolve_y(y: SeriesOrDataFrameLike) -> pl.Series | pl.DataFrame:
    if isinstance(y, (str, os.PathLike)):
        return pl.read_parquet(y)
    return y

@dataclass
class OutcomeDescriptor:
    outcome_dimension: Literal['single','multi',]
    outcome_type: Literal['continuous','count','categorical',]
    
    @classmethod
    def infer(
        cls,
        y: np.ndarray | pl.DataFrame | pl.Series,
        outcome_type: Literal['continuous','count','categorical',] | None
        ) -> Self:
        if outcome_type is None:
            if isinstance( y, pl.Series ):
                if y.dtype.is_float():
                    return cls(
                        outcome_dimension = 'single',
                        outcome_type = 'continuous',
                    )
                #
                elif y.dtype.is_integer():
                    if not ( y >= 0 ).all():
                        raise ValueError(
                            "Require nonnegative y entries for integer values"
                        )
                    #
                    return cls(
                        outcome_dimension = 'single',
                        outcome_type = 'count',
                    )
                #
                elif isinstance( y.dtype, ( pl.Categorical, pl.Enum ) ):
                    return cls(
                        outcome_dimension = 'single',
                        outcome_type = 'categorical',
                    )
                #
                else:
                    raise TypeError(
                        "Unrecognized y.dtype={}".format( y.dtype )
                    )
                #/switch y.dtype
            elif isinstance( y, pl.DataFrame ):
                # Check for homogeneity
                if not len( set( y.dtypes ) ) == 1:
                    raise ValueError(
                        "Require same dtype for each y column; found y.dtypes={}".format(y.dtypes)
                    )
                #
                dtype = y.dtypes[0]
                if dtype.is_float():
                    return cls.infer(
                        y = y,
                        outcome_type = 'continuous',
                    )
                #
                elif dtype.is_integer():
                    if not ( y >= 0 ).to_series().all():
                        raise ValueError(
                            "Require nonnegative y entries for integer values"
                        )
                    #/if not ( y >= 0 ).to_series().all()
                    return cls.infer(
                        y = y,
                        outcome_type = 'count',
                    )
                #
                elif isinstance( dtype, ( pl.Categorical, pl.Enum ) ):
                    return cls.infer(
                        y = y,
                        outcome_type = 'categorical',
                    )
                #
                else:
                    raise TypeError(
                        "Unrecognized dtype={}".format( dtype )
                    )
                #/switch y.dtype
            #
            elif isinstance( y, np.ndarray ):
                if np.issubdtype( y, np.float ):
                    return cls.infer(
                        y = y,
                        outcome_type = 'continuous',
                    )
                elif np.issubdtype( y, np.integer ):
                    if not np.all( y >= 0 ):
                        raise ValueError(
                            "Require nonnegative y entries for integer values"
                        )
                    #/if not np.all( y >= 0 )
                    return cls.infer(
                        y = y,
                        outcome_type = 'count',
                    )
                else:
                    raise TypeError(
                        "Unrecognized y.dtype={}".format( y.dtype )
                    )
                #/switch np.issubdtype( y, ... )
            #
            else:
                raise TypeError(
                    "Unrecognized type(y)={}".format(type(y))
                )
            #/switch type(y)
        #
        else:
            _width: int = y.shape[1] if len( y.shape ) > 1 else 1
            if _width > 1:
                return cls(
                    outcome_dimension = 'multi',
                    outcome_type = outcome_type,
                )
            #
            else:
                return cls(
                    outcome_dimension = 'single',
                    outcome_type = outcome_type,
                )
            #/if _width > 1
        #/if outcome_type is None
    #/def infer
#/class OutcomeDescriptor

def choices_from_weights(
    X: np.ndarray,
    rng: np.random.Generator,
    ) -> np.ndarray:
    # Make rowwise probability choices
    return np.fromiter(
        (
            rng.choice(
                X.shape[1],
                p = X[i,:],
            ) for i in range( X.shape[0] )
        ),
        dtype = int,
    )
#/def choices_from_weights

def get_oheDict(
    X: pl.DataFrame,
    drop_first: bool = True,
    categories_override: dict[ str, list[ str ] ] | None = None,
    ) -> dict[ str, int | tuple[ int,... ] ]:
    """
        Map each column in X to its position(s) in the OHE array.

        Numeric columns map to a single int index; categorical columns map to
        a tuple of ints (one per dummy column after optional drop_first).

        Matches `get_ohe_df`'s canonical column order: all non-categorical
        columns first (in their original relative order), then each
        categorical column's dummy block in turn -- NOT X.columns' original
        interleaved order, since `to_dummies` moves dummy columns to the end.

        :param categories_override: When provided, use this as the category
            count for the named categorical columns instead of the unique values
            present in X.  Must match the override passed to get_ohe_df/np.
    """
    counts_adjust: int = 1 if drop_first else 0

    ohe_dict: dict[ str, int | tuple[ int,... ] ] = {}
    col_iterator: int = 0

    non_categorical_columns: list[ str ] = [
        col for col in X.columns if X.schema[ col ] != pl.Categorical
    ]
    categorical_columns: list[ str ] = [
        col for col in X.columns if X.schema[ col ] == pl.Categorical
    ]

    for col in non_categorical_columns:
        ohe_dict[ col ] = col_iterator
        col_iterator += 1
    #/for col in non_categorical_columns

    for col in categorical_columns:
        if categories_override and col in categories_override:
            n_cats = len( categories_override[ col ] )
        else:
            n_cats = X[ col ].cast( pl.Utf8 ).n_unique()
        ohe_dict[ col ] = tuple(
            range( col_iterator, col_iterator + n_cats - counts_adjust )
        )
        col_iterator += ( n_cats - counts_adjust )
    #/for col in categorical_columns

    return ohe_dict
#/def get_oheDict

def makeChoices_ohe(
    X: np.ndarray,
    categories: pl.Series,
    name: str | None = None,
    method: Literal['max','softmax',] = 'max',
    drop_first: bool = True,
    rng: np.random.Generator | None = None,
    ) -> pl.Series:
    """
        :param X: A subset of some array of data, which contains weights or probabilities for some one hot encoding of a categorical variable. Width is number of categories, width + 1 if `drop_first`
        :param name: Name for the series, passed to `pl.Series`
    """
    indices: np.ndarray

    if drop_first:
        X = np.concatenate(
            (
                (
                    1 - np.sum(X, axis = 1)
                ).reshape( -1, 1),
                X,
            ),
            axis = 1,
        )
    #/if drop_first
    
    if method == 'max':
        indices = np.argmax(
            X, axis = 1,
        )
    #
    elif method == 'softmax':
        indices = choices_from_weights(
            X / np.sum( X, axis = 1 ).reshape( -1,1 ),
            rng = rng,
        )
    #
    else:
        raise ValueError("Unrecognized method={}".format(method))
    #/switch method
    
    choices: pl.Series = pl.Series(
        name   = name,
        values = ( str( categories[ int(k) ] ) for k in indices ),
        dtype  = pl.Categorical,
    )

    return choices
#/def makeChoices_ohe

def collapse_ohe(
    X: pl.DataFrame,
    X_ohe: np.ndarray,
    oheDict: dict[ str, int | tuple[ int,... ] ] | None = None,
    method: Literal['max','softmax',] = 'max',
    logit: bool = False,
    drop_first: bool = True,
    rng: np.random.Generator | None = None,
    ) -> pl.DataFrame:
    """
        :param X: Original data, with potentially categorical columns. Used to get column names and categories, not for any direct data
        :param Xk: Knockoff data including one hot encoded or probability data
        :param oheDict: Maps columns from X to columns in X_ohe
        :param method: Whether to take largest probability, or pick according to probabilities by using softmax
        :param logit: If the categories are in log probabilities. If so, takes the exponent. 
    """
    
    if oheDict is None:
        oheDict: dict[ str, int | tuple[ int,... ] ] = get_oheDict( X )
    #
    
    categories_dict: dict[ str, pl.Series ] = {
        col: X[col].cat.get_categories().sort()\
            for col, val in oheDict.items()\
            if isinstance( val, tuple )
        #/
    }
    
    if logit:
        if drop_first:
            # Take first ohe column to be 0 of log weights
            # Then its weight is 1
            return pl.DataFrame(
                {
                    col: (
                        makeChoices_ohe(
                            X = np.concatenate(
                                (
                                    np.ones(
                                        # First category
                                        # exp( 0 )
                                        (X_ohe.shape[0],1,),
                                    ),
                                    np.exp(
                                        # logit weights
                                        X_ohe[:, val ]
                                    )
                                ),
                                axis = 1
                            ),
                            categories = categories_dict[ col ],
                            method = method,
                            drop_first = False,
                            rng = rng,
                        ) if isinstance( val, tuple )\
                            else pl.Series(
                                # Literal X values since it's numeric
                                values = X_ohe[:,val]
                            )
                        #/
                    ) for col, val in oheDict.items()
                },
                schema = X.schema,
            )
        #
        else:
            # Log weights as is
            return pl.DataFrame(
                {
                    col: (
                        makeChoices_ohe(
                            X = np.exp(
                                X_ohe[:, val ]
                            ),
                            categories = categories_dict[ col ],
                            method = method,
                            drop_first = False,
                            rng = rng,
                        ) if isinstance( val, tuple )\
                            else pl.Series(
                                # Literal X values since it's numeric
                                values = X_ohe[:,val]
                            )
                        #/
                    ) for col, val in oheDict.items()
                },
                schema = X.schema,
            )
        #/if drop_first/else
    #
    else:
        return pl.DataFrame(
            {
                col: (
                    makeChoices_ohe(
                        X = X_ohe[:, val ],
                        categories = categories_dict[ col ],
                        method = method,
                        drop_first = drop_first,
                        rng = rng,
                    ) if isinstance( val, tuple )\
                        else pl.Series(
                            values = X_ohe[:,val]
                        )
                    #/
                ) for col, val in oheDict.items()
            },
            schema = X.schema,
        )
    #/if logit/else
#/def collapse_ohe

def get_ohe_df(
    X: pl.DataFrame,
    drop_first: bool = True,
    categories_override: dict[ str, list[ str ] ] | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        One-hot encode all pl.Categorical columns in X, dropping the first
        lexically-sorted category when drop_first=True.

        :param categories_override: When provided, specifies the full sorted
            category list for each column.  Any category not present in X's
            data will be added as an all-zero dummy column.  Use this to keep
            Xk's OHE consistent with X's OHE when Xk may be missing some
            category values.
    """
    separator: str = '_'
    categorical_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items()
        if dtype == pl.Categorical
    )

    if not categorical_columns:
        return X

    # Use present values to determine category set (avoids global-cache bleed
    # where polars 1.x merges Categorical vocabularies across series/frames).
    categories_dict: dict[ str, list[ str ] ] = {
        col: (
            categories_override[ col ]
            if categories_override and col in categories_override
            else sorted( X[ col ].cast( pl.Utf8 ).unique().drop_nulls().to_list() )
        )
        for col in categorical_columns
    }

    X_keep: pl.DataFrame = X.to_dummies(
        list( categorical_columns ),
        drop_first = False,
        separator  = separator,
    )

    # Add all-zero columns for categories present in the override but absent
    # from the data (e.g. rare categories that knockoff never sampled).
    n = len( X )
    for col in categorical_columns:
        for cat in categories_dict[ col ]:
            dummy_name = "{}{}{}".format( col, separator, cat )
            if dummy_name not in X_keep.columns:
                X_keep = X_keep.with_columns(
                    pl.lit( 0, dtype = pl.UInt8 ).alias( dummy_name )
                )
    #/for col

    # Select columns in canonical order: non-categorical originals first, then
    # for each categorical column its dummies in sorted-category order (with
    # the first category omitted when drop_first=True).
    final_cols: list[ str ] = [
        c for c in X.columns if X.schema[ c ] != pl.Categorical
    ]
    for col in categorical_columns:
        cats = categories_dict[ col ]
        start = 1 if drop_first else 0
        final_cols.extend(
            "{}{}{}".format( col, separator, cat )
            for cat in cats[ start: ]
        )

    return X_keep.select( final_cols )
#/def get_ohe_df

def get_ohe_np(
    X: pl.DataFrame,
    drop_first: bool = True,
    **kwargs,
    ) -> np.ndarray:
    """
        :param **kwargs: Passed to `X.to_dummies`. Most likely:
            - drop_null: bool = False
            
            separator: str = '_' hardly matters since we just return a numpy array
    """
    return get_ohe_df(
        X = X,
        drop_first = drop_first,
        **kwargs,
    ).to_numpy()
#/def get_ohe_np

@dataclass
class AR1SimpleCase:
    X: pl.DataFrame
    Xk: pl.DataFrame
    y: np.ndarray
    oracle: np.ndarray
    function_str: str
    relevant_vars: list[ str ]
#/class AR1SimpleCase

def get_ar1_simple_case(
    n: int,
    p_numeric: int,
    p_categorical: int,
    categories: int,
    rho: float,
    p_relevant: float,
    noise_sd: float = 1.0,
    oracle_knockoffs: bool = False,
    rng: np.random.Generator | None = None,
    knockoff_kwargs: dict | None = None,
    ) -> AR1SimpleCase:
    """
        Build a simple mixed numeric/categorical knockoff test case from an
        AR1 latent Gaussian model, with a known oracle E[y|X].

        Numeric columns are latent values directly; categorical columns are
        sampled from drop_first one-hot log weights (i.e. the latent values
        for a categorical column are treated as logits for all categories
        except the first, whose logit is implicitly 0).

        The oracle mean function interacts groups of 2-3 relevant variables
        (never an isolated main effect, never a full power set), with each
        additive term's coefficient chosen so every term has ~unit variance.

        :param p_relevant: Proportion (rounded) of numeric and, separately,
            categorical variables that are relevant to the oracle.
        :param oracle_knockoffs: If True, build numeric second-order
            knockoffs of the raw Gaussian AR1 latent array itself, then
            collapse both X and Xk from the latent/knockoff-latent arrays --
            the statistically exact construction, since the true covariance
            is known. If False, collapse X alone and derive Xk from it via
            `knockoff.get_knockoffs`.
        :param knockoff_kwargs: Passed to `knockoff.get_knockoffs` (if
            `not oracle_knockoffs`) or to `rbridge.get_knockoffs_second_order_np`
            (if `oracle_knockoffs`).
    """
    import sympy as sp
    from sympypl import sympy_to_pl_expr, CatMap
    from . import knockoff

    if rng is None:
        rng = np.random.default_rng()
    #

    knockoff_kwargs = knockoff_kwargs or {}

    numeric_names: list[ str ] = [ 'num_{}'.format(i) for i in range(p_numeric) ]
    categorical_names: list[ str ] = [ 'cat_{}'.format(i) for i in range(p_categorical) ]
    category_labels: list[ str ] = [ str(i) for i in range(categories) ]

    n_rel_numeric: int = round( p_relevant * p_numeric )
    n_rel_categorical: int = round( p_relevant * p_categorical )
    if n_rel_numeric + n_rel_categorical < 2:
        raise ValueError(
            "Need at least 2 relevant variables to form an interaction; "
            "got {}".format( n_rel_numeric + n_rel_categorical )
        )
    #

    L: int = p_numeric + p_categorical * ( categories - 1 )

    latent_idx = np.arange( L )
    Sigma: np.ndarray = rho ** np.abs( latent_idx[:,None] - latent_idx[None,:] )
    chol: np.ndarray = np.linalg.cholesky( Sigma )
    Z: np.ndarray = rng.standard_normal( (n, L) ) @ chol.T

    placeholder: pl.DataFrame = pl.DataFrame(
        {
            **{
                name: [ 0.0 ] * categories for name in numeric_names
            },
            **{
                name: pl.Series( category_labels, dtype = pl.Utf8 ).cast( pl.Categorical )\
                    for name in categorical_names
            },
        }
    )
    oheDict: dict[ str, int | tuple[ int,... ] ] = get_oheDict( placeholder, drop_first = True )

    if oracle_knockoffs:
        from . import rbridge

        Zk: np.ndarray = rbridge.get_knockoffs_second_order_np(
            X = Z,
            **knockoff_kwargs,
        )

        X: pl.DataFrame = collapse_ohe(
            X = placeholder, X_ohe = Z, oheDict = oheDict,
            method = 'softmax', logit = True, drop_first = True, rng = rng,
        )
        Xk: pl.DataFrame = collapse_ohe(
            X = placeholder, X_ohe = Zk, oheDict = oheDict,
            method = 'softmax', logit = True, drop_first = True, rng = rng,
        )
    #
    else:
        X: pl.DataFrame = collapse_ohe(
            X = placeholder, X_ohe = Z, oheDict = oheDict,
            method = 'softmax', logit = True, drop_first = True, rng = rng,
        )
        Xk: pl.DataFrame = knockoff.get_knockoffs(
            X, method = 'second_order', rng = rng,
            categorical_method = 'ohe', **knockoff_kwargs,
        )
    #/if oracle_knockoffs/else

    relevant_vars: list[ str ] = sorted(
        rng.choice( numeric_names, size = n_rel_numeric, replace = False )
    ) + sorted(
        rng.choice( categorical_names, size = n_rel_categorical, replace = False )
    )

    groups: list[ list[ str ] ] = []
    k: int = len( relevant_vars )
    i: int = 0
    if k % 2 == 1:
        while i < k - 3:
            groups.append( relevant_vars[ i:i+2 ] )
            i += 2
        #
        groups.append( relevant_vars[ i:i+3 ] )
    #
    else:
        while i < k:
            groups.append( relevant_vars[ i:i+2 ] )
            i += 2
        #
    #/if k % 2 == 1/else

    bindings: dict[ sp.Symbol, str ] = {}
    reps: dict[ str, sp.Expr ] = {}
    for name in relevant_vars:
        symbol: sp.Symbol = sp.Symbol( name )
        bindings[ symbol ] = name
        if name in categorical_names:
            sorted_labels: list[ str ] = sorted( X[ name ].cat.get_categories().to_list() )
            codes: dict[ str, int ] = { label: idx for idx, label in enumerate( sorted_labels ) }
            realized_codes: np.ndarray = X[ name ].cast( pl.Utf8 ).replace_strict( codes ).to_numpy().astype( float )
            code_mean: float = realized_codes.mean()
            code_std: float = realized_codes.std()
            reps[ name ] = CatMap.from_dict(
                symbol,
                {
                    label: ( code - code_mean ) / code_std\
                        for label, code in codes.items()
                },
            )
        #
        else:
            col_mean: float = X[ name ].mean()
            col_std: float = X[ name ].std()
            reps[ name ] = ( symbol - col_mean ) / col_std
        #/if name in categorical_names/else
    #/for name in relevant_vars

    final_terms: list[ sp.Expr ] = []
    for group in groups:
        term_expr: sp.Expr = sp.Mul( *( reps[ name ] for name in group ) )
        raw_values: np.ndarray = X.select(
            sympy_to_pl_expr( term_expr, bindings ).alias( '_term' )
        )[ '_term' ].to_numpy()
        term_std: float = raw_values.std()
        coef: float = round( 1.0 / term_std, 2 )
        final_terms.append( sp.Float( coef ) * term_expr )
    #/for group in groups

    oracle_expr: sp.Expr = sp.Add( *final_terms )
    oracle: np.ndarray = X.select(
        sympy_to_pl_expr( oracle_expr, bindings ).alias( '_oracle' )
    )[ '_oracle' ].to_numpy()

    y: np.ndarray = oracle + rng.normal( 0, noise_sd, n )

    return AR1SimpleCase(
        X = X,
        Xk = Xk,
        y = y,
        oracle = oracle,
        function_str = str( oracle_expr ),
        relevant_vars = relevant_vars,
    )
#/def get_ar1_simple_case

def get_linear_probabilities_for_column(
    X: pl.DataFrame,
    col: str,
    logit: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Calculate probabilities sklearn.linear_model.LogisticRegression
        
        Always gives all probabilities, not dropping any
        
        :param kwargs: Passed to sklearn.linear_model.LogisticRegression
        :param weight: Optional length-n sample weight, forwarded to
            LogisticRegression.fit(sample_weight=weight). None (default)
            fits unweighted.
    """
    from sklearn.linear_model import LogisticRegression
    
    _X: np.ndarray = get_ohe_np(
        X.drop(col),
        drop_first = True
    )
    _y: pl.Series = X[col].cast( pl.Int32 )
    
    model = LogisticRegression(
        **{
            'solver': 'lbfgs',
            'max_iter': 4000,
        } | kwargs
    )
    
    model.fit( _X, _y, sample_weight = weight )
    
    if logit:
        return model.predict_log_proba( _X )
    else:
        return model.predict_proba( _X )
    #/if logit/else
#/def get_linear_probabilities_for_column

def get_ohe_linear_probabilities_np(
    X: pl.DataFrame,
    logit: bool = True,
    drop_first: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Use sklearn logistic regression to convert categorical columns to probabilities, or log probabilities if `logit`, in which case we take the logs of each, subtracting the first column if `drop_first`
        
        :param kwargs: Passed  to sklearn.linear_model.LogisticRegression
        :param weight: Optional length-n sample weight, forwarded to each
            per-column LogisticRegression fit. None (default) fits unweighted.
    """
    columns_dict: dict[
        str, # categorical column in X
        np.ndarray # probabilities, perhaps with drop_first
    ] = {
        col: get_linear_probabilities_for_column(
            X = X,
            col = col,
            logit = logit,
            weight = weight,
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
#/def get_ohe_probabilities_np
