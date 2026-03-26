#
#//  utilities.py
#//  rangerknockoffpy
#//
#//  Created by Evan Mason on 2/10/26.
#//

import polars as pl
import numpy as np

from typing import Literal

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
    drop_first = True
    ) -> dict[ str, int | tuple[ int,... ] ]:
    """
        :param drop_first: Have k-1 columns when k categories
        
        Result maps original column names to single column if numeric, tuple of OHE columns for categories
        
        If drop_first, it does number of categories minus 1
    """
    counts_adjust: int = 1 if drop_first else 0
    
    ohe_dict: dict[ int, int | tuple[ int,... ] ] = {}
    col_iterator: int = 0
    
    for col in X.columns:
        if X.schema[ col ] == pl.Categorical:
            # OHE columns = value_counts - 1
            value_counts = len( X[ col ].value_counts() )
            ohe_dict[ col ] = tuple(
                range(
                    col_iterator,
                    col_iterator + value_counts - counts_adjust
                )
            )
            col_iterator += ( value_counts - counts_adjust )
        #
        else:
            # Numeric, one column
            ohe_dict[ col ] = col_iterator
            col_iterator += 1
        #/if X.schema[ col ] == pl.Categorical
    #/for col in X.columns
    
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
        name = name,
        values = (
            categories[ int(k) ] for k in indices
        ),
        dtype = pl.Categorical,
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
    **kwargs
    ) -> pl.DataFrame:
    """
        :param **kwargs: Passed to `X.to_dummies`. Most likely:
            - drop_null: bool = False
    """
    separator: str = '_'
    categorical_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items()\
            if dtype == pl.Categorical
        #/
    )
    
    categories_dict: dict[ str, pl.Series ] = {
        col: X[col].cat.get_categories().sort()\
            for col in categorical_columns
        #/
    }
            
    # Hack to drop_first correctly: keep all dummies then drop the first category
    X_keep = X.with_columns(
        (
            pl.col( col ).cast(
                pl.Enum(
                    categories_dict[ col ]
                )
            ) for col in categorical_columns
        )
    ).to_dummies(
        categorical_columns,
        drop_first = False,
        separator = '_',
    )
    
    if drop_first:
        return X_keep.drop(
            (
                # drop '{col}_{first value}
                "{}{}{}".format(
                    col,
                    separator,
                    categories_dict[col][0],
                ) for col in categorical_columns
            )
        )
    #
    else:
        return X_keep
    #
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

def get_linear_probabilities_for_column(
    X: pl.DataFrame,
    col: str,
    logit: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> np.ndarray:
    """
        Calculate probabilities sklearn.linear_model.LogisticRegression
        
        Always gives all probabilities, not dropping any
        
        :param kwargs: Passed to sklearn.linear_model.LogisticRegression
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
    
    model.fit( _X, _y )
    
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
    **kwargs,
    ) -> np.ndarray:
    """
        Use sklearn logistic regression to convert categorical columns to probabilities, or log probabilities if `logit`, in which case we take the logs of each, subtracting the first column if `drop_first`
        
        :param kwargs: Passed  to sklearn.linear_model.LogisticRegression
    """
    columns_dict: dict[
        str, # categorical column in X
        np.ndarray # probabilities, perhaps with drop_first
    ] = {
        col: get_linear_probabilities_for_column(
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
#/def get_ohe_probabilities_np
