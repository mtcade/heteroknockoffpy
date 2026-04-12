#
#//  PoissonLasso.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 3/31/26.
#//

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold

from typing import Self

class PoissonLassoCV:
    """
    Sklearn-style cross-validated Poisson GLM with L1 (LASSO) penalty.

    Uses statsmodels GLM with family=Poisson and fit_regularized(L1_wt=1.0) for the
    actual fit; sklearn KFold for cross-validating the regularization coefficient alpha.

    Attributes set after fit():
        coef_  : np.ndarray — estimated coefficients (intercept excluded)
        alpha_ : float      — best alpha selected by cross-validation
    """

    def __init__(
        self: Self,
        fit_intercept: bool = True,
        alphas: np.ndarray | None = None,
        n_splits: int = 5,
        max_iter: int = 200,
        L1_wt: float = 1.0,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.alphas = np.logspace( -4, 2, 10 ) if alphas is None else alphas
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.L1_wt = L1_wt
        self.coef_: np.ndarray | None = None
        self.alpha_: float | None = None
    #/def __init__

    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> 'PoissonLassoCV':
        X_sm: np.ndarray = sm.add_constant( X ) if self.fit_intercept else X
        kf: KFold = KFold( n_splits = self.n_splits, shuffle = True )

        best_alpha: float = self.alphas[0]
        best_score: float = np.inf

        alpha: float
        for alpha in self.alphas:
            fold_scores: list[ float ] = []
            train_idx: np.ndarray
            val_idx: np.ndarray
            for train_idx, val_idx in kf.split( X ):
                result_fold = sm.GLM(
                    y[ train_idx ],
                    X_sm[ train_idx ],
                    family = sm.families.Poisson(),
                ).fit_regularized(
                    method = 'elastic_net',
                    alpha = alpha,
                    L1_wt = self.L1_wt,
                    maxiter = self.max_iter,
                )
                y_pred: np.ndarray = result_fold.predict( X_sm[ val_idx ] )
                y_val: np.ndarray = y[ val_idx ]
                # Poisson deviance on validation fold
                fold_scores.append(
                    2.0 * float(
                        np.sum(
                            y_val * np.log(
                                np.maximum( y_val, 1e-10 ) / np.maximum( y_pred, 1e-10
                                )
                            ) - ( y_val - y_pred )
                        )
                    )
                )
            #/for train_idx, val_idx

            mean_score: float = float(
                np.mean( fold_scores )
            )
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = alpha
            #
        #/for alpha in self.alphas

        # Final fit on all data with best alpha
        result_final = sm.GLM(
            y,
            X_sm,
            family = sm.families.Poisson(),
        ).fit_regularized(
            method = 'elastic_net',
            alpha = best_alpha,
            L1_wt = self.L1_wt,
            maxiter = self.max_iter,
        )

        self.alpha_ = best_alpha
        params: np.ndarray = np.asarray( result_final.params )
        self.coef_ = params[1:] if self.fit_intercept else params

        return self
    #/def fit
#/class PoissonLassoCV
