"""
Properties of the PRISM consistency changes: categorical predictors scored by
the frequency-weighted SD (2026-10-22; was the prediction range / Omega-diameter
of 2026-10-20) over valid level encodings, with level frequencies pooled over a
predictor and its knockoff, Mahalanobis *norm* aggregation, and the per-SD
central difference in xgb PRISM.
"""
import numpy as np
import polars as pl
import pytest
import torch

from heteroknockoffpy.heteroknockofftorch.prismImportances import (
    _localGrad_forCategories_t,
    _localGrad_forNumeric_t,
    _pooled_level_freqs,
    _ridge_inv_cov_t,
)


class _LinearModel:
    """predict_t(X) = X @ B, (n, out) -- enough of the model interface."""
    def __init__( self, B: torch.Tensor ) -> None:
        self.B = B
    def predict_t( self, X_t: torch.Tensor ) -> torch.Tensor:
        return X_t @ self.B


def _level_preds( model, X_t, j, ohe_vals ):
    """Predictions at the reference level and each one-hot level of block j."""
    preds = []
    for h in [ None ] + list( range( len( j ) ) ):
        _X = X_t.clone()
        for k in j:
            _X[:, k] = ohe_vals[ k ][ 0 ]
        if h is not None:
            _X[:, j[ h ] ] = ohe_vals[ j[ h ] ][ 1 ]
        preds.append( model.predict_t( _X ) )
    return torch.stack( preds, dim=-1 )  # (n, out, K)


def _weighted_sd( lv, pi ):
    """Frequency-weighted SD across the last axis of (n, K) level predictions."""
    dev = lv - ( lv * pi ).sum( dim=-1, keepdim=True )
    return ( ( dev ** 2 ) * pi ).sum( dim=-1 ).sqrt()


def test_categorical_predictor_scalar_response_is_level_range():
    torch.manual_seed( 0 )
    X_t = torch.randn( 50, 4 )
    j = [ 1, 2, 3 ]
    ohe_vals = { k: ( -0.6, 1.7 ) for k in j }
    model = _LinearModel( torch.tensor( [ [ 0.5 ], [ 2.0 ], [ -1.0 ], [ 0.3 ] ] ) )

    got = _localGrad_forCategories_t( j=j, X_t=X_t, model=model, drop_first=True, ohe_vals=ohe_vals )
    lv = _level_preds( model, X_t, j, ohe_vals )[:, 0, :]
    # 2026-10-22: the score is now the frequency-weighted SD (equal weights when
    # level_freqs is omitted), not the range.
    if False:
        expected = lv.amax( dim=1 ) - lv.amin( dim=1 )
    #
    expected = _weighted_sd( lv, torch.full( ( 4, ), 0.25 ) )
    assert torch.allclose( got, expected, atol=1e-6 )


def test_categorical_predictor_scalar_response_is_weighted_sd():
    torch.manual_seed( 4 )
    X_t = torch.randn( 50, 4 )
    j = [ 1, 2, 3 ]
    ohe_vals = { k: ( -0.6, 1.7 ) for k in j }
    model = _LinearModel( torch.tensor( [ [ 0.5 ], [ 2.0 ], [ -1.0 ], [ 0.3 ] ] ) )
    pi = torch.tensor( [ 0.4, 0.1, 0.3, 0.2 ] )

    got = _localGrad_forCategories_t(
        j=j, X_t=X_t, model=model, drop_first=True, ohe_vals=ohe_vals, level_freqs=pi,
    )
    lv = _level_preds( model, X_t, j, ohe_vals )[:, 0, :]
    assert torch.allclose( got, _weighted_sd( lv, pi ), atol=1e-6 )


def test_binary_predictor_is_standardized_coefficient():
    """A binary predictor in raw 0/1 coding scores |beta| sqrt(pi(1-pi)), the
    coefficient of its standardized indicator."""
    X_t = torch.randn( 30, 2 )
    beta = -1.7
    model = _LinearModel( torch.tensor( [ [ 0.4 ], [ beta ] ] ) )
    p1 = 0.2
    got = _localGrad_forCategories_t(
        j=[ 1 ], X_t=X_t, model=model, drop_first=True,
        ohe_vals={ 1: ( 0.0, 1.0 ) }, level_freqs=torch.tensor( [ 1.0 - p1, p1 ] ),
    )
    expected = abs( beta ) * ( p1 * ( 1.0 - p1 ) ) ** 0.5
    assert torch.allclose( got, torch.full_like( got, expected ), atol=1e-6 )


def test_pooled_level_freqs_are_swap_invariant():
    """Original and knockoff blocks get the same pooled weights, in level order
    [reference, dummy_1, ...]."""
    rng = np.random.default_rng( 5 )
    n = 200
    def dummies( probs ):
        lev = rng.choice( 3, size=n, p=probs )
        return np.stack( [ lev == 1, lev == 2 ], axis=1 ).astype( float )
    num = rng.normal( size=( n, 1 ) )
    X_all_np = np.concatenate( ( num, dummies( [ 0.5, 0.3, 0.2 ] ), num, dummies( [ 0.6, 0.3, 0.1 ] ) ), axis=1 )
    oheDict = { 'a': 0, 'c': ( 1, 2 ), 'a~': 3, 'c~': ( 4, 5 ) }

    freqs = _pooled_level_freqs( X_all_np, oheDict, drop_first=True )
    assert set( freqs ) == { ( 1, 2 ), ( 4, 5 ) }
    assert torch.allclose( freqs[ ( 1, 2 ) ], freqs[ ( 4, 5 ) ] )
    d = ( X_all_np[:, 1:3].mean( axis=0 ) + X_all_np[:, 4:6].mean( axis=0 ) ) / 2.0
    expected = torch.tensor( [ 1.0 - d.sum(), d[ 0 ], d[ 1 ] ], dtype=torch.float32 )
    assert torch.allclose( freqs[ ( 1, 2 ) ], expected, atol=1e-6 )


def test_categorical_response_diameter_invariant_to_reference_class():
    torch.manual_seed( 1 )
    X_t = torch.randn( 80, 5 )
    j = [ 2, 3, 4 ]
    ohe_vals = { k: ( 0.0, 1.0 ) for k in j }
    B = torch.randn( 5, 3 )

    def score( B_ ):
        model = _LinearModel( B_ )
        logits = model.predict_t( X_t )
        inv_cov = _ridge_inv_cov_t( torch.cov( ( logits[:, 1:] - logits[:, 0:1] ).T ) )
        return _localGrad_forCategories_t(
            j=j, X_t=X_t, model=model, drop_first=True, inv_cov_t=inv_cov, ohe_vals=ohe_vals,
        )

    base = score( B )
    # Make class 2 the reference instead of class 0.
    permuted = score( B[:, [ 2, 1, 0 ] ] )
    assert torch.allclose( base, permuted, rtol=1e-4, atol=1e-5 )


def test_binary_response_score_is_scalar_score_over_sd():
    torch.manual_seed( 2 )
    X_t = torch.randn( 60, 4 )
    j = [ 1, 2, 3 ]
    ohe_vals = { k: ( 0.0, 1.0 ) for k in j }
    model = _LinearModel( torch.randn( 4, 2 ) )
    logits = model.predict_t( X_t )
    contrast = logits[:, 1:] - logits[:, 0:1]
    var = torch.cov( contrast.T )
    inv_cov = _ridge_inv_cov_t( var )

    got = _localGrad_forCategories_t(
        j=j, X_t=X_t, model=model, drop_first=True, inv_cov_t=inv_cov, ohe_vals=ohe_vals,
    )
    lv = _level_preds( model, X_t, j, ohe_vals )
    lv = lv[:, 1, :] - lv[:, 0, :]
    # 2026-10-22: weighted-SD analog of the old range/sd reduction.
    if False:
        expected = ( lv.amax( dim=1 ) - lv.amin( dim=1 ) ) * inv_cov.reshape( () ).sqrt()
    #
    expected = _weighted_sd( lv, torch.full( ( 4, ), 0.25 ) ) * inv_cov.reshape( () ).sqrt()
    assert torch.allclose( got, expected, rtol=1e-4, atol=1e-6 )


def test_numeric_categorical_response_returns_mahalanobis_norm():
    torch.manual_seed( 3 )
    X_t = torch.randn( 40, 3 )
    model = _LinearModel( torch.randn( 3, 3 ) )
    inv_cov = torch.eye( 2 )
    got = _localGrad_forNumeric_t( j=0, X_t=X_t, model=model, bandwidth=1.0, inv_cov_t=inv_cov )
    d = ( model.B[ 0, 1: ] - model.B[ 0, 0 ] ).expand( 40, 2 )
    assert torch.allclose( got, d.norm( dim=1 ), atol=1e-5 )


def test_xgb_prism_numeric_importance_is_per_sd():
    """Scaling a numeric column leaves its xgb PRISM score unchanged: the step
    is h*sd and the difference is divided by 2h, a per-SD derivative."""
    from heteroknockoffpy.importance import xgbPrismImportances as prism_importances

    rng = np.random.default_rng( 0 )
    n = 400
    X = pl.DataFrame( { '0': rng.normal( size=n ), '1': rng.normal( size=n ) } )
    Xk = pl.DataFrame( { '0': rng.normal( size=n ), '1': rng.normal( size=n ) } )
    y = pl.Series( 'y', 2.0 * X[ '0' ].to_numpy() + rng.normal( scale=0.1, size=n ) )
    kw = dict( model_kwargs={ 'n_estimators': 50, 'max_depth': 3, 'random_state': 0 } )

    base = prism_importances( X, Xk, y, **kw )
    scale = 10.0
    X_s  = X.with_columns( pl.col( '0' ) * scale )
    Xk_s = Xk.with_columns( pl.col( '0' ) * scale )
    scaled = prism_importances( X_s, Xk_s, y, **kw )
    assert scaled[ 0 ] == pytest.approx( base[ 0 ], rel=1e-3 )


def test_lasso_weighted_sd_is_reference_invariant():
    """The paper's example: effects (0, 1, 2) with the first level as reference and
    the equivalent (-1, 0, 1) with the second both give sqrt(2/3) at equal
    frequencies; the Omega version is likewise shift-invariant."""
    from heteroknockoffpy.importance import _collapse_cat_weighted_sd
    pi = np.full( 3, 1.0 / 3.0 )
    a = _collapse_cat_weighted_sd( np.array( [ 0.0, 1.0, 2.0 ] ), pi )
    b = _collapse_cat_weighted_sd( np.array( [ -1.0, 0.0, 1.0 ] ), pi )
    assert a == pytest.approx( np.sqrt( 2.0 / 3.0 ) ) and b == pytest.approx( a )
    V = np.array( [ [ 0.0, 0.5, -1.2 ], [ 0.0, 2.0, 0.3 ] ] )
    VI = np.array( [ [ 2.0, 0.3 ], [ 0.3, 1.0 ] ] )
    pi = np.array( [ 0.5, 0.3, 0.2 ] )
    assert _collapse_cat_weighted_sd( V - V[ :, [ 1 ] ], pi, VI ) == pytest.approx( _collapse_cat_weighted_sd( V, pi, VI ) )


def test_lasso_pooled_ohe_level_freqs():
    from heteroknockoffpy.importance import _pooled_ohe_level_freqs
    X_ohe = np.array( [ [ 0, 1, 0, 0 ], [ 1, 0, 0, 1 ], [ 0, 0, 1, 0 ], [ 0, 0, 0, 0 ] ], dtype=float )
    oheDict = { 'c': ( 0, 1 ), 'c~': ( 2, 3 ) }
    f = _pooled_ohe_level_freqs( X_ohe, oheDict, 'c' )
    assert np.allclose( f, _pooled_ohe_level_freqs( X_ohe, oheDict, 'c~' ) )
    assert np.allclose( f, [ 1.0 - 0.25 - 0.25, 0.25, 0.25 ] )
