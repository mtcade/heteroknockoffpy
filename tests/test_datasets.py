#
#//  test_datasets.py
#//  heteroknockoffpy
#//

"""heteroknockoffpy.datasets: fetch logic (offline, pooch mocked), pure
transforms, and loaders against real PRISM data when available.

Data-backed tests use $HETEROKNOCKOFFPY_PRISM_ROOT (an extracted data/
directory, e.g. a PRISMdata checkout's data/) or the downloaded cache, and
are skipped otherwise -- the test suite never downloads.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import datasets
from heteroknockoffpy.datasets import fetch
from heteroknockoffpy.datasets.prism import transforms


# --- fetch (offline) ---------------------------------------------------------

class _FakePooch:
    """Stands in for pooch: retrieve() writes a tiny data.zip instead of downloading."""

    def __init__( self, tmp_path: Path, *, top_level: str = "data" ) -> None:
        self.calls: list[dict] = []
        self.top_level = top_level
        self.cache = tmp_path / "os_cache"
    #/def __init__

    def os_cache( self, project: str ) -> Path:
        return self.cache / project
    #/def os_cache

    def retrieve( self, *, url: str, known_hash: str, fname: str, path: Path, progressbar: bool ) -> str:
        self.calls.append( dict( url = url, known_hash = known_hash, fname = fname, path = path ) )
        zip_path = Path( path ) / fname
        with zipfile.ZipFile( zip_path, "w" ) as zf:
            zf.writestr( f"{self.top_level}/linear/index.parquet", b"stub" )
        #
        return str( zip_path )
    #/def retrieve
#/class _FakePooch


@pytest.fixture
def fake_pooch( tmp_path, monkeypatch ) -> _FakePooch:
    fake = _FakePooch( tmp_path )
    monkeypatch.setattr( fetch, "_require_pooch", lambda: fake )
    monkeypatch.delenv( fetch.DATA_HOME_ENV, raising = False )
    return fake
#/def fake_pooch


def test_fetch_downloads_verifies_extracts_and_caches( tmp_path, fake_pooch ):
    root = datasets.fetch_prism_data( tmp_path / "home", progress = False )

    assert root == tmp_path / "home" / "prism-v1" / "data"
    assert ( root / "linear" / "index.parquet" ).read_bytes() == b"stub"
    assert not ( root.parent / "data.zip" ).exists(), "zip removed by default"
    assert ( root.parent / ".complete" ).exists()
    ( call, ) = fake_pooch.calls
    assert call["url"] == f"doi:{fetch.PRISM_DOI}/data.zip"
    assert call["known_hash"] == f"sha256:{fetch.PRISM_SHA256}"

    # cached: no second download
    assert datasets.fetch_prism_data( tmp_path / "home", progress = False ) == root
    assert len( fake_pooch.calls ) == 1
    assert datasets.prism_data_dir( tmp_path / "home" ) == root
    assert datasets.is_cached( tmp_path / "home" )

    # force: downloads again
    datasets.fetch_prism_data( tmp_path / "home", force = True, keep_zip = True, progress = False )
    assert len( fake_pooch.calls ) == 2
    assert ( root.parent / "data.zip" ).exists()
#/def test_fetch_downloads_verifies_extracts_and_caches


def test_incomplete_extraction_is_not_treated_as_cached( tmp_path, fake_pooch ):
    ( tmp_path / "home" / "prism-v1" / "data" / "linear" ).mkdir( parents = True )  # no .complete marker
    assert not datasets.is_cached( tmp_path / "home" )
    datasets.fetch_prism_data( tmp_path / "home", progress = False )
    assert len( fake_pooch.calls ) == 1
#/def test_incomplete_extraction_is_not_treated_as_cached


def test_zip_without_data_dir_fails_and_stays_uncached( tmp_path, monkeypatch ):
    fake = _FakePooch( tmp_path, top_level = "something_else" )
    monkeypatch.setattr( fetch, "_require_pooch", lambda: fake )
    with pytest.raises( RuntimeError, match = "no top-level data/" ):
        datasets.fetch_prism_data( tmp_path / "home", progress = False )
    #
    assert not datasets.is_cached( tmp_path / "home" )
#/def test_zip_without_data_dir_fails_and_stays_uncached


def test_missing_data_raises_with_instructions( tmp_path, fake_pooch ):
    with pytest.raises( FileNotFoundError, match = "heteroknockoffpy-fetch-data" ):
        datasets.prism_data_dir( tmp_path / "empty" )
    #
    with pytest.raises( FileNotFoundError ):
        datasets.load_prism( "linear", data_home = tmp_path / "empty", download_if_missing = False )
    #
    assert fake_pooch.calls == []
#/def test_missing_data_raises_with_instructions


def test_data_home_resolution( tmp_path, fake_pooch, monkeypatch ):
    assert datasets.get_data_home( tmp_path / "arg" ) == tmp_path / "arg"
    assert datasets.get_data_home() == fake_pooch.cache / "heteroknockoffpy"
    monkeypatch.setenv( fetch.DATA_HOME_ENV, str( tmp_path / "env" ) )
    assert datasets.get_data_home() == tmp_path / "env"
    assert datasets.get_data_home( tmp_path / "arg" ) == tmp_path / "arg"
#/def test_data_home_resolution


def test_cli( tmp_path, fake_pooch, capsys ):
    assert fetch.main( [ "--data-home", str( tmp_path ), "--path" ] ) == 0
    assert capsys.readouterr().out.strip() == str( tmp_path / "prism-v1" / "data" )
    assert fake_pooch.calls == []

    assert fetch.main( [ "--data-home", str( tmp_path ), "--no-progress" ] ) == 0
    assert capsys.readouterr().out.strip() == str( tmp_path / "prism-v1" / "data" )
    assert len( fake_pooch.calls ) == 1
#/def test_cli


def test_missing_pooch_points_at_extra( monkeypatch ):
    import builtins

    real_import = builtins.__import__

    def no_pooch( name, *args, **kwargs ):
        if name == "pooch":
            raise ImportError( "no pooch" )
        #
        return real_import( name, *args, **kwargs )
    #/def no_pooch

    monkeypatch.setattr( builtins, "__import__", no_pooch )
    with pytest.raises( ImportError, match = r"heteroknockoffpy\[example-data\]" ):
        fetch._require_pooch()
    #
#/def test_missing_pooch_points_at_extra


# --- transforms (pure) -------------------------------------------------------

def test_relevant_features_parses_fn_str():
    fn = '((0.99*col(2)*col(7)) + (1.00*col(13)*catmap(col(23), "0":-1.34, "1":0.45)) + col(2))'
    assert transforms.relevant_features( fn ) == [ 2, 7, 13, 23 ]
#/def test_relevant_features_parses_fn_str


def test_kwargs_build_layers_for_torch_methods():
    raw = {
        "method": "torch_prism_grip2", "standardize_y": True, "local_grad_method": "x",
        "first_layer_ratio": 2.0, "layer_count": 3, "layer_ratio": 0.5, "epochs": 10,
    }
    for method in ( "torch_grip2", "torch_prism" ):
        assert transforms.heteroknockoffpy_kwargs( method, raw, p = 30 ) == { "epochs": 10, "layers": [ 60, 30, 15 ] }
    #
#/def test_kwargs_build_layers_for_torch_methods


def test_kwargs_nest_xgb_model_kwargs():
    raw = { "method": "xgb_prism", "standardize_y": False, "max_depth": 4, "learning_rate": 0.1, "other": 1 }
    assert transforms.heteroknockoffpy_kwargs( "xgb_prism", raw, p = 5 ) == {
        "other": 1, "model_kwargs": { "max_depth": 4, "learning_rate": 0.1 },
    }
#/def test_kwargs_nest_xgb_model_kwargs


def test_importance_array_roundtrip_and_dispatch():
    imp, imp_k = np.arange( 3.0 ), np.arange( 3.0, 6.0 )
    a, b = transforms.split_importance_array( transforms.combine_importance_arrays( imp, imp_k ) )
    np.testing.assert_array_equal( a, imp )
    np.testing.assert_array_equal( b, imp_k )
    assert transforms.importance_function( "LASSO" ).__name__ == "lassoImportances"
    with pytest.raises( ValueError, match = "unknown importance_method" ):
        transforms.importance_function( "nope" )
    #
#/def test_importance_array_roundtrip_and_dispatch


# --- loaders (real data, skipped when absent) ---------------------------------

def _prism_root() -> Path | None:
    if env := os.environ.get( "HETEROKNOCKOFFPY_PRISM_ROOT" ):
        return Path( env )
    #
    try:
        return datasets.prism_data_dir()
    except ( FileNotFoundError, ImportError ):
        return None
    #
#/def _prism_root


@pytest.fixture( scope = "module" )
def prism_root() -> Path:
    root = _prism_root()
    if root is None:
        pytest.skip( "PRISM data not available (set HETEROKNOCKOFFPY_PRISM_ROOT or run heteroknockoffpy-fetch-data)" )
    #
    return root
#/def prism_root


@pytest.mark.parametrize( "outcome", datasets.OUTCOMES )
def test_catalog_is_complete( prism_root, outcome ):
    ds = datasets.load_prism( outcome, root = prism_root )
    catalog = ds.catalog
    assert len( ds ) == catalog.height == catalog["bundle_id"].n_unique() > 0
    assert catalog.null_count().sum_horizontal().item() == 0
    assert ( catalog["p"] == catalog["p_numeric"] + catalog["p_categorical"] ).all()
    assert ( catalog["n_relevant"] > 0 ).all()
#/def test_catalog_is_complete


def test_filter_and_index( prism_root ):
    ds = datasets.load_prism( "count", root = prism_root )
    view = ds.filter( importance_method = "torch_prism", scenario = [ "n_sweep", "rho_sweep" ] )
    assert 0 < len( view ) < len( ds )
    assert view.catalog["importance_methods"].list.contains( "torch_prism" ).all()
    assert set( view.scenarios ) <= { "n_sweep", "rho_sweep" }
    assert ds.filter( pl.col( "n" ) < 0 ).catalog.is_empty()

    trial = view[0]
    assert view[trial.bundle_id].bundle_id == trial.bundle_id
    with pytest.raises( KeyError ):
        ds.filter( not_a_column = 1 )
    #
    with pytest.raises( KeyError ):
        view["not-a-bundle"]
    #
#/def test_filter_and_index


def test_trial_contents( prism_root ):
    ds = datasets.load_prism( "categorical", root = prism_root )
    row = ds.catalog.row( 0, named = True )
    trial = ds[0]
    assert ( trial.n, trial.p ) == ( row["n"], row["p"] )
    assert trial.Xk.schema == trial.X.schema
    assert trial.y.len() == trial.n
    assert trial.relevant_features == row["relevant_features"]
    assert sorted( trial.methods ) == row["importance_methods"]
    assert trial.knockoff_kwargs == { "method": row["knockoff_method"], "categorical_method": row["knockoff_categorical_method"] }
    assert trial.importance_array( trial.methods[0] ).shape == ( 2 * trial.p, )
#/def test_trial_contents


def test_multi_method_trial_requires_method( prism_root ):
    ds = datasets.load_prism( "continuous", root = prism_root )
    trial = ds.filter( pl.col( "importance_methods" ).list.len() == 2 )[0]
    with pytest.raises( ValueError, match = "pass method=" ):
        trial.selected_features()
    #
    for method in trial.methods:
        assert "layers" in trial.importance_kwargs( method )
    #
#/def test_multi_method_trial_requires_method


@pytest.mark.parametrize( "outcome", datasets.OUTCOMES )
def test_rethresholding_reproduces_stored_selections( prism_root, outcome ):
    ds = datasets.load_prism( outcome, root = prism_root )
    for i in range( 0, len( ds ), max( 1, len( ds ) // 15 ) ):
        trial = ds[i]
        for method in trial.methods:
            stored = trial.selected_features( method )
            assert trial.selected_features( method, fdr = trial.fdr( method ) ) == stored
            relevant = set( trial.relevant_features )
            assert trial.power( method ) == len( relevant & set( stored ) ) / len( relevant )
        #
    #
#/def test_rethresholding_reproduces_stored_selections


def test_long_tables_cover_the_view( prism_root ):
    view = datasets.load_prism( "linear", root = prism_root ).filter( scenario = "n_sweep" )
    imp = view.importances()
    assert set( imp["bundle_id"].unique() ) == set( view.bundle_ids )
    assert { "importance_method", "feature_idx", "importance", "importance_k" } <= set( imp.columns )
#/def test_long_tables_cover_the_view
#
