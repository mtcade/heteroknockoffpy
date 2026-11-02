#
#//  dataset.py
#//  heteroknockoffpy
#//

"""High-level loaders for the PRISM example data.

:class:`PrismOutcome` is one outcome type's collection of trials, with a
filterable :attr:`~PrismOutcome.catalog` (one row per trial, generation
parameters already joined in). :class:`PrismTrial` is everything about a
single trial -- ``X``, ``Xk``, ``y``, stored importances and selections,
ground-truth relevant features, and the keyword arguments needed to
recompute any of it with heteroknockoffpy.

    >>> from heteroknockoffpy import datasets
    >>> linear = datasets.load_prism( "linear" )
    >>> linear.catalog.select( "scenario", "n", "p", "importance_methods" )
    >>> big = linear.filter( n = 4096, importance_method = "LASSO" )
    >>> trial = big[0]
    >>> trial.X, trial.Xk, trial.y
    >>> trial.selected_features(), trial.relevant_features, trial.fdp(), trial.power()
"""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import polars as pl

from heteroknockoffpy.datasets.prism import raw, transforms
from heteroknockoffpy.datasets.prism.raw import OutcomeType

# (par_group, par_name) -> catalog column, and which bundle owns that group.
_CATALOG_PARAMETERS: tuple[tuple[str, str, str, Literal["x", "xk", "trial"]], ...] = (
    ( "y", "idn", "scenario", "trial" ),
    ( "X", "n", "n", "x" ),
    ( "X", "p_numeric", "p_numeric", "x" ),
    ( "X", "p_categorical", "p_categorical", "x" ),
    ( "X", "covariance_type", "covariance_type", "x" ),
    ( "X", "covariance_rho", "covariance_rho", "x" ),
    ( "X", "modes", "modes", "x" ),
    ( "X", "modes_magnitude", "modes_magnitude", "x" ),
    ( "y", "magnitude", "magnitude", "trial" ),
    ( "y", "p_relevant_prob", "p_relevant_prob", "trial" ),
    ( "Xk", "method", "knockoff_method", "xk" ),
    ( "Xk", "categorical_method", "knockoff_categorical_method", "xk" ),
    ( "selections", "fdr", "fdr", "trial" ),
    ( "selections", "offset", "offset", "trial" ),
    ( "y", "fn_str", "fn_str", "trial" ),
)

_OWNER_KEY = { "x": "x_bundle_id", "xk": "xk_bundle_id", "trial": "bundle_id" }


def _parameter_column( params: pl.DataFrame, group: str, name: str, alias: str ) -> pl.DataFrame:
    """``(bundle_id, alias)`` for one scalar parameter, from whichever ``par_val_*`` column holds it."""
    sub = params.filter( ( pl.col( "par_group" ) == group ) & ( pl.col( "par_name" ) == name ) )
    for column in transforms._PARAMETER_VALUE_COLUMNS:
        if column in sub.columns and sub[column].null_count() < sub.height:
            return sub.select( "bundle_id", pl.col( column ).alias( alias ) ).unique( "bundle_id" )
        #
    #
    return sub.select( "bundle_id", pl.lit( None ).alias( alias ) ).unique( "bundle_id" )
#/def _parameter_column


@dataclass( frozen = True, repr = False )
class PrismTrial:
    """One simulated trial: data, stored results, ground truth, and recompute helpers.

    Methods taking ``method`` default to the trial's only importance method,
    and raise if it stores more than one (see :attr:`methods`).
    """

    outcome: OutcomeType
    bundle_id: str
    x_bundle_id: str
    xk_bundle_id: str
    X: pl.DataFrame
    """Design matrix: ``p_numeric`` ``Float32`` columns, then ``p_categorical`` ``Categorical`` columns."""
    Xk: pl.DataFrame
    """Stored knockoff copy of ``X`` (same shape and dtypes)."""
    y: pl.Series
    """Response, row-aligned with ``X``."""
    importances: pl.DataFrame
    """``importance_method, feature_idx, importance, importance_k``."""
    selections: pl.DataFrame
    """``importance_method, feature_idx, fdr_target, selected``."""
    parameters: dict[str, dict[str, Any]] = field( default_factory = dict )
    """Generation parameters by group (``X``, ``Xk``, ``y``, ``importances``, ``selections``)."""
    loss_value: float | None = None

    def __repr__( self ) -> str:
        return (
            f"PrismTrial({self.outcome!r}, {self.bundle_id}, n={self.n}, p={self.p}, "
            f"methods={list( self.methods )}, relevant={self.relevant_features})"
        )
    #/def __repr__

    @property
    def n( self ) -> int:
        return self.X.height
    #/def n

    @property
    def p( self ) -> int:
        return self.X.width
    #/def p

    @property
    def methods( self ) -> tuple[str, ...]:
        """Importance methods stored for this trial (usually one; two for a shared torch fit)."""
        return tuple( self.importances["importance_method"].unique( maintain_order = True ) )
    #/def methods

    @property
    def fn_str( self ) -> str:
        """The response's generating expression over ``col(i)``."""
        return self.parameters["y"]["fn_str"]
    #/def fn_str

    @property
    def relevant_features( self ) -> list[int]:
        """Ground-truth relevant feature indices (every ``col(i)`` in :attr:`fn_str`)."""
        return transforms.relevant_features( self.fn_str )
    #/def relevant_features

    @property
    def offset( self ) -> float:
        return self.parameters["selections"]["offset"]
    #/def offset

    @property
    def knockoff_kwargs( self ) -> dict[str, Any]:
        """Keyword arguments for :func:`heteroknockoffpy.knockoff.get_knockoffs` that built :attr:`Xk`."""
        return dict( self.parameters["Xk"] )
    #/def knockoff_kwargs

    def _method( self, method: str | None ) -> str:
        if method is None:
            if len( self.methods ) != 1:
                raise ValueError( f"trial stores several importance methods {self.methods}; pass method=" )
            #
            return self.methods[0]
        #
        if method not in self.methods:
            raise ValueError( f"trial has no stored {method!r} importances; it has {self.methods}" )
        #
        return method
    #/def _method

    def fdr( self, method: str | None = None ) -> float:
        """Target FDR the stored selections for ``method`` used."""
        method = self._method( method )
        # stored as Float32; round away the float32 -> float64 noise (0.2 -> 0.20000000298...)
        return round( float( self.selections.filter( pl.col( "importance_method" ) == method )["fdr_target"][0] ), 6 )
    #/def fdr

    def importance_kwargs( self, method: str | None = None ) -> dict[str, Any]:
        """Keyword arguments for :meth:`importance_function` ``(method)`` matching the stored run."""
        method = self._method( method )
        return transforms.heteroknockoffpy_kwargs( method, self.parameters["importances"], p = self.p )
    #/def importance_kwargs

    def importance_function( self, method: str | None = None ) -> Callable[..., np.ndarray]:
        """The ``heteroknockoffpy.importance`` function that produced ``method``'s importances."""
        return transforms.importance_function( self._method( method ) )
    #/def importance_function

    def importance_array( self, method: str | None = None ) -> np.ndarray:
        """Stored importances as heteroknockoffpy's flat ``(2*p,)`` array (originals, then knockoffs)."""
        sub = self.importances.filter( pl.col( "importance_method" ) == self._method( method ) )
        return transforms.combine_importance_arrays(
            sub["importance"].to_numpy(), sub["importance_k"].to_numpy()
        )
    #/def importance_array

    def selected_features( self, method: str | None = None, *, fdr: float | None = None ) -> list[int]:
        """Selected feature indices.

        :param fdr: ``None`` returns the stored selections. A value
            re-thresholds the stored importances at that target FDR (with
            the trial's own ``offset``) -- no model refit.
        """
        method = self._method( method )
        if fdr is None:
            sub = self.selections.filter( ( pl.col( "importance_method" ) == method ) & pl.col( "selected" ) )
            return sorted( sub["feature_idx"].to_list() )
        #
        from heteroknockoffpy import importance

        W = importance.wFromImportances( self.importance_array( method ), W_method = "difference" )
        T = importance.selection_threshold( W, fdr = fdr, offset = self.offset )
        return [ int( i ) for i in np.flatnonzero( ( W >= T ) & ( W > 0 ) ) ]
    #/def selected_features

    def fdp( self, method: str | None = None, *, fdr: float | None = None ) -> float:
        """False discovery proportion of :meth:`selected_features` against :attr:`relevant_features`."""
        selected = self.selected_features( method, fdr = fdr )
        if not selected:
            return 0.0
        #
        relevant = set( self.relevant_features )
        return sum( i not in relevant for i in selected ) / len( selected )
    #/def fdp

    def power( self, method: str | None = None, *, fdr: float | None = None ) -> float:
        """Fraction of :attr:`relevant_features` that :meth:`selected_features` recovers."""
        relevant = set( self.relevant_features )
        if not relevant:
            return float( "nan" )
        #
        return len( relevant.intersection( self.selected_features( method, fdr = fdr ) ) ) / len( relevant )
    #/def power

    def compute_knockoffs( self, rng: np.random.Generator, **overrides: Any ) -> pl.DataFrame:
        """Draw a fresh knockoff copy of :attr:`X` with the stored construction settings.

        Fresh randomness, so values differ from :attr:`Xk`; the construction
        is the same. ``overrides`` replace stored keyword arguments.
        """
        from heteroknockoffpy import knockoff

        return knockoff.get_knockoffs( self.X, rng = rng, **{ **self.knockoff_kwargs, **overrides } )
    #/def compute_knockoffs

    def compute_importances(
        self,
        method: str | None = None,
        *,
        rng: np.random.Generator,
        Xk: pl.DataFrame | None = None,
        **overrides: Any,
        ) -> np.ndarray:
        """Recompute ``method``'s importances with the stored hyperparameters.

        :param Xk: knockoffs to score against; defaults to the stored :attr:`Xk`.
        :return: heteroknockoffpy's flat ``(2*p,)`` importance array.
        """
        method = self._method( method )
        kwargs = { **self.importance_kwargs( method ), **overrides }
        fn = transforms.importance_function( method )
        return fn( self.X, self.Xk if Xk is None else Xk, self.y, rng = rng, **kwargs )
    #/def compute_importances
#/class PrismTrial


class _Tables:
    """Per-outcome tables read once and shared by every filtered view."""

    def __init__( self, outcome: OutcomeType, root: Path ) -> None:
        self.outcome = outcome
        self.root = root
    #/def __init__

    @cached_property
    def trials( self ) -> pl.DataFrame:
        return raw.trials( self.outcome, self.root )
    #/def trials

    @cached_property
    def parameters( self ) -> pl.DataFrame:
        return raw.parameters( self.outcome, self.root )
    #/def parameters

    @cached_property
    def results( self ) -> pl.DataFrame:
        return raw.results( self.outcome, self.root )
    #/def results

    @cached_property
    def catalog( self ) -> pl.DataFrame:
        catalog = self.trials
        for group, name, alias, owner in _CATALOG_PARAMETERS:
            column = _parameter_column( self.parameters, group, name, alias )
            catalog = catalog.join(
                column.rename( { "bundle_id": _OWNER_KEY[owner] } ), on = _OWNER_KEY[owner], how = "left"
            )
        #
        methods = (
            raw.scan( self.outcome, "importances", self.root )
            .select( "bundle_id", "importance_method" )
            .unique()
            .group_by( "bundle_id" )
            .agg( pl.col( "importance_method" ).sort().alias( "importance_methods" ) )
            .collect()
        )
        relevant = (
            pl.col( "fn_str" )
            .str.extract_all( r"col\(\d+\)" )
            .list.eval( pl.element().str.extract( r"(\d+)", 1 ).cast( pl.Int64 ) )
            .list.unique()
            .list.sort()
        )
        catalog = (
            catalog
            .join( methods, on = "bundle_id", how = "left" )
            .join( self.results.select( "bundle_id", "loss_value" ), on = "bundle_id", how = "left" )
            .with_columns(
                ( pl.col( "p_numeric" ) + pl.col( "p_categorical" ) ).alias( "p" ),
                relevant.alias( "relevant_features" ),
            )
            .with_columns( pl.col( "relevant_features" ).list.len().alias( "n_relevant" ) )
        )
        leading = [
            "bundle_id", "x_bundle_id", "xk_bundle_id", "importance_methods",
            "scenario", "n", "p", "p_numeric", "p_categorical", "n_relevant",
        ]
        return catalog.select( *leading, pl.exclude( leading ) ).sort( "bundle_id" )
    #/def catalog
#/class _Tables


class PrismOutcome:
    """The trials for one outcome type, optionally filtered.

    Index by position (``ds[0]``) or bundle id (``ds["01a1..."]``) to get a
    :class:`PrismTrial`; iterate to get every trial in :attr:`catalog`
    order. :meth:`filter` returns a narrower view sharing the same cached
    tables.
    """

    def __init__(
        self,
        outcome: OutcomeType,
        root: str | os.PathLike | None = None,
        *,
        _tables: _Tables | None = None,
        _catalog: pl.DataFrame | None = None,
        ) -> None:
        self.outcome: OutcomeType = outcome
        self.path: Path = raw.outcome_dir( outcome, root )
        self._tables = _tables if _tables is not None else _Tables( outcome, self.path.parent )
        self._catalog = _catalog
    #/def __init__

    @property
    def root( self ) -> Path:
        """Directory holding every outcome's subdirectory."""
        return self.path.parent
    #/def root

    @property
    def catalog( self ) -> pl.DataFrame:
        """One row per trial in this view.

        Columns: ``bundle_id``/``x_bundle_id``/``xk_bundle_id`` (ids for
        :mod:`~heteroknockoffpy.datasets.prism.raw`), ``importance_methods``
        (every method stored for the trial -- sometimes two), ``scenario``
        (the simulation sweep the trial belongs to), the ``X`` design
        (``n``, ``p``, ``p_numeric``, ``p_categorical``, ``covariance_type``,
        ``covariance_rho``, ``modes``, ``modes_magnitude``), the response
        (``magnitude``, ``p_relevant_prob``, ``fn_str``, ground-truth
        ``relevant_features``/``n_relevant``), the knockoff construction
        (``knockoff_method``, ``knockoff_categorical_method``), selection
        settings (``fdr``, ``offset``), and the fit's ``loss_value``.
        """
        if self._catalog is None:
            self._catalog = self._tables.catalog
        #
        return self._catalog
    #/def catalog

    @property
    def bundle_ids( self ) -> list[str]:
        return self.catalog["bundle_id"].to_list()
    #/def bundle_ids

    @property
    def importance_methods( self ) -> list[str]:
        """Every importance method present in this view."""
        return sorted( self.catalog["importance_methods"].explode().drop_nulls().unique().to_list() )
    #/def importance_methods

    @property
    def scenarios( self ) -> list[str]:
        """Every simulation sweep (``scenario``) present in this view."""
        return sorted( self.catalog["scenario"].drop_nulls().unique().to_list() )
    #/def scenarios

    def __len__( self ) -> int:
        return self.catalog.height
    #/def __len__

    def __iter__( self ) -> Iterator[PrismTrial]:
        for bundle_id in self.bundle_ids:
            yield self.trial( bundle_id )
        #
    #/def __iter__

    def __getitem__( self, key: int | str ) -> PrismTrial:
        return self.trial( key )
    #/def __getitem__

    def __repr__( self ) -> str:
        return (
            f"PrismOutcome({self.outcome!r}, trials={len( self )}, "
            f"methods={self.importance_methods}, scenarios={self.scenarios})"
        )
    #/def __repr__

    def filter( self, *predicates: pl.Expr, importance_method: str | None = None, **equals: Any ) -> Self:
        """A view of the trials matching every condition.

        :param predicates: polars expressions over :attr:`catalog` columns,
            e.g. ``pl.col("n") >= 2048``.
        :param importance_method: keep trials storing this method (a trial
            can store more than one).
        :param equals: ``column=value`` equality on :attr:`catalog` columns,
            e.g. ``scenario="magnitude_sweep", p_categorical=10``. A list or
            tuple value means "any of".
        """
        catalog = self.catalog
        unknown = set( equals ) - set( catalog.columns )
        if unknown:
            raise KeyError( f"unknown catalog columns {sorted( unknown )}; available: {catalog.columns}" )
        #
        conditions = list( predicates )
        for column, value in equals.items():
            if isinstance( value, ( list, tuple, set ) ):
                conditions.append( pl.col( column ).is_in( list( value ) ) )
            #
            else:
                conditions.append( pl.col( column ) == value )
            #
        #
        if importance_method is not None:
            conditions.append( pl.col( "importance_methods" ).list.contains( importance_method ) )
        #
        if conditions:
            catalog = catalog.filter( *conditions )
        #
        return type( self )( self.outcome, self.root, _tables = self._tables, _catalog = catalog )
    #/def filter

    def trial( self, key: int | str ) -> PrismTrial:
        """Load one trial, by position in :attr:`catalog` or by ``bundle_id``."""
        if isinstance( key, int ):
            row = self.catalog.row( key, named = True )
        #
        else:
            matches = self.catalog.filter( pl.col( "bundle_id" ) == key )
            if matches.is_empty():
                raise KeyError( f"no trial {key!r} in this {self.outcome!r} view" )
            #
            row = matches.row( 0, named = True )
        #
        bundle_id, x_id, xk_id = row["bundle_id"], row["x_bundle_id"], row["xk_bundle_id"]

        params = self._tables.parameters
        merged: dict[str, dict[str, Any]] = {}
        for bid in ( x_id, xk_id, bundle_id ):
            merged.update( transforms.parameters_to_dict( params.filter( pl.col( "bundle_id" ) == bid ) ) )
        #

        def _load( artifact: raw.Artifact, bid: str ) -> pl.DataFrame:
            return raw.load_bundle( self.outcome, artifact, bid, self.root )
        #/def _load

        importances = _load( "importances", bundle_id ).drop( "par_bundle_idn", strict = False )
        selections = _load( "selections", bundle_id ).drop( "par_bundle_idn", strict = False )
        return PrismTrial(
            outcome = self.outcome,
            bundle_id = bundle_id,
            x_bundle_id = x_id,
            xk_bundle_id = xk_id,
            X = _load( "X", x_id ),
            Xk = _load( "Xk", xk_id ),
            y = _load( "y", bundle_id ).to_series(),
            importances = importances.select(
                "importance_method", "feature_idx", "importance", "importance_k"
            ).sort( "importance_method", "feature_idx" ),
            selections = selections.select(
                "importance_method", "feature_idx", "fdr_target", "selected"
            ).sort( "importance_method", "feature_idx" ),
            parameters = merged,
            loss_value = row["loss_value"],
        )
    #/def trial

    def _long( self, artifact: raw.Artifact ) -> pl.DataFrame:
        return (
            raw.scan( self.outcome, artifact, self.root )
            .join( self.catalog.lazy().select( "bundle_id" ), on = "bundle_id", how = "semi" )
            .collect()
            .select( "bundle_id", pl.exclude( "bundle_id" ) )
        )
    #/def _long

    def importances( self ) -> pl.DataFrame:
        """Stored importances for every trial in this view (long format), with ``bundle_id``.

        Join to :attr:`catalog` on ``bundle_id`` for trial-level parameters.
        """
        return self._long( "importances" )
    #/def importances

    def selections( self ) -> pl.DataFrame:
        """Stored selections for every trial in this view (long format), with ``bundle_id``."""
        return self._long( "selections" )
    #/def selections

    def parameters( self ) -> pl.DataFrame:
        """Raw long-format parameters for every bundle of this outcome (see :func:`raw.parameters`)."""
        return self._tables.parameters
    #/def parameters

    def x_modes( self ) -> pl.DataFrame:
        """Raw per-``X``-bundle mixture-mode metadata (see :func:`raw.x_modes`)."""
        return raw.x_modes( self.outcome, self.root )
    #/def x_modes
#/class PrismOutcome
#
