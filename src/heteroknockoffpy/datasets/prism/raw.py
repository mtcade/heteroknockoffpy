#
#//  raw.py
#//  heteroknockoffpy
#//

"""Raw, on-disk-shaped access to the PRISM simulation data.

Per outcome type (see :data:`OutcomeType`), ``{root}/{outcome}/`` holds:

- ``{X,Xk,y,importances,selections}/{bundle_id}.parquet`` -- one file per
  bundle, named by that bundle's ``par_bundle_idn`` (as lowercase hex).
- ``index.parquet`` -- the join table. One row per ``X``/``Xk``/``trial``
  bundle; a ``trial`` row names the ``X``/``Xk`` bundle it was generated
  from. ``X``/``Xk`` bundles are *reused* across many trials, so never join
  artifacts by filename -- use :func:`trials`.
- ``parameters_table.parquet`` -- per-bundle generation parameters (long format).
- ``results.parquet`` -- per-bundle ``loss_value``.
- ``X_modes.parquet`` -- per-``X``-bundle mixture-mode metadata.

Every function takes an optional ``root`` (the directory holding the
outcome subdirectories). ``None`` means the downloaded cache
(:func:`heteroknockoffpy.datasets.prism_data_dir`); this module never
downloads by itself. Most users want
:func:`heteroknockoffpy.datasets.load_prism` instead.
"""

import os
from pathlib import Path
from typing import Literal

import polars as pl

OutcomeType = Literal["linear", "continuous", "count", "categorical"]
Artifact = Literal["X", "Xk", "y", "importances", "selections"]
ParStage = Literal["X", "Xk", "trial"]

OUTCOMES: tuple[OutcomeType, ...] = ( "linear", "continuous", "count", "categorical" )
ARTIFACTS: tuple[Artifact, ...] = ( "X", "Xk", "y", "importances", "selections" )


def resolve_root( root: str | os.PathLike | None = None ) -> Path:
    """``root`` as a :class:`Path`, defaulting to the downloaded cache."""
    if root is not None:
        return Path( root ).expanduser()
    #
    from heteroknockoffpy.datasets.fetch import prism_data_dir

    return prism_data_dir()
#/def resolve_root


def outcome_dir( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> Path:
    """``{root}/{outcome}``, validated."""
    if outcome not in OUTCOMES:
        raise ValueError( f"outcome must be one of {OUTCOMES}, got {outcome!r}" )
    #
    path = resolve_root( root ) / outcome
    if not path.is_dir():
        raise FileNotFoundError( f"no PRISM data for outcome {outcome!r} at {path}" )
    #
    return path
#/def outcome_dir


def bundle_path(
    outcome: OutcomeType, artifact: Artifact, bundle_id: str, root: str | os.PathLike | None = None
    ) -> Path:
    """Path to a single bundle's parquet file."""
    return outcome_dir( outcome, root ) / artifact / f"{bundle_id}.parquet"
#/def bundle_path


def list_bundle_ids(
    outcome: OutcomeType, artifact: Artifact, root: str | os.PathLike | None = None
    ) -> list[str]:
    """Bundle ids (parquet filename stems) available for ``outcome``/``artifact``."""
    return sorted( p.stem for p in ( outcome_dir( outcome, root ) / artifact ).glob( "*.parquet" ) )
#/def list_bundle_ids


def load_bundle(
    outcome: OutcomeType, artifact: Artifact, bundle_id: str, root: str | os.PathLike | None = None
    ) -> pl.DataFrame:
    """Load a single bundle's parquet file."""
    return pl.read_parquet( bundle_path( outcome, artifact, bundle_id, root ) )
#/def load_bundle


def scan(
    outcome: OutcomeType, artifact: Artifact, root: str | os.PathLike | None = None
    ) -> pl.LazyFrame:
    """Lazily scan every bundle for ``outcome``/``artifact``, tagged with ``bundle_id``.

    Not meaningful for ``X``/``Xk``, whose column count and dtypes differ
    bundle to bundle -- load those one at a time with :func:`load_bundle`.
    """
    lf = pl.scan_parquet(
        outcome_dir( outcome, root ) / artifact / "*.parquet", include_file_paths = "_path"
    )
    return lf.with_columns(
        pl.col( "_path" ).str.extract( r"([^/\\]+)\.parquet$", 1 ).alias( "bundle_id" )
    ).drop( "_path", "par_bundle_idn", strict = False )
#/def scan


def load( outcome: OutcomeType, artifact: Artifact, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """Eagerly load and concatenate every bundle for ``outcome``/``artifact``."""
    return scan( outcome, artifact, root ).collect()
#/def load


def _with_bundle_id( df: pl.DataFrame, *, column: str = "par_bundle_idn" ) -> pl.DataFrame:
    """Replace a binary ``par_bundle_idn`` column with its hex ``bundle_id`` string."""
    return df.with_columns( pl.col( column ).bin.encode( "hex" ).alias( "bundle_id" ) ).drop( column )
#/def _with_bundle_id


def _path_to_bundle_id( col: str ) -> pl.Expr:
    return pl.col( col ).str.extract( r"([^/\\]+)\.parquet$", 1 )
#/def _path_to_bundle_id


def index( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """The full bundle index for ``outcome`` (``X``, ``Xk``, and ``trial`` rows).

    ``bundle_id`` is this row's own id; for ``trial`` rows,
    ``x_bundle_id``/``xk_bundle_id`` name the ``X``/``Xk`` bundle used to
    generate it.
    """
    df = _with_bundle_id( pl.read_parquet( outcome_dir( outcome, root ) / "index.parquet" ) )
    return df.with_columns(
        _path_to_bundle_id( "X_path" ).alias( "x_bundle_id" ),
        _path_to_bundle_id( "Xk_path" ).alias( "xk_bundle_id" ),
    ).select( "bundle_id", "par_stage", "x_bundle_id", "xk_bundle_id" )
#/def index


def trials( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """``trial`` rows of :func:`index`: one per ``y``/``importances``/``selections`` bundle."""
    return index( outcome, root ).filter( pl.col( "par_stage" ) == "trial" ).drop( "par_stage" )
#/def trials


def parameters( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """Long-format generation parameters for every bundle (``X``, ``Xk``, and trial)."""
    return _with_bundle_id( pl.read_parquet( outcome_dir( outcome, root ) / "parameters_table.parquet" ) )
#/def parameters


def results( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """Per-bundle ``loss_value``."""
    return _with_bundle_id( pl.read_parquet( outcome_dir( outcome, root ) / "results.parquet" ) )
#/def results


def x_modes( outcome: OutcomeType, root: str | os.PathLike | None = None ) -> pl.DataFrame:
    """Per-``X``-bundle mixture-mode metadata."""
    return _with_bundle_id( pl.read_parquet( outcome_dir( outcome, root ) / "X_modes.parquet" ) )
#/def x_modes
#
