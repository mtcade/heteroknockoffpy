#
#//  __init__.py
#//  heteroknockoffpy
#//

"""Optional example datasets, downloaded on demand (never at install time).

Install the downloader with ``pip install 'heteroknockoffpy[example-data]'``,
then either pre-fetch from the shell::

    heteroknockoffpy-fetch-data            # or: python -m heteroknockoffpy.datasets

or just load -- the first call downloads (~580 MB) and caches::

    from heteroknockoffpy import datasets

    linear = datasets.load_prism( "linear" )
    trial = linear.filter( importance_method = "LASSO" )[0]

The cache lives under :func:`get_data_home` (override with ``data_home=``
or ``$HETEROKNOCKOFFPY_DATA``). An already-extracted copy elsewhere (e.g. a
PRISMdata checkout's ``data/``) can be used directly with ``root=``.
"""

import os

from heteroknockoffpy.datasets import fetch, prism
from heteroknockoffpy.datasets.fetch import (
    DATA_HOME_ENV,
    PRISM_DOI,
    fetch_prism_data,
    get_data_home,
    is_cached,
    prism_data_dir,
)
from heteroknockoffpy.datasets.prism import OUTCOMES, OutcomeType, PrismOutcome, PrismTrial


def load_prism(
    outcome: OutcomeType,
    *,
    root: str | os.PathLike | None = None,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    ) -> PrismOutcome:
    """The PRISM simulation trials for one outcome type.

    :param outcome: ``"linear"``, ``"continuous"``, ``"count"``, or ``"categorical"``.
    :param root: an existing extracted data directory (holding the outcome
        subdirectories); skips the cache and never downloads.
    :param data_home: cache root (see :func:`get_data_home`).
    :param download_if_missing: download into the cache if it's empty;
        ``False`` raises :class:`FileNotFoundError` instead.
    """
    if root is None:
        root = prism_data_dir( data_home, download_if_missing = download_if_missing )
    #
    return PrismOutcome( outcome, root )
#/def load_prism


__all__ = [
    "DATA_HOME_ENV",
    "OUTCOMES",
    "PRISM_DOI",
    "OutcomeType",
    "PrismOutcome",
    "PrismTrial",
    "fetch",
    "fetch_prism_data",
    "get_data_home",
    "is_cached",
    "load_prism",
    "prism",
    "prism_data_dir",
]
#
