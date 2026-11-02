#
#//  fetch.py
#//  heteroknockoffpy
#//

"""Download the PRISM example data from Zenodo, on demand.

Nothing is downloaded at install time or on import. The data (a ~580 MB
``data.zip``, ~620 MB extracted) is fetched the first time it's asked for
-- by :func:`fetch_prism_data`, by
:func:`heteroknockoffpy.datasets.load_prism` (unless
``download_if_missing=False``), or by the ``heteroknockoffpy-fetch-data``
command -- and cached under :func:`get_data_home`.

Requires the optional extra::

    pip install 'heteroknockoffpy[example-data]'

Cache layout::

    {data_home}/prism-v1/
        data/{linear,continuous,count,categorical}/...   extracted tree
        .complete                                        written after a full extraction
        data.zip                                         only with keep_zip=True
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

# Version DOI (pinned to v1 for reproducibility). The concept DOI
# 10.5281/zenodo.22085305 always resolves to the *latest* version instead.
PRISM_DOI = "10.5281/zenodo.22085306"
PRISM_VERSION = "v1"
PRISM_FILENAME = "data.zip"
PRISM_SHA256 = "1865a0f3bb6fb917298c5a26b13939ed4eeb7eb021b6d6c0ecccadb538000de3"
PRISM_ZIP_BYTES = 578_979_401

DATA_HOME_ENV = "HETEROKNOCKOFFPY_DATA"

_COMPLETE_MARKER = ".complete"


def get_data_home( data_home: str | os.PathLike | None = None ) -> Path:
    """Root directory for cached example data.

    Resolution order: the ``data_home`` argument, then the
    ``HETEROKNOCKOFFPY_DATA`` environment variable, then the platform's
    user cache directory (e.g. ``~/Library/Caches/heteroknockoffpy`` on
    macOS, ``~/.cache/heteroknockoffpy`` on Linux).
    """
    if data_home is not None:
        return Path( data_home ).expanduser()
    #
    if env := os.environ.get( DATA_HOME_ENV ):
        return Path( env ).expanduser()
    #
    return Path( _require_pooch().os_cache( "heteroknockoffpy" ) )
#/def get_data_home


def _release_dir( home: Path ) -> Path:
    return home / f"prism-{PRISM_VERSION}"
#/def _release_dir


def prism_data_dir(
    data_home: str | os.PathLike | None = None,
    *,
    download_if_missing: bool = False,
    progress: bool = True,
    ) -> Path:
    """Directory holding the extracted PRISM data (one subdirectory per outcome).

    :param download_if_missing: if the data isn't cached yet, download it
        (via :func:`fetch_prism_data`) instead of raising
        :class:`FileNotFoundError`.
    """
    home = get_data_home( data_home )
    release = _release_dir( home )
    root = release / "data"
    if ( release / _COMPLETE_MARKER ).exists() and root.is_dir():
        return root
    #
    if download_if_missing:
        return fetch_prism_data( home, progress = progress )
    #
    raise FileNotFoundError(
        f"PRISM example data not found under {release}. Download it with "
        "`heteroknockoffpy-fetch-data` or `heteroknockoffpy.datasets.fetch_prism_data()` "
        "(needs: pip install 'heteroknockoffpy[example-data]'), "
        f"or point {DATA_HOME_ENV} at an existing cache."
    )
#/def prism_data_dir


def is_cached( data_home: str | os.PathLike | None = None ) -> bool:
    """Whether the PRISM data is already downloaded and extracted."""
    try:
        prism_data_dir( data_home )
    except FileNotFoundError:
        return False
    #
    return True
#/def is_cached


def fetch_prism_data(
    data_home: str | os.PathLike | None = None,
    *,
    force: bool = False,
    keep_zip: bool = False,
    progress: bool = True,
    ) -> Path:
    """Download (once), verify, and extract the PRISM data; return its root.

    Safe to call repeatedly: an already-extracted cache is returned without
    touching the network. The zip is checked against a pinned SHA-256
    before extraction, extracted to a temporary directory, and only then
    moved into place, so an interrupted run never leaves a half-extracted
    tree that looks complete.

    :param force: re-download and re-extract even if the cache looks complete.
    :param keep_zip: keep ``data.zip`` next to the extracted tree (by
        default it's deleted after extraction to save ~580 MB).
    :param progress: show a download progress bar (needs ``tqdm``).
    :return: the extracted data root (one subdirectory per outcome).
    """
    pooch = _require_pooch()
    home = get_data_home( data_home )
    release = _release_dir( home )
    root = release / "data"
    marker = release / _COMPLETE_MARKER

    if not force and marker.exists() and root.is_dir():
        return root
    #
    release.mkdir( parents = True, exist_ok = True )
    marker.unlink( missing_ok = True )
    zip_path = release / PRISM_FILENAME
    if force:
        zip_path.unlink( missing_ok = True )
    #

    if progress:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            progress = False
        #
    #
    pooch.retrieve(
        url = f"doi:{PRISM_DOI}/{PRISM_FILENAME}",
        known_hash = f"sha256:{PRISM_SHA256}",
        fname = PRISM_FILENAME,
        path = release,
        progressbar = progress,
    )

    partial = release / ".data.partial"
    shutil.rmtree( partial, ignore_errors = True )
    with zipfile.ZipFile( zip_path ) as zf:
        zf.extractall( partial )
    #
    extracted = partial / "data"
    if not extracted.is_dir():
        raise RuntimeError( f"{zip_path} has no top-level data/ directory" )
    #
    shutil.rmtree( root, ignore_errors = True )
    extracted.rename( root )
    shutil.rmtree( partial, ignore_errors = True )
    marker.write_text( f"{PRISM_DOI}\nsha256:{PRISM_SHA256}\n" )

    if not keep_zip:
        zip_path.unlink( missing_ok = True )
    #
    return root
#/def fetch_prism_data


def _require_pooch():
    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "Downloading example data needs the optional extra: "
            "pip install 'heteroknockoffpy[example-data]'"
        ) from e
    #
    return pooch
#/def _require_pooch


def main( argv: list[str] | None = None ) -> int:
    """``heteroknockoffpy-fetch-data`` command-line entry point."""
    parser = argparse.ArgumentParser(
        prog = "heteroknockoffpy-fetch-data",
        description = (
            f"Download the PRISM example data (Zenodo DOI {PRISM_DOI}, "
            f"~{PRISM_ZIP_BYTES / 1e6:.0f} MB) into the heteroknockoffpy data cache."
        ),
    )
    parser.add_argument(
        "--data-home", type = Path, default = None,
        help = f"cache root (default: ${DATA_HOME_ENV}, else the user cache directory)",
    )
    parser.add_argument( "--force", action = "store_true", help = "re-download even if cached" )
    parser.add_argument( "--keep-zip", action = "store_true", help = "keep data.zip after extracting" )
    parser.add_argument( "--no-progress", action = "store_true", help = "hide the progress bar" )
    parser.add_argument(
        "--path", action = "store_true",
        help = "only print where the data is (or would be) cached; don't download",
    )
    args = parser.parse_args( argv )

    if args.path:
        print( _release_dir( get_data_home( args.data_home ) ) / "data" )
        return 0
    #
    root = fetch_prism_data(
        args.data_home,
        force = args.force,
        keep_zip = args.keep_zip,
        progress = not args.no_progress,
    )
    print( root )
    return 0
#/def main


if __name__ == "__main__":
    raise SystemExit( main() )
#
