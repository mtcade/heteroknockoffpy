#
#//  __init__.py
#//  heteroknockoffpy
#//

"""The PRISM simulation data (Zenodo DOI 10.5281/zenodo.22085306).

- :mod:`.dataset` -- :class:`PrismOutcome` / :class:`PrismTrial`, the intuitive interface.
- :mod:`.raw` -- tables in their on-disk shape.
- :mod:`.transforms` -- conversions between stored shapes and heteroknockoffpy calls.
"""

from heteroknockoffpy.datasets.prism import raw, transforms
from heteroknockoffpy.datasets.prism.dataset import PrismOutcome, PrismTrial
from heteroknockoffpy.datasets.prism.raw import ARTIFACTS, OUTCOMES, Artifact, OutcomeType

__all__ = [
    "ARTIFACTS",
    "OUTCOMES",
    "Artifact",
    "OutcomeType",
    "PrismOutcome",
    "PrismTrial",
    "raw",
    "transforms",
]
#
