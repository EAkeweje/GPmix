"""
The :mod:`GPmix` package provides implementation of the algorithm introduced in the paper
"Learning Mixtures of Gaussian Processes through Random Projection" to cluster functional data.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package isn’t installed in the environment; fallback:
    __version__ = "0.0.0"

from .smoother import Smoother
from .projector import Projector
from .unigmm import UniGaussianMixtureEnsemble, GaussianMixtureParameterEstimator
# from . import misc


__all__ = [
    "Smoother", "Projector",
    "GaussianMixtureParameterEstimator",
    "UniGaussianMixtureEnsemble"#, "misc"
]