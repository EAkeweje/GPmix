"""
The :mod:`GPmix` package provides implementation of the algorithm introduced in the paper
"Learning Mixtures of Gaussian Processes through Random Projection" to cluster functional data.
"""

from .smoother import Smoother
from .projector import Projector
from .unigmm import GaussianMixtureParameterEstimator, UniGaussianMixtureEnsemble

__all__ = [
    "Smoother", "Projector",
    "GaussianMixtureParameterEstimator",
    "UniGaussianMixtureEnsemble"
]