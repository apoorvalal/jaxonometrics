"""
jaxonometrics: a Python package for econometric analysis in JAX.
"""

__version__ = "0.0.1"

from .base import BaseEstimator
from .causal import EntropyBalancing
from .gmm import GMM, LinearIVGMM, TwoStepGMM
from .linear import LinearRegression

__all__ = [
    "BaseEstimator",
    "EntropyBalancing",
    "GMM",
    "LinearIVGMM",
    "TwoStepGMM",
    "LinearRegression",
]
