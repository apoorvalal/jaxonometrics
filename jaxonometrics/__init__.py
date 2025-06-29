"""
jaxonometrics: a Python package for econometric analysis in JAX.
"""

__version__ = "0.0.1"

from .base import BaseEstimator
from .causal import EntropyBalancing, IPW, AIPW # Added IPW, AIPW
from .gmm import GMM, LinearIVGMM, TwoStepGMM
from .linear import LinearRegression
from .mle import Logit, PoissonRegression, MaximumLikelihoodEstimator # Added MLE models

__all__ = [
    "BaseEstimator",
    "EntropyBalancing",
    "IPW", # Added
    "AIPW", # Added
    "GMM",
    "LinearIVGMM",
    "TwoStepGMM",
    "LinearRegression",
    "MaximumLikelihoodEstimator",
    "Logit",
    "PoissonRegression",
]
