from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax.tree_util import tree_leaves
from jaxopt import LevenbergMarquardt

from ..base import BaseEstimator


class TwoStepGMM(BaseEstimator):
    """
    Two-step GMM estimator for models with nuisance parameters.

    This class is designed for situations where the model has two sets of
    parameters: the parameters of interest (theta) and nuisance parameters
    (eta). The estimation is done in two steps: first, the nuisance parameters
    are estimated, and then the parameters of interest are estimated, taking
    the nuisance parameters as given.

    This is useful for models like AIPW, where the regression and propensity
    score models are estimated first (nuisance parameters), and then the
    causal effect is estimated using the influence function.
    """

    def __init__(self, moment_fn: Callable):
        """
        Initialize the TwoStepGMM estimator.

        Args:
            moment_fn: A function that computes the moment conditions.
                It should have the signature `moment_fn(params, data)`, where
                `params` is a tuple `(theta, eta)`.
        """
        super().__init__()
        self.moment_fn = moment_fn

    def fit(
        self,
        data: Dict,
        init_theta: Dict,
        init_eta: Dict,
    ) -> "TwoStepGMM":
        """
        Fit the two-step GMM model.

        Args:
            data: A dictionary containing the data.
            init_theta: Initial values for the parameters of interest.
            init_eta: Initial values for the nuisance parameters.

        Returns:
            The fitted estimator.
        """
        params_init = (init_theta, init_eta)
        solver = LevenbergMarquardt(self.moment_fn)
        sol = solver.run(params_init, data=data)
        self.params = {
            "theta": sol.params[0],
            "eta": sol.params[1],
        }
        return self
