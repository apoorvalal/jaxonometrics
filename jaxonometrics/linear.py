from typing import Dict, Optional

import jax.numpy as jnp
import lineax as lx

from .base import BaseEstimator


class LinearRegression(BaseEstimator):
    """
    Linear regression model using lineax for efficient solving.

    This class provides a simple interface for fitting a linear regression
    model, especially useful for high-dimensional problems where n > p.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> "LinearRegression":
        """
        Fit the linear model.

        Args:
            X: The design matrix of shape (n_samples, n_features).
            y: The target vector of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        sol = lx.linear_solve(
            operator=lx.MatrixLinearOperator(X),
            vector=y,
            solver=lx.AutoLinearSolver(well_posed=None),
        )
        self.params = {"beta": sol.value}
        return self
