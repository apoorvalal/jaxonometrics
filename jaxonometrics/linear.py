from typing import Dict, Optional

import numpy as np
import jax # Ensure jax is imported
import jax.numpy as jnp
import lineax as lx

from .base import BaseEstimator


# Helper function for JIT compilation of vcov calculations
@jax.jit
def _calculate_vcov_details(
    coef: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, se_type: str, n: int, k: int
):
    """Helper function to compute SEs, designed to be JIT compiled."""
    ε = y - X @ coef
    if se_type == "HC1":
        M = jnp.einsum("ij,i,ik->jk", X, ε**2, X)
        XtX_inv = jnp.linalg.inv(X.T @ X)
        Σ = XtX_inv @ M @ XtX_inv
        return jnp.sqrt((n / (n - k)) * jnp.diag(Σ))
    elif se_type == "classical":
        XtX_inv = jnp.linalg.inv(X.T @ X)
        return jnp.sqrt(jnp.diag(XtX_inv) * jnp.var(ε, ddof=k))
    return None # Should not be reached if se_type is valid


class LinearRegression(BaseEstimator):
    """
    Linear regression model using lineax for efficient solving.

    This class provides a simple interface for fitting a linear regression
    model, especially useful for high-dimensional problems where p > n.
    """

    def __init__(self, solver="lineax"):
        """Initialize the LinearRegression model.

        Args:
            solver (str, optional): Solver. Defaults to "lineax", can also be "jax" or "numpy".
        """
        super().__init__()
        self.solver: str = solver

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        se: str = None,
    ) -> "LinearRegression":
        """
        Fit the linear model.

        Args:
            X: The design matrix of shape (n_samples, n_features).
            y: The target vector of shape (n_samples,).
            se: Whether to compute standard errors. "HC1" for robust standard errors, "classical" for classical SEs.

        Returns:
            The fitted estimator.
        """

        if self.solver == "lineax":
            sol = lx.linear_solve(
                operator=lx.MatrixLinearOperator(X),
                vector=y,
                solver=lx.AutoLinearSolver(well_posed=None),
            )
            self.params = {"coef": sol.value}

        elif self.solver == "jax":
            sol = jnp.linalg.lstsq(X, y)
            self.params = {"coef": sol[0]}
        elif self.solver == "numpy":  # for completeness
            X_np, y_np = np.array(X), np.array(y) # Convert to numpy arrays for numpy solver
            sol = np.linalg.lstsq(X_np, y_np, rcond=None)
            self.params = {"coef": jnp.array(sol[0])} # Convert back to jax array

        if se:
            self._vcov(
                y=y,
                X=X,
                se_type=se, # Renamed to avoid conflict with self.se if it existed
            )
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)
        return jnp.dot(X, self.params["coef"])

    def _vcov(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray,
        se_type: str = "HC1", # Renamed from 'se'
    ) -> None:
        n, k = X.shape
        if self.params and "coef" in self.params:
            coef = self.params["coef"]
            se_values = _calculate_vcov_details(coef, X, y, se_type, n, k)
            if se_values is not None:
                self.params["se"] = se_values
        else:
            # This case should ideally not be reached if fit() is called first.
            print("Coefficients not available for SE calculation.")
