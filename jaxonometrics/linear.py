from typing import Dict, Optional

import numpy as np

import jax.numpy as jnp
import lineax as lx

from .base import BaseEstimator


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
                # per lineax docs, passing well_posed None is remarkably general:
                # If the operator is non-square, then use lineax.QR. (Most likely case)
                # If the operator is diagonal, then use lineax.Diagonal.
                # If the operator is tridiagonal, then use lineax.Tridiagonal.
                # If the operator is triangular, then use lineax.Triangular.
                # If the matrix is positive or negative (semi-)definite, then use lineax.Cholesky.
            )
            self.params = {"coef": sol.value}

        elif self.solver == "jax":
            sol = jnp.linalg.lstsq(X, y)
            self.params = {"coef": sol[0]}
        elif self.solver == "numpy":  # for completeness
            X, y = np.array(X), np.array(y)
            sol = np.linalg.lstsq(X, y, rcond=None)
            self.params = {"coef": jnp.array(sol[0])}

        if se:
            self._vcov(
                y=y,
                X=X,
                se=se,
            )  # set standard errors in params
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)
        return jnp.dot(X, self.params["coef"])

    def _vcov(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray,
        se: str = "HC1",
    ) -> None:
        n, k = X.shape
        ε = y - X @ self.params["coef"]
        if se == "HC1":
            M = jnp.einsum("ij,i,ik->jk", X, ε**2, X)  # yer a wizard harry
            XtX = jnp.linalg.inv(X.T @ X)
            Σ = XtX @ M @ XtX
            self.params["se"] = jnp.sqrt((n / (n - k)) * jnp.diag(Σ))
        elif se == "classical":
            XtX_inv = jnp.linalg.inv(X.T @ X)
            self.params["se"] = jnp.sqrt(jnp.diag(XtX_inv) * jnp.var(ε, ddof=k))
