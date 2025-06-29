from typing import Dict, Optional

import jax.numpy as jnp

from .base import GMM


def iv_moment_fn(params: jnp.ndarray, data: Dict) -> jnp.ndarray:
    """Moment condition for linear IV."""
    y, X, Z = data["y"], data["X"], data["Z"]
    residuals = y - X @ params
    return Z * residuals[:, None]


class LinearIVGMM(GMM):
    """
    GMM estimator for linear instrumental variable models.

    This is a convenience class that is pre-configured for linear IV.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LinearIVGMM estimator.

        Args:
            **kwargs: Additional keyword arguments to pass to the GMM solver.
        """
        super().__init__(moment_fn=iv_moment_fn, **kwargs)

    def fit(
        self, X: jnp.ndarray, y: jnp.ndarray, Z: jnp.ndarray, **kwargs
    ) -> "LinearIVGMM":
        """
        Fit the linear IV model.

        Args:
            X: The matrix of endogenous variables.
            y: The vector of outcomes.
            Z: The matrix of instruments.
            **kwargs: Additional keyword arguments to pass to the solver.

        Returns:
            The fitted estimator.
        """
        data = {"X": X, "y": y, "Z": Z}
        init_params = jnp.zeros(X.shape[1])
        super().fit(data, init_params, **kwargs)
        return self
