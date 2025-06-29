from typing import Dict, Optional

import jax.numpy as jnp
from jaxopt import LBFGS

from .base import BaseEstimator


class EntropyBalancing(BaseEstimator):
    """
    Entropy Balancing for causal inference.

    This class implements the entropy balancing method for estimating causal
    effects by reweighting the control group to match the covariate moments
    of the treated group.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _eb_moment(b: jnp.ndarray, X0: jnp.ndarray, X1: jnp.ndarray) -> jnp.ndarray:
        """The moment condition for entropy balancing."""
        return jnp.log(jnp.exp(-1 * X0 @ b).sum()) + X1 @ b

    def fit(
        self, X0: jnp.ndarray, X1: jnp.ndarray, maxiter: int = 100
    ) -> "EntropyBalancing":
        """
        Compute the entropy balancing weights.

        Args:
            X0: The covariate matrix for the control group.
            X1: The mean of the covariate matrix for the treated group.
            maxiter: The maximum number of iterations for the optimizer.

        Returns:
            The fitted estimator with the computed weights.
        """
        init_par = jnp.repeat(1.0, X0.shape[1])
        solver = LBFGS(fun=self._eb_moment, maxiter=maxiter)
        res = solver.run(init_par, X0=X0, X1=X1)
        wt = jnp.exp(-1 * X0 @ res.params)
        wt /= wt.sum()
        self.params = {"weights": wt}
        return self
