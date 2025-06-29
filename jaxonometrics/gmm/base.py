from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt, OptaxSolver
import optax

from ..base import BaseEstimator


class GMM(BaseEstimator):
    """
    A general-purpose GMM estimator.

    This class provides a flexible interface for fitting GMM models. It can be
    configured with different solvers and is not tied to any specific model.
    """

    def __init__(self, moment_fn: Callable, solver: str = "lm", **solver_kwargs):
        """
        Initialize the GMM estimator.

        Args:
            moment_fn: A function that computes the moment conditions.
                It should have the signature `moment_fn(params, data)`.
            solver: The solver to use. Can be "lm" for Levenberg-Marquardt
                or "sgd" for stochastic gradient descent.
            **solver_kwargs: Additional keyword arguments to pass to the solver.
        """
        super().__init__()
        self.moment_fn = moment_fn
        self.solver_name = solver
        self.solver_kwargs = solver_kwargs

    def fit(
        self,
        data: Dict,
        init_params: Optional[jnp.ndarray] = None,
    ) -> "GMM":
        """
        Fit the GMM model.

        Args:
            data: A dictionary containing the data.
            init_params: Initial values for the parameters.

        Returns:
            The fitted estimator.
        """
        if self.solver_name == "lm":
            solver = LevenbergMarquardt(self.moment_fn, **self.solver_kwargs)
            sol = solver.run(init_params, data=data)
            self.params = sol.params
        elif self.solver_name == "sgd":
            if "optimizer" not in self.solver_kwargs:
                self.solver_kwargs["optimizer"] = optax.adam(1e-2)
            solver = OptaxSolver(fun=self.moment_fn, **self.solver_kwargs)
            state = solver.init_state(init_params)

            @jax.jit
            def update_step(params, opt_state, data):
                grads, opt_state = solver.update(params=params, state=opt_state, data=data)
                params = optax.apply_updates(params, grads)
                return params, opt_state

            for _ in range(self.solver_kwargs.get("maxiter", 1000)):
                init_params, state = update_step(init_params, state, data)

            self.params = init_params
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

        return self
