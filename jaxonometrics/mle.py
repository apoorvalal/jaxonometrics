from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax

from .base import BaseEstimator


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Base class for Maximum Likelihood Estimators.
    """

    def __init__(self, optimizer: Optional[Any] = None, maxiter: int = 5000, tol: float = 1e-4):
        super().__init__()
        self.optimizer = optimizer if optimizer else optax.adam(learning_rate=1e-3)
        self.maxiter = maxiter
        self.tol = tol # Tolerance for convergence, though OptaxSolver doesn't use it directly for stopping

    @abstractmethod
    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Computes the negative log-likelihood for the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, init_params: Optional[jnp.ndarray] = None) -> "MaximumLikelihoodEstimator":
        """
        Fit the model using the specified optimizer.

        Args:
            X: Design matrix of shape (n_samples, n_features).
               It's assumed that X includes an intercept column if one is desired.
            y: Target vector of shape (n_samples,).
            init_params: Optional initial parameters. If None, defaults to zeros.

        Returns:
            The fitted estimator.
        """
        n_features = X.shape[1]
        if init_params is None:
            # Try to use a key for initialization if available, otherwise simple zeros
            try:
                key = jax.random.PRNGKey(0) # Simple fixed key for reproducibility
                init_params = jax.random.normal(key, (n_features,)) * 0.01
            except:
                init_params = jnp.zeros(n_features)


        # Define the objective function for the solver
        def objective_fn(params, data):
            # Data here is expected to be a tuple (X, y)
            X_data, y_data = data
            return self._negative_log_likelihood(params, X_data, y_data)

        solver = OptaxSolver(fun=objective_fn, opt=self.optimizer, maxiter=self.maxiter, tol=self.tol)

        # Prepare data as a tuple for OptaxSolver
        data = (X, y)
        sol = solver.run(init_params, data=data)
        self.params = {"coef": sol.params}

        # Store optimization results if needed, e.g., sol.state.iter_num
        self.opt_state = sol.state

        return self

    # It's good practice to implement a method for SEs if common for these models
    # For now, focusing on the core fit and predict as per the plan.
    # def _vcov(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
    #     """Placeholder for variance-covariance matrix calculation."""
    #     pass

    def summary(self) -> None:
        """Print a summary of the model results."""
        if self.params is None:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 30)
        if "coef" in self.params:
            print(f"Coefficients: {self.params['coef']}")
        # Could add more details like SEs if calculated
        if hasattr(self, 'opt_state') and hasattr(self.opt_state, 'iter_num') and self.opt_state.iter_num is not None:
            print(f"Optimization terminated after {self.opt_state.iter_num} iterations.")
        elif hasattr(self, 'opt_state') and hasattr(self.opt_state, 'error') and self.opt_state.error is not None:
            # OptaxSolver state might have 'error' if tol was used and not met, or other issues.
            print(f"Optimization completed with error: {self.opt_state.error:.4e}")
        print("=" * 30)


class Logit(MaximumLikelihoodEstimator):
    """
    Logistic Regression model.
    """

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Computes the negative log-likelihood for logistic regression.
        L(β) = -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        where p_i = σ(X_i @ β) = 1 / (1 + exp(-X_i @ β))
        """
        logits = X @ params
        # Log-sum-exp trick for numerical stability: log(1 + exp(x)) = log_sum_exp(0, x)
        # log(p_i) = log(σ(X_i @ β)) = log(1 / (1 + exp(-X_i @ β))) = -log(1 + exp(-X_i @ β))
        # log(1 - p_i) = log(1 - σ(X_i @ β)) = log(exp(-X_i @ β) / (1 + exp(-X_i @ β))) = -X_i @ β - log(1 + exp(-X_i @ β))
        # Using jax.nn.log_sigmoid for log(σ(z)) and log(1-σ(z)) = log(σ(-z))
        # NLL = - sum ( y * log_sigmoid(logits) + (1-y) * log_sigmoid(-logits) )
        log_p = jax.nn.log_sigmoid(logits)
        log_one_minus_p = jax.nn.log_sigmoid(-logits) # log(1 - sigmoid(x)) = log(sigmoid(-x))

        nll = -jnp.sum(y * log_p + (1 - y) * log_one_minus_p)
        return nll / X.shape[0] # Average NLL

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict probabilities for each class.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Array of probabilities of shape (n_samples,).
        """
        if self.params is None or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        logits = X @ self.params["coef"]
        return jax.nn.sigmoid(logits)

    def predict(self, X: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
        """
        Predict class labels.

        Args:
            X: Design matrix of shape (n_samples, n_features).
            threshold: Probability threshold for class assignment.

        Returns:
            Array of predicted class labels (0 or 1).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(jnp.int32)


class PoissonRegression(MaximumLikelihoodEstimator):
    """
    Poisson Regression model.
    """

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Computes the negative log-likelihood for Poisson regression.
        L(β) = -Σ [y_i * (X_i @ β) - exp(X_i @ β) - log(y_i!)]
        The log(y_i!) term is constant w.r.t params, so can be ignored for optimization.
        NLL = Σ [exp(X_i @ β) - y_i * (X_i @ β)]
        """
        linear_predictor = X @ params
        # lambda_i = exp(X_i @ β)
        lambda_i = jnp.exp(linear_predictor)

        # Ignore log(y!) as it's constant wrt params
        nll = jnp.sum(lambda_i - y * linear_predictor)
        return nll / X.shape[0] # Average NLL

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict expected counts.

        Args:
            X: Design matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted counts (lambda_i).
        """
        if self.params is None or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        linear_predictor = X @ self.params["coef"]
        return jnp.exp(linear_predictor)
