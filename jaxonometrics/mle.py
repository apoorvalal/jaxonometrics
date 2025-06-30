from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax


from .base import BaseEstimator


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Base class for Maximum Likelihood Estimators using Optax.
    """

    def __init__(
        self,
        optimizer: Optional[optax.GradientTransformation] = None,
        maxiter: int = 5000,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.optimizer = optimizer if optimizer is not None else optax.lbfgs()
        self.maxiter = maxiter
        # Tol is not directly used by basic optax loops for stopping but can be a reference
        # or used if a convergence check is manually added.
        self.tol = tol
        self.params: Dict[str, jnp.ndarray] = {}  # Initialize params
        self.history: Dict[str, list] = {"loss": []}  # To store loss history

    @abstractmethod
    def _negative_log_likelihood(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Computes the negative log-likelihood for the model.
        Must be implemented by subclasses.
        Args:
            params: Model parameters.
            X: Design matrix.
            y: Target vector.
        Returns:
            Negative log-likelihood value.
        """
        raise NotImplementedError

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        init_params: Optional[jnp.ndarray] = None,
        verbose: bool = False,
    ) -> "MaximumLikelihoodEstimator":
        """
        Fit the model using the specified Optax optimizer.

        Args:
            X: Design matrix of shape (n_samples, n_features).
               It's assumed that X includes an intercept column if one is desired.
            y: Target vector of shape (n_samples,).
            init_params: Optional initial parameters. If None, defaults to zeros
                         or small random numbers if a PRNGKey can be obtained.

        Returns:
            The fitted estimator.
        """
        n_features = X.shape[1]
        if init_params is None:
            try:  # Try to use a key for initialization for better starting points
                key = jax.random.PRNGKey(0)  # Simple fixed key for reproducibility
                init_params_val = jax.random.normal(key, (n_features,)) * 0.01
            except Exception:  # Fallback if key generation fails or not in context
                init_params_val = jnp.zeros(n_features)
        else:
            init_params_val = init_params

        # Define the loss function to be used by value_and_grad
        # This function now closes over X and y
        def loss_fn(params_lg):
            return self._negative_log_likelihood(params_lg, X, y)

        # Get the gradient function
        value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

        # Initialize optimizer state
        opt_state = self.optimizer.init(init_params_val)

        current_params = init_params_val
        self.history["loss"] = []  # Reset loss history

        # Optimization loop
        for i in range(self.maxiter):
            loss_val, grads = value_and_grad_fn(current_params, state=opt_state)
            updates, opt_state = self.optimizer.update(
                grads,
                opt_state,
                current_params,
                value=loss_val,
                grad=grads,
                value_fn=loss_fn,
            )
            current_params = optax.apply_updates(
                current_params,
                updates,
            )
            self.history["loss"].append(loss_val)
            if i > 10 and self.tol > 0:
                loss_change = abs(
                    self.history["loss"][-2] - self.history["loss"][-1]
                ) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        self.params = {"coef": current_params}
        self.iterations_run = i + 1  # Store how many iterations actually ran

        return self

    def summary(self) -> None:
        """Print a summary of the model results."""
        if not self.params or "coef" not in self.params:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 30)
        print(f"Optimizer: {self.optimizer}")
        if hasattr(self, "iterations_run"):
            print(
                f"Optimization ran for {self.iterations_run}/{self.maxiter} iterations."
            )
        if self.history["loss"]:
            print(f"Final Loss: {self.history['loss'][-1]:.4e}")

        print(f"Coefficients: {self.params['coef']}")
        print("=" * 30)


class LogisticRegression(MaximumLikelihoodEstimator):
    """
    Logistic Regression model.
    """

    def _negative_log_likelihood(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Computes the negative log-likelihood for logistic regression.
        NLL = -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        where p_i = σ(X_i @ β) = 1 / (1 + exp(-X_i @ β))
        Using numerically stable log_sigmoid:
        log(p_i) = log_sigmoid(X_i @ β)
        log(1-p_i) = log_sigmoid(-(X_i @ β))
        """
        logits = X @ params
        # alt: Using jax.nn.log_sigmoid for log(σ(z)) and log(1-σ(z)) = log(σ(-z))
        h = jax.scipy.special.expit(logits)
        nll = -jnp.sum(y * jnp.log(h) + (1 - y) * jnp.log1p(-h))
        return nll  # / X.shape[0] if averaging

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict probabilities for each class.
        Args:
            X: Design matrix of shape (n_samples, n_features).
        Returns:
            Array of probabilities of shape (n_samples,).
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        logits = X @ self.params["coef"]
        return jax.nn.sigmoid(logits)  # jax.scipy.special.expit is equivalent

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

    def _negative_log_likelihood(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Computes the negative log-likelihood for Poisson regression.
        The log(y_i!) term is constant w.r.t params, so ignored for optimization.
        NLL = Σ [exp(X_i @ β) - y_i * (X_i @ β)]
        """
        linear_predictor = X @ params
        lambda_i = jnp.exp(linear_predictor)  # Predicted rates

        # Sum over samples
        nll = jnp.sum(lambda_i - y * linear_predictor)
        return nll  # / X.shape[0] if averaging

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict expected counts (lambda_i).
        Args:
            X: Design matrix of shape (n_samples, n_features).
        Returns:
            Array of predicted counts.
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        linear_predictor = X @ self.params["coef"]
        return jnp.exp(linear_predictor)
