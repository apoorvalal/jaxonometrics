from typing import Dict, Optional, Any

import jax # Ensure jax is imported
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
    @jax.jit
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


class IPW(BaseEstimator):
    """
    Inverse Propensity Weighting estimator for Average Treatment Effect (ATE).
    This implementation uses Logistic Regression for propensity score estimation.
    """

    def __init__(self, logit_optimizer: Optional[Any] = None, logit_maxiter: int = 5000):
        """
        Initialize the IPW estimator.

        Args:
            logit_optimizer: Optional jaxopt optimizer for the Logit model.
            logit_maxiter: Maximum iterations for the Logit model optimization.
        """
        super().__init__()
        # Import Logit here to avoid circular dependency issues at module load time
        # if mle.py also imported from causal.py (though it doesn't currently)
        from .mle import Logit
        self.logit_model = Logit(optimizer=logit_optimizer, maxiter=logit_maxiter)
        self.params: Dict[str, Any] = {"ate": None, "propensity_scores": None}


    def fit(self, X: jnp.ndarray, T: jnp.ndarray, y: jnp.ndarray) -> "IPW":
        """
        Estimate the Average Treatment Effect (ATE) using IPW.

        Args:
            X: Covariate matrix of shape (n_samples, n_features).
               It's assumed that X includes an intercept column if one is desired for the propensity score model.
            T: Treatment assignment vector (binary, 0 or 1) of shape (n_samples,).
            y: Outcome vector of shape (n_samples,).

        Returns:
            The fitted estimator with ATE and propensity scores.
        """
        # Ensure T is jnp.ndarray for Logit model
        if not isinstance(T, jnp.ndarray):
            T_jax = jnp.array(T)
        else:
            T_jax = T

        # 1. Estimate propensity scores P(T=1|X) using Logit
        self.logit_model.fit(X, T_jax)
        propensity_scores = self.logit_model.predict_proba(X)

        # Clip propensity scores to avoid division by zero or extreme weights
        epsilon = 1e-6 # Small constant to prevent extreme values
        propensity_scores = jnp.clip(propensity_scores, epsilon, 1 - epsilon)

        self.params["propensity_scores"] = propensity_scores

        # 2. Calculate IPW weights
        # Weight for treated: 1 / p_score
        # Weight for control: 1 / (1 - p_score)
        weights = T_jax / propensity_scores + (1 - T_jax) / (1 - propensity_scores)

        # 3. Estimate ATE: E[Y(1)] - E[Y(0)]
        # E[Y(1)] = sum(T_i * y_i / p_i) / sum(T_i / p_i)
        # E[Y(0)] = sum((1-T_i) * y_i / (1-p_i)) / sum((1-T_i) / (1-p_i))
        # ATE = sum( (T_i/p_i - (1-T_i)/(1-p_i)) * y_i ) / N  (Hahn, 1998 type estimator)
        # Or, more commonly for Horvitz-Thompson type:
        # E[Y_1] = sum(T*y / ps) / sum(T/ps)
        # E[Y_0] = sum((1-T)*y / (1-ps)) / sum((1-T)/(1-ps))
        # ATE = E[Y_1] - E[Y_0]

        # Using the simpler weighted average formulation for ATE:
        # ATE = (1/N) * Σ [ (T_i * Y_i / e(X_i)) - ((1-T_i) * Y_i / (1-e(X_i))) ]
        # This can also be seen as E[ (T - e)Y / (e(1-e)) ]
        # However, the difference of means of weighted outcomes is more standard:

        mean_y1 = jnp.sum(T_jax * y * weights) / jnp.sum(T_jax * weights)
        mean_y0 = jnp.sum((1 - T_jax) * y * weights) / jnp.sum((1 - T_jax) * weights)

        # The above is equivalent to:
        # mean_y1 = jnp.sum( (T_jax * y) / propensity_scores ) / jnp.sum( T_jax / propensity_scores )
        # mean_y0 = jnp.sum( ((1-T_jax) * y) / (1-propensity_scores) ) / jnp.sum( (1-T_jax) / (1-propensity_scores) )

        ate = mean_y1 - mean_y0
        self.params["ate"] = ate

        return self

    def summary(self) -> None:
        super().summary() # Calls BaseEstimator summary
        if self.params and "ate" in self.params and self.params["ate"] is not None:
            print(f"  Estimated ATE: {self.params['ate']:.4f}")
        if self.params and "propensity_scores" in self.params and self.params["propensity_scores"] is not None:
            print(f"  Propensity scores min: {jnp.min(self.params['propensity_scores']):.4f}, max: {jnp.max(self.params['propensity_scores']):.4f}")

# Need to add `Any` to imports for type hinting
# from typing import Dict, Optional, Any
# Need to add this at the top of causal.py

from .linear import LinearRegression # For default outcome model in AIPW

class AIPW(BaseEstimator):
    """
    Augmented Inverse Propensity Weighting (AIPW) estimator for ATE.
    Also known as doubly robust estimator.
    """

    def __init__(self,
                 outcome_model: Optional[BaseEstimator] = None,
                 propensity_model: Optional[Any] = None, # Should be a Logit instance or similar
                 ps_clip_epsilon: float = 1e-6):
        """
        Initialize the AIPW estimator.

        Args:
            outcome_model: A regression model (like LinearRegression or a custom one)
                           to estimate E[Y|X, T=t]. If None, LinearRegression is used.
                           The model should have a `fit(X,y)` and `predict(X)` method.
            propensity_model: A binary classifier (like Logit) to estimate P(T=1|X).
                              If None, Logit() is used. Model should have `fit(X,T)`
                              and `predict_proba(X)` methods.
            ps_clip_epsilon: Small constant to clip propensity scores to avoid extreme values.
        """
        super().__init__()
        from .mle import Logit # Local import for Logit

        self.outcome_model_template = outcome_model if outcome_model else LinearRegression()
        # We need two instances of the outcome model, one for T=1 and one for T=0
        self.propensity_model = propensity_model if propensity_model else Logit()

        self.ps_clip_epsilon = ps_clip_epsilon
        self.params: Dict[str, Any] = {"ate": None, "propensity_scores": None,
                                       "mu0_params": None, "mu1_params": None}

    def fit(self, X: jnp.ndarray, T: jnp.ndarray, y: jnp.ndarray) -> "AIPW":
        """
        Estimate the Average Treatment Effect (ATE) using AIPW.

        Args:
            X: Covariate matrix of shape (n_samples, n_features).
               It's assumed that X includes an intercept column if one is desired for the outcome and propensity score models.
            T: Treatment assignment vector (binary, 0 or 1) of shape (n_samples,).
            y: Outcome vector of shape (n_samples,).

        Returns:
            The fitted estimator with ATE.
        """
        if not isinstance(T, jnp.ndarray): T_jax = jnp.array(T)
        else: T_jax = T
        if not isinstance(y, jnp.ndarray): y_jax = jnp.array(y)
        else: y_jax = y
        if not isinstance(X, jnp.ndarray): X_jax = jnp.array(X)
        else: X_jax = X

        n_samples = X_jax.shape[0]

        # 1. Estimate propensity scores P(T=1|X) = e(X)
        self.propensity_model.fit(X_jax, T_jax)
        propensity_scores = self.propensity_model.predict_proba(X_jax)
        propensity_scores = jnp.clip(propensity_scores, self.ps_clip_epsilon, 1 - self.ps_clip_epsilon)
        self.params["propensity_scores"] = propensity_scores

        # 2. Estimate outcome models E[Y|X, T=1] = μ_1(X) and E[Y|X, T=0] = μ_0(X)
        # Need to handle potential issues if one group has no samples (though unlikely with real data)
        X_treated = X_jax[T_jax == 1]
        y_treated = y_jax[T_jax == 1]
        X_control = X_jax[T_jax == 0]
        y_control = y_jax[T_jax == 0]

        # Create fresh instances of the outcome model for fitting
        # This assumes the outcome_model_template can be re-used (e.g. by creating a new instance or being stateless after fit)
        # For sklearn-like models, this means creating new instances.
        # For our JAX models, they are re-fitted.

        model1 = self.outcome_model_template.__class__() # Create a new instance of the same type
        if X_treated.shape[0] > 0:
            model1.fit(X_treated, y_treated)
            mu1_X = model1.predict(X_jax)
            self.params["mu1_params"] = model1.params
        else: # Should not happen in typical scenarios
            mu1_X = jnp.zeros(n_samples)
            self.params["mu1_params"] = None

        model0 = self.outcome_model_template.__class__() # Create a new instance
        if X_control.shape[0] > 0:
            model0.fit(X_control, y_control)
            mu0_X = model0.predict(X_jax)
            self.params["mu0_params"] = model0.params
        else: # Should not happen
            mu0_X = jnp.zeros(n_samples)
            self.params["mu0_params"] = None


        # 3. Calculate AIPW estimator components
        # ψ_i = μ_1(X_i) - μ_0(X_i) + T_i/e(X_i) * (Y_i - μ_1(X_i)) - (1-T_i)/(1-e(X_i)) * (Y_i - μ_0(X_i))

        term1 = mu1_X - mu0_X
        term2 = (T_jax / propensity_scores) * (y_jax - mu1_X)
        term3 = ((1 - T_jax) / (1 - propensity_scores)) * (y_jax - mu0_X)

        psi_i = term1 + term2 - term3

        ate = jnp.mean(psi_i)
        self.params["ate"] = ate

        return self

    def summary(self) -> None:
        super().summary()
        if self.params and "ate" in self.params and self.params["ate"] is not None:
            print(f"  Estimated ATE (AIPW): {self.params['ate']:.4f}")
        if self.params and "propensity_scores" in self.params and self.params["propensity_scores"] is not None:
            print(f"  Propensity scores min: {jnp.min(self.params['propensity_scores']):.4f}, max: {jnp.max(self.params['propensity_scores']):.4f}")
        # Could add info about outcome model parameters if desired
