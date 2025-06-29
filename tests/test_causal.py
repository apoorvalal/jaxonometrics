import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit # Sigmoid function, same as jax.nn.sigmoid

from jaxonometrics.causal import IPW, AIPW
from jaxonometrics.linear import LinearRegression
from jaxonometrics.mle import Logit

# Function to generate synthetic data for causal inference tests
def generate_causal_data(n_samples=1000, n_features=3, true_ate=2.0, seed=42):
    """
    Generates synthetic data for testing causal inference estimators.
    Args:
        n_samples: Number of samples.
        n_features: Number of covariates.
        true_ate: The true Average Treatment Effect.
        seed: Random seed for reproducibility.

    Returns:
        X (jnp.ndarray): Covariates.
        T (jnp.ndarray): Treatment assignment (0 or 1).
        y (jnp.ndarray): Outcome.
        true_propensity (jnp.ndarray): True propensity scores.
        true_mu0 (jnp.ndarray): True E[Y|X, T=0].
        true_mu1 (jnp.ndarray): True E[Y|X, T=1].
    """
    key = jax.random.PRNGKey(seed)
    key_X, key_T_noise, key_Y_noise, key_beta_T, key_beta_Y = jax.random.split(key, 5)

    X = jax.random.normal(key_X, (n_samples, n_features))

    # True coefficients for propensity score model (logit)
    beta_T_true = jax.random.uniform(key_beta_T, (n_features,), minval=-1, maxval=1)
    true_propensity_logits = X @ beta_T_true - 0.5 # Centering term
    true_propensity = jax.nn.sigmoid(true_propensity_logits)

    # Generate treatment based on true propensity + noise
    T_noise = jax.random.uniform(key_T_noise, (n_samples,))
    T = (T_noise < true_propensity).astype(jnp.int32)

    # True coefficients for outcome model (linear)
    beta_Y_common = jax.random.uniform(key_beta_Y, (n_features,), minval=-0.5, maxval=0.5)

    # E[Y|X,T=0] = X @ beta_Y_common + intercept_0
    # E[Y|X,T=1] = X @ beta_Y_common + intercept_0 + true_ate
    intercept_0 = 0.5

    true_mu0 = X @ beta_Y_common + intercept_0
    true_mu1 = X @ beta_Y_common + intercept_0 + true_ate

    # Outcome = T*mu1 + (1-T)*mu0 + noise
    Y_noise = jax.random.normal(key_Y_noise, (n_samples,)) * 0.5 # Noise level
    y = T * true_mu1 + (1 - T) * true_mu0 + Y_noise

    # Add intercept to X for models that expect it (like our LinearRegression and Logit)
    X_intercept = jnp.hstack([jnp.ones((n_samples, 1)), X])

    return X_intercept, T, y, true_propensity, true_mu0, true_mu1, true_ate


@pytest.fixture
def causal_sim_data():
    return generate_causal_data(n_samples=2000, n_features=3, true_ate=1.5, seed=123)


def test_ipw_ate_estimation(causal_sim_data):
    X, T, y, _, _, _, true_ate = causal_sim_data

    ipw_estimator = IPW(logit_maxiter=10000)
    # The X passed to IPW should not have an intercept if Logit adds one,
    # or Logit should be told not to add one. Our Logit currently assumes X has intercept.
    ipw_estimator.fit(X, T, y)
    estimated_ate = ipw_estimator.params["ate"]

    print(f"IPW - True ATE: {true_ate}, Estimated ATE: {estimated_ate}")
    # Check if the estimated ATE is reasonably close to the true ATE.
    # This can have some variance due to sampling and model misspecification if any.
    assert estimated_ate is not None
    np.testing.assert_allclose(estimated_ate, true_ate, rtol=0.2, atol=0.2) # Looser tolerance for IPW


def test_aipw_ate_estimation(causal_sim_data):
    X, T, y, _, _, _, true_ate = causal_sim_data

    # Using default LinearRegression for outcome, Logit for propensity
    aipw_estimator = AIPW(
        outcome_model=LinearRegression(solver="lineax"), # Explicitly pass an instance
        propensity_model=Logit(maxiter=10000) # Explicitly pass an instance
    )
    # X should include intercept for LinearRegression and Logit as currently implemented
    aipw_estimator.fit(X, T, y)
    estimated_ate = aipw_estimator.params["ate"]

    print(f"AIPW - True ATE: {true_ate}, Estimated ATE: {estimated_ate}")
    # AIPW is often more stable and precise if models are reasonably specified.
    assert estimated_ate is not None
    np.testing.assert_allclose(estimated_ate, true_ate, rtol=0.15, atol=0.15) # Potentially tighter tolerance for AIPW

    # Test if nuisance model parameters are stored
    assert "mu0_params" in aipw_estimator.params
    assert "mu1_params" in aipw_estimator.params
    assert "propensity_scores" in aipw_estimator.params
    assert aipw_estimator.params["mu0_params"] is not None or X[T==0].shape[0] == 0
    assert aipw_estimator.params["mu1_params"] is not None or X[T==1].shape[0] == 0


# Test AIPW with pre-fitted models (not typical usage but tests flexibility)
def test_aipw_with_custom_models(causal_sim_data):
    X, T, y, _, _, _, true_ate = causal_sim_data

    # 1. Fit propensity score model
    ps_model = Logit(maxiter=10000)
    ps_model.fit(X, T) # X includes intercept

    # 2. Fit outcome models
    X_treated = X[T == 1]
    y_treated = y[T == 1]
    X_control = X[T == 0]
    y_control = y[T == 0]

    outcome_model_t = LinearRegression()
    if X_treated.shape[0] > 0:
        outcome_model_t.fit(X_treated, y_treated)

    outcome_model_c = LinearRegression()
    if X_control.shape[0] > 0:
        outcome_model_c.fit(X_control, y_control)

    # Create a custom outcome "model" object for AIPW that uses pre-fitted models
    # This is a bit of a hack for testing; ideally, AIPW would allow passing fitted models
    # or be robust to how models are handled.
    # Our current AIPW re-fits models, so this test is more about ensuring the logic
    # could support it if AIPW was refactored to take already fitted nuisance models.
    # For now, we pass new instances that will be re-fitted by AIPW.

    aipw_estimator = AIPW(
        outcome_model=LinearRegression(), # It will create new instances and fit
        propensity_model=Logit(maxiter=10000) # It will create a new instance and fit
    )
    aipw_estimator.fit(X, T, y)
    estimated_ate = aipw_estimator.params["ate"]

    print(f"AIPW (custom path) - True ATE: {true_ate}, Estimated ATE: {estimated_ate}")
    np.testing.assert_allclose(estimated_ate, true_ate, rtol=0.15, atol=0.15)

# Consider adding a test where one of the nuisance models is deliberately misspecified
# to check the "doubly robust" property (i.e., if one is correct, ATE is still consistent).
# This is more involved to set up correctly.
# For now, these tests cover the basic functionality.
