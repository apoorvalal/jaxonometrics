import pytest
import jax.numpy as jnp
import numpy as np  # For sklearn data generation and comparison
from sklearn.linear_model import LogisticRegression as SklearnLogit
from sklearn.linear_model import PoissonRegressor as SklearnPoisson
from sklearn.datasets import (
    make_classification,
    make_regression,
)  # make_regression can simulate poisson data
from sklearn.preprocessing import StandardScaler

from jaxonometrics.mle import LogisticRegression, PoissonRegression

import optax


# Fixture for Logit data
@pytest.fixture
def logit_data():
    X_np, y_np = make_classification(
        n_samples=200,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        random_state=42,
        n_classes=2,
    )
    # Add an intercept column
    X_np_intercept = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
    scaler = StandardScaler()
    X_np_scaled = scaler.fit_transform(
        X_np_intercept[:, 1:]
    )  # Scale only non-intercept features
    X_np_intercept_scaled = np.hstack([X_np_intercept[:, :1], X_np_scaled])

    return jnp.array(X_np_intercept_scaled), jnp.array(y_np)


# Fixture for Poisson data
@pytest.fixture
def poisson_data():
    X_np, y_np_reg = make_regression(
        n_samples=200,
        n_features=3,
        n_informative=2,
        n_targets=1,
        random_state=123,
        noise=0.1,
    )
    # Add an intercept column
    X_np_intercept = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
    scaler = StandardScaler()
    X_np_scaled = scaler.fit_transform(X_np_intercept[:, 1:])
    X_np_intercept_scaled = np.hstack([X_np_intercept[:, :1], X_np_scaled])

    # Transform y to be count data, ensuring it's non-negative and integer-like for Poisson
    # This is a simplistic way to generate data that vaguely resembles Poisson counts
    # True Poisson data generation would involve X @ beta and then np.random.poisson(exp(X @ beta))
    # For testing purposes, this should be okay if sklearn's PoissonRegressor can handle it.
    y_poisson_np = np.abs(
        np.round(np.exp(y_np_reg / np.std(y_np_reg) * 0.5))
    )  # Scale and transform

    # Ensure no zeros if using log-link and some y are zero (though sklearn handles it)
    # For our model, y=0 is fine.

    return jnp.array(X_np_intercept_scaled), jnp.array(y_poisson_np)


def test_logit_fit_predict(logit_data):
    X, y = logit_data

    # Jaxonometrics Logit
    jax_logit = LogisticRegression(maxiter=20)
    jax_logit.fit(X, y)
    jax_coef = jax_logit.params["coef"]

    # Sklearn Logit for comparison (using 'liblinear' which is good for small datasets, no penalty)
    # Sklearn's LogisticRegression has regularization by default, so we need to turn it off or make it very weak.
    # C is inverse of regularization strength. Large C = weak regularization.
    # We also need to tell it not to add an intercept if we already have one.
    sklearn_logit = SklearnLogit(
        solver="liblinear",
        C=1e9,
        fit_intercept=False,
        random_state=42,
        tol=1e-6,
    )
    sklearn_logit.fit(np.array(X), np.array(y))
    sklearn_coef = sklearn_logit.coef_.flatten()

    # print(f"Jax Logit Coef: {jax_coef}")
    # print(f"Sklearn Logit Coef: {sklearn_coef}")

    assert jax_coef is not None
    assert jax_coef.shape == (X.shape[1],)
    # Check if coefficients are reasonably close. This can be sensitive to optimizer settings.
    # A looser tolerance might be needed depending on exact optimizer behavior.
    np.testing.assert_allclose(jax_coef, sklearn_coef, rtol=0.1, atol=0.1)

    # Test predict_proba and predict
    jax_probas = jax_logit.predict_proba(X)
    jax_preds = jax_logit.predict(X)

    assert jax_probas.shape == (X.shape[0],)
    assert jnp.all((jax_probas >= 0) & (jax_probas <= 1))
    assert jax_preds.shape == (X.shape[0],)
    assert jnp.all((jax_preds == 0) | (jax_preds == 1))


def test_poisson_fit_predict(poisson_data):
    X, y = poisson_data

    # Jaxonometrics Poisson
    jax_poisson = PoissonRegression(maxiter=50)
    jax_poisson.fit(X, y)
    jax_coef = jax_poisson.params["coef"]

    # Sklearn Poisson for comparison
    # Sklearn's PoissonRegressor also has alpha for regularization (L2). Set to 0 for no regularization.
    sklearn_poisson = SklearnPoisson(
        alpha=0, fit_intercept=False, max_iter=1000, tol=1e-6
    )
    sklearn_poisson.fit(np.array(X), np.array(y))
    sklearn_coef = sklearn_poisson.coef_.flatten()

    # print(f"Jax Poisson Coef: {jax_coef}")
    # print(f"Sklearn Poisson Coef: {sklearn_coef}")

    assert jax_coef is not None
    assert jax_coef.shape == (X.shape[1],)
    # Poisson can be a bit more sensitive
    np.testing.assert_allclose(jax_coef, sklearn_coef, rtol=0.15, atol=0.15)

    # Test predict
    jax_counts = jax_poisson.predict(X)
    assert jax_counts.shape == (X.shape[0],)
    assert jnp.all(jax_counts >= 0)


# It might be good to also add a test for the summary methods,
# but that primarily checks printing, not core functionality.
# For now, focusing on fit/predict.
