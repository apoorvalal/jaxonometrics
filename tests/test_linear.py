import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from jaxonometrics.linear import LinearRegression


def test_linear_regression():
    """Test the LinearRegression estimator against scikit-learn."""
    X = np.random.rand(100, 10)
    y = X @ np.random.rand(10) + np.random.rand(100)

    # scikit-learn
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X, y)
    sklearn_coef = sklearn_model.coef_

    # jaxonometrics
    X_with_intercept = jnp.c_[jnp.ones(X.shape[0]), X]
    jax_model = LinearRegression()
    jax_model.fit(X_with_intercept, jnp.array(y))
    jax_coef = jax_model.params["coef"][1:]

    assert np.allclose(sklearn_coef, jax_coef, atol=1e-6)
