import jax.numpy as jnp
import numpy as np

from jaxonometrics.gmm import GMM, LinearIVGMM


def dgp(b=np.array([1, 2]), N=1_000, K=3):
    """A simple data generating process for IV."""
    np.random.seed(42)
    Z = np.random.normal(0, 1, (N, K))
    pi = np.random.uniform(1, 2, K)
    w = Z @ pi + np.random.normal(0, 1, N)
    X = np.c_[np.ones(N), w]
    y = X @ b + np.random.normal(0, 1, N)
    return {"y": y, "X": X, "Z": Z}


def iv_moment_fn(params, data):
    """The moment condition for IV."""
    y, X, Z = data["y"], data["X"], data["Z"]
    resid = y - X @ params
    return Z * resid[:, None]


def test_gmm():
    """Test the general-purpose GMM estimator."""
    data = dgp()
    gmm = GMM(moment_fn=iv_moment_fn)
    init_params = jnp.zeros(data["X"].shape[1])
    gmm.fit(data, init_params)
    assert np.allclose(gmm.params, np.array([1, 2]), atol=1e-1)


def test_linear_iv_gmm():
    """Test the LinearIVGMM estimator."""
    data = dgp()
    iv_gmm = LinearIVGMM()
    iv_gmm.fit(data["X"], data["y"], data["Z"])
    assert np.allclose(iv_gmm.params, np.array([1, 2]), atol=1e-1)
