# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: metrics
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# %%
# create dataset
X, y, β = make_regression(n_features=3, coef=True, random_state=42)
X, X_test, y, y_test = train_test_split(X, y)
X, X_test = jnp.c_[jnp.ones(X.shape[0]), X], jnp.c_[jnp.ones(X_test.shape[0]), X_test]
β, X[:2, :], y[:2]

# %%
# param dict
params = {
    "b": jnp.zeros(X.shape[1]),
}


# forward pass: Xbeta
def forward(params, X):
    return jnp.dot(X, params["b"])


@jax.jit
def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)


def update(params, grads, lr=0.05):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


# %%
# the main training loop
for _ in range(50):
    loss = loss_fn(params, X_test, y_test)
    grads = grad_fn(params, X, y)
    params = update(params, grads)


# %%
β, params['b'][1:]
