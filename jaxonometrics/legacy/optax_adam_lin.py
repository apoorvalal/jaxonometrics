# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)


# %%
n, p = 1000, 10

print(β := jax.random.uniform(key, (p,)))
X = jax.random.normal(key, (n, p))
y = X @ β + jax.random.normal(key, (n,))


# %% [markdown]
# ## single-layer neural network with linear activation function (OLS)

# %%
@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
    return jnp.dot(params, x)


def compute_loss(params, x, y):
    y_pred = network(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss


# %% [markdown]
# Optimisation

# %%
optimizer = optax.adam(1e-2)

# Initialize parameters of the model + optimizer.
params = jnp.repeat(0.0, p)
opt_state = optimizer.init(params)
# A simple update loop.
for _ in range(1000):
    grads = jax.grad(compute_loss)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

# %%
print(np.c_[
        β,          # truth
        params,        # opt estimate
        np.linalg.lstsq(X, y, rcond=None)[0] # closed form
        ])


# %% [markdown]
# Works well.
