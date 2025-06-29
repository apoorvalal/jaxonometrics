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
import numpy as np
import statsmodels.api as sm

# %%
import torch
from torchmin import minimize as torchmin

# %%
import jax.numpy as jnp
import jax

from jax.scipy.optimize import minimize as jaxmin


# %%
def dgp(n=500, k=3, reg=False, torchout = False):
    X = np.random.normal(0, 1, n * k).reshape(n, k)
    Y0 = X[:, 0] + X[:, 0] ** 2 + np.random.uniform(-0.5, 0.5, n)
    Y1 = X[:, 1] + X[:, 1] ** 2 + np.random.uniform(-1, 1, n)
    Z = np.random.binomial(1, 0.6, n)
    Y = Y0 * (1 - Z) + Y1 * Z
    if torchout:
        return torch.tensor(np.c_[Y, Z, X])
    return np.c_[Y, Z, X]

def linreg(data):
    Y, Z, X = data[:, 0], data[:, 1], data[:, 2:]
    XX = sm.add_constant(np.c_[Z, X, Z[:, None] * X.mean(axis = 0)])
    model = sm.OLS(Y, XX).fit()
    return model.params

linreg(dgp())


# %% [markdown]
# $$
# Y_i = \alpha + \tau W_i + X_i'\beta + W_i \tilde{X}_i'\gamma + \epsilon_i
# $$
#
# where $\tilde{X}_i = X_i - \bar{X}$ is a centered version of $X_i$. The goal is to estimate $\tau$.

# %% [markdown]
# ## Torch

# %%
def moment_cond_cent(theta, data):
    Z, Y, X = data[:, 1], data[:, 0], data[:, 2:]
    n, p = X.shape
    mu, beta = theta[:p], theta[p:]
    Xcent = X - mu
    ones = torch.ones(n, 1, device=data.device)
    Xtilde = torch.cat([ones, Z.view(-1, 1), X, Z.view(-1, 1) * Xcent], dim=1)
    resid = (Y - torch.matmul(Xtilde, beta)).view(-1, 1)
    m = Xtilde * resid
    return torch.cat([m, Xcent], dim=1)


# %%
data = dgp(n = 1_000, k = 2, torchout=True)
print(np.round(linreg(data.numpy()), 3))
# m-estimation
k = data.shape[1]
theta_init = torch.tensor(np.random.rand((k-1)*2 +(k-2)))

def loss(theta):
    m = moment_cond_cent(theta, data)
    return torch.sum(m.mean(axis=0) ** 2)

params = torchmin(loss, x0 = theta_init, method="l-bfgs")
params.x[2:].numpy().round(3)


# %% [markdown]
# ## JAX

# %%
def moment_cond_jax(theta, data):
    Z, Y, X = jnp.array(data[:, 1]), jnp.array(data[:, 0]), jnp.array(data[:, 2:])
    n, p = X.shape
    mu, beta = theta[:p], theta[p:]
    Xcent = X - mu
    Xtilde = jnp.c_[np.ones(n), Z, X, Z[:, None] * Xcent]
    resid = (Y - Xtilde @ beta)[:, None]
    m = Xtilde * resid
    return jnp.c_[m, Xcent]


# %%
def gmm_objective(theta, data):
    moments = moment_cond_jax(theta, data)
    return np.sum(moments.mean(axis=0) ** 2)

@jax.jit
def optimize_gmm(theta_init, data):
    return jaxmin(lambda theta: gmm_objective(theta, data), theta_init, method="BFGS")



# %%
data = dgp(n = 1_000, k = 2, torchout=False)
print(np.round(linreg(data), 3))

k = data.shape[1]
theta_init = np.random.rand((k-1)*2 +(k-2))

print(np.round(optimize_gmm(theta_init, data).x[2:], 3))
