# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3.10 (recommended)
#     language: python
#     name: python310
# ---

# %% [markdown]
# # Minimum Norm Interpolant

# %%
import numpy as np
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS


# %% [markdown]
# $n>p$ dgp, OLS solution not unique

# %%
def sparse_dgp(n = 10_000, p = 20_000, eta = 0.1):
    X = np.c_[np.repeat(1, n),
            np.random.normal(size = n*p).reshape((n, p))
        ]
    # initialize coef vector
    β, nzcount = np.repeat(0.0, p + 1), int(eta * p)
    # choose nzcount number of non-zero coef
    nzid = np.random.choice(p, nzcount, replace=False)
    # set them to random values
    β[nzid] = np.random.randn(nzcount)
    # build in heteroskedasticity
    e = np.random.normal(0, 0.5 + (0.1 * X[:, 1]>0), n)
    # generate y
    y = X @ β + e
    return y, X

y, X = sparse_dgp()


# %% [markdown]
# ### statsmodels

# %%
# %%time
smols = OLS(y, X).fit()

# %%
np.linalg.norm(smols.params)

# %% [markdown]
# Statsmodels is very slow with such problems.

# %% [markdown]
# ### scikit

# %%
# %%time
m = LinearRegression()
m.fit(X, y)
(y - m.predict(X)).max()


# %%
np.linalg.norm(m.coef_)


# %% [markdown]
# ### lineax
#
# Very fast least squares solver (including for minimum norm interpolation problems). 
#

# %%
# %%time
sol = lx.linear_solve(                                    # solve # Ax = b
        operator = lx.MatrixLinearOperator(jnp.array(X)), # A
        vector = jnp.array(y),                            # b
        solver=lx.AutoLinearSolver(well_posed=None),      
    )

betahat = sol.value
# does it interpolate
(y - X @ betahat).max()

# %%
np.linalg.norm(betahat)

