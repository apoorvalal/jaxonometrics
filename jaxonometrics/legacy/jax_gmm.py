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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp


# %% [markdown]
# # GMM done easy
#
# moment eqn fn gets a $n \times k$ matrix of residuals (implied by the moment condition) and minimizes the implied objective function.
# Uses implicit gradient (and therefore performs better than `scipy.optimize.minimize`).

# %%
from jaxopt import LevenbergMarquardt
from tqdm import tqdm
from joblib import Parallel, delayed

jax.config.update("jax_enable_x64", True)


# %%
# IV dataset
np.random.seed(42)
def dgp(b = np.array([1, 2]), N = 1_000, K = 3):
    # Instruments
    Z = np.random.normal(0, 1, (N, K))
    # Covariates
    pi = np.random.uniform(1, 2, K)
    w = Z @ pi + np.random.normal(0, 1, N) # endogenous treatment
    X = np.c_[np.ones(N), w]
    # Outcome
    y = X @ b + np.random.normal(0, 1, N)
    return {"y": y, "X": X, "Z": Z}



# %%
@jax.jit
def moment_cond(b, dat):
    y, X, Z = jnp.array(dat['y']), jnp.array(dat['X']), jnp.array(dat['Z'])
    resid = y - X @ b
    return jnp.array(Z * resid[:, None])

def solve_gmm(dat):
    x_init = jnp.zeros(dat['X'].shape[1])
    gn = LevenbergMarquardt(moment_cond)
    return gn.run(x_init, dat = dat).params

solve_gmm(dgp())


# %%
# %%time
def gmmsim():
    df = dgp()
    return solve_gmm(df)

res = Parallel(n_jobs = -1)(delayed(gmmsim)() for _ in tqdm(range(1000)))
res = np.array(res)


# %%
f, ax = plt.subplots(1, 2, figsize=(12, 4))
# Plot histogram of the 'intercept' column
ax[0].hist(np.array(res[:, 0]), color="red", alpha=0.5)
ax[0].axvline(x=1, color="black", linestyle="--", label="True Value")

# Plot histogram of the 'slope' column
ax[1].hist(np.array(res[:, 1]), color="red", alpha=0.5)
ax[1].axvline(x=2, color="black", linestyle="--", label="True Value")

plt.show()

# %% [markdown]
# ## balancing weights
#
# solve dual of ERM problem for exact balance. For details, see [Wang and Zubizarreta](http://jrzubizarreta.com/minimal.pdf)

# %%
from jaxopt import LBFGS
import empirical_calibration as ec
import matplotlib.pyplot as plt


# %%
@jax.jit
def eb_moment(b, X0, X1):
    return jnp.log(jnp.exp(-1 * X0 @ b).sum()) + X1 @ b


def ebwt(X0, X1):
    init_par = jnp.repeat(1.0, X0.shape[1])
    solver = LBFGS(fun=eb_moment, maxiter=100)
    res = solver.run(init_par, X0=X0, X1=X1)
    wt = np.exp(-1 * X0 @ res.params)
    wt /= wt.sum()
    return wt


# %%
sim = ec.data.kang_schafer.Simulation(size=1000)
w, y, X = sim.treatment, sim.outcome, np.c_[np.repeat(1, 1000), sim.covariates]
X0, X1 = X[w == 0, :], X[w == 1, :].mean(0)
b_start = np.random.rand(X.shape[1])
eb_moment(b_start, X0, X1)
# wt = ebwt(X0, X1)


# %%
# kang shafer 2007 dgp
def onesim():
    sim = ec.data.kang_schafer.Simulation(size=1000)
    w, y, X = sim.treatment, sim.outcome, np.c_[np.repeat(1, 1000), sim.covariates]
    X0, X1 = X[w == 0, :], X[w == 1, :].mean(0)
    wt = ebwt(X0, X1)
    naive = y[w == 1].mean() - y[w == 0].mean()
    wtd = y[w == 1].mean() - np.average(y[w == 0], weights=wt)
    return naive, wtd

onesim()


# %%
# %%time
res = Parallel(n_jobs=-1)(delayed(onesim)() for _ in tqdm(range(1_000)))


# %%
simres = pd.DataFrame(res, columns=["naive", "wtd"])

sns.histplot(simres["naive"], color="red", alpha=0.5, kde=True, label="naive")
sns.histplot(simres["wtd"], color="blue", alpha=0.5, kde=True, label="wtd")
plt.xlabel("bias")
plt.legend()
