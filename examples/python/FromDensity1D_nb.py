# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Construct map from density
#
# A way to construct a transport map is from an unnormalized density. 

# First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable before importing MParT.

# +
import math

# +
import math

# +
import numpy as np

# +
from scipy.optimize import minimize

# +
from scipy.stats import norm

# +
import matplotlib.pyplot as plt

# +
from mpart import *

# +
print('Kokkos is using', Concurrency(), 'threads')
# -
# The target distribution is given by $x\sim\mathcal{N}(2, 0.5)$.

num_points = 5000
mu = 2
sigma = .5
x = np.random.randn(num_points)[None,:]

# As the reference density we choose the standard normal. 

reference_density = norm(loc=mu,scale=sigma)
t = np.linspace(-3,6,100)
rho_t = reference_density.pdf(t)

num_bins = 50
# Before optimization plot
plt.figure()
plt.hist(x.flatten(), num_bins, facecolor='blue', alpha=0.5, density=True, label='Reference samples')
plt.plot(t, rho_t,label="Target density")
plt.legend()
plt.show()

# Next we create a multi-index set and create a map.

multis = np.array([[0], [1]])  # affine transform enough to capture Gaussian target
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective

def objective(coeffs, monotoneMap, x, rv):
    num_points = x.shape[0]
    monotoneMap.SetCoeffs(coeffs)
    map_of_x = monotoneMap.Evaluate(x)
    pi_of_map_of_x = rv.logpdf(map_of_x)
    log_det = monotoneMap.LogDeterminant(x)
    return -np.sum(pi_of_map_of_x + log_det)/num_points

# Optimize
print('Starting coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap(), monotoneMap, x, reference_density)))
res = minimize(objective, monotoneMap.CoeffMap(), args=(monotoneMap, x, reference_density), method="Nelder-Mead")
print('Final coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap(), monotoneMap, x, reference_density)))

# After optimization plot
map_of_x = monotoneMap.Evaluate(x)
plt.figure()
plt.hist(map_of_x.flatten(), num_bins, facecolor='blue', alpha=0.5, density=True, label='Mapped samples')
plt.plot(t,rho_t,label="Target density")
plt.legend()
plt.show()
