# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="u_tcbBTTACPG"
# # Transport Map from Density

# + [markdown] id="esisiuaAFutM"
# ##### Problem description
#
# Knowing the closed form of unnormalized posterior $\bar{\pi}(\boldsymbol{\theta} |\mathbf{y})= \pi(\mathbf{y}|\boldsymbol{\theta})\pi(\boldsymbol{\theta})$, the objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ that is a good approximation to the posterior $\pi(\boldsymbol{\theta} |\mathbf{y})$.
#
# In order to characterize this posterior density, one method is to build a transport map.
#
# For the map from unnormalized density estimation, the objective function reads
#
# $$
# J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),
# $$
#
# where $T$ is the transport map pushing forward the standard normal $mathcal{N}(\mathbf{0},\mathbf{I}_d)$ to the target density $\pi(\mathbf{x})$, which will be the the posterior density in Bayesian inference applications.  The gradient of this objective function reads
#
# $$
# \nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).
# $$

# + [markdown] id="qBQhKZKO_zUC"
# ## Imports
# First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.

# + id="8HEOlZ5P_3D3"
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import os
os.environ['KOKKOS_NUM_THREADS'] = '8'

import mpart as mt
print('Kokkos is using', mt.Concurrency(), 'threads')
plt.rcParams['figure.dpi'] = 110

# + [markdown] id="B0xLTmNXASl4"
# ## Reference density and samples
#
# In this example we use a 2D target density known as the *banana* density where the unnormalized probability density, samples and the exact transport map are known.
#
# The banana density is defined as:
# $$
# \pi(x_1,x_2) \propto N_1(x_1)\times N_1(x_2-x_1^2)
# $$
# where $N_1$ is the 1D standard normal density.
#
# The exact transport map that transport $\pi$ to the 2D standard normal density is defined as:
# $$
#     {T}^\text{true}(x_1,x_2)=
#     \begin{bmatrix}
# x_1\\
# x_2 + x_1^2
# \end{bmatrix}
# $$

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 972, "status": "ok", "timestamp": 1660413440049, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="TUHp3Q1HQm66" outputId="8884120b-dd73-4451-ce48-9b1dc26d9866"
# Make reference samples for training
num_points = 1000
x = np.random.randn(2,num_points)

# Make target samples for testing
test_x = np.random.randn(2,10000)


def target_logpdf(x):
  rv1 = multivariate_normal(np.zeros(1),np.eye(1))
  rv2 = multivariate_normal(np.zeros(1),np.eye(1))
  logpdf1 = rv1.logpdf(x[0])
  logpdf2 = rv2.logpdf(x[1]-x[0]**2)
  logpdf = logpdf1 + logpdf2
  return logpdf


def target_grad_logpdf(x):
  grad1 = -x[0,:] + (2*x[0,:]*(x[1,:]-x[0,:]**2))
  grad2 = (x[0,:]**2-x[1,:])
  return np.vstack((grad1,grad2))


Ngrid = 100
ref_distribution = multivariate_normal(np.zeros(2),np.eye(2))  #standard normal
t = np.linspace(-5,5,Ngrid)
grid = np.meshgrid(t,t)
target_logpdf_at_grid = target_logpdf([grid[0].flatten(),grid[1].flatten()]).reshape(Ngrid,Ngrid)
target_pdf_at_grid = np.exp(target_logpdf_at_grid)


# Set-up map and initize map coefficients
opts = MapOptions()
map_order = 2
transport_map = CreateTriangular(2,2,map_order,opts)
coeffs = np.zeros(transport_map.numCoeffs)
transport_map.SetCoeffs(coeffs)


# Before optimization plot
plt.figure()
plt.contour(*grid, target_pdf_at_grid)
plt.scatter(test_x[0],test_x[1], facecolor='blue', alpha=0.1, label='Reference samples')
plt.legend()
plt.show()


# KL divergence objective
def obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    logpdf= target_logpdf(map_of_x)
    log_det = transport_map.LogDeterminant(x)
    return -np.sum(logpdf + log_det)/num_points


def grad_obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    sens_vecs = target_grad_logpdf(map_of_x)
    grad_logpdf = transport_map.CoeffGrad(x, sens_vecs)
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x)
    return -np.sum(grad_logpdf + grad_log_det, 1)/num_points


# Print initial coeffs and objective
print('Starting coeffs')
print(transport_map.CoeffMap())
print('and error: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, x)))
print('==================')

options={'gtol': 1e-4, 'disp': True}
res = minimize(obj, transport_map.CoeffMap(), args=(transport_map, x), jac=grad_obj, method='BFGS', options=options)

# Print final coeffs and objective
print('Final coeffs')
print(transport_map.CoeffMap())
print('and error: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, x)))
print('==================')


# After optimization plot
map_of_test_x = transport_map.Evaluate(test_x)
plt.figure()
plt.contour(*grid, target_pdf_at_grid)
plt.scatter(map_of_test_x[0],map_of_test_x[1], facecolor='blue', alpha=0.1, label='Push-forward samples')
plt.legend()
plt.show()
