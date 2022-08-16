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
# # Transport Map from density
#
# The objective of this example is to show how a transport map can be build in MParT when the the unnormalized probability density function of the target density is known.

# + [markdown] id="esisiuaAFutM"
# ## Problem description
#
# We consider $T(\mathbf{z};\mathbf{w})$ a monotone triangular transport map parameterized by $\mathbf{w}$ (e.g., polynomial coefficients). This map which is invertible and has an invertible Jacobian for any parameter $\mathbf{w}$, transports samples $\mathbf{z}^i$ from the reference density $\eta$ to samples $T(\mathbf{z}^i;\mathbf{w})$ from the map induced density $\tilde{\pi}_\mathbf{w}(\mathbf{z})$ defined as:
# $$ \tilde{\pi}_\mathbf{w}(\mathbf{z}) = \eta(T^{-1}(\mathbf{z};\mathbf{w}))|\text{det } T^{-1}(\mathbf{z};\mathbf{w})|,$$
# where $\text{det } T^{-1}$ is the determinant of the inverse map Jacobian at the point $\mathbf{z}$. We refer to $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ as the *map-induced* density or *pushforward distribution* and will commonly interchange notation for densities and measures to use the notation $\tilde{\pi} = T_{\sharp} \eta$.
#
# The objective of this example is, knowing some unnormalized target density $\bar{\pi}$, find the map $T$ that transport samples drawn from $\eta$ to samples drawn from the target $\pi$.

# + [markdown] id="qBQhKZKO_zUC"
# ## Imports
# First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1660685063832, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="8HEOlZ5P_3D3" outputId="b777ffce-e1e9-4fa4-defa-cfcf486ea0e9"
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
# ## Target density and exact map
#
# In this example we use a 2D target density known as the *banana* density where the unnormalized probability density, samples and the exact transport map are known.
#
# The banana density is defined as:
# $$
# \pi(x_1,x_2) \propto N_1(x_1)\times N_1(x_2-x_1^2)
# $$
# where $N_1$ is the 1D standard normal density.
#
# The exact transport map that transport the 2D standard normal density to $\pi$ is known as:
# $$
#     {T}^\text{true}(z_1,z_2)=
#     \begin{bmatrix}
# z_1\\
# z_2 + z_1^2
# \end{bmatrix}
# $$

# + [markdown] id="zXg7O7e28ACZ"
# Contours of the target density can be visualized as:

# + colab={"base_uri": "https://localhost:8080/", "height": 418} executionInfo={"elapsed": 1496, "status": "ok", "timestamp": 1660685065326, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="4lp6EcTg7-47" outputId="026b0c0e-336e-4d9f-d294-5bf79200beae"
# Unnomalized target density required for objective
def target_logpdf(x): 
  rv1 = multivariate_normal(np.zeros(1),np.eye(1))
  rv2 = multivariate_normal(np.zeros(1),np.eye(1))
  logpdf1 = rv1.logpdf(x[0])
  logpdf2 = rv2.logpdf(x[1]-x[0]**2)
  logpdf = logpdf1 + logpdf2
  return logpdf

# Gride for plotting
ngrid=100
x1_t = np.linspace(-3,3,ngrid)
x2_t = np.linspace(-3,7.5,ngrid)
xx1,xx2 = np.meshgrid(x1_t,x2_t)

xx = np.vstack((xx1.reshape(1,-1),xx2.reshape(1,-1)))

# For plotting and computing densities

target_pdf_at_grid = np.exp(target_logpdf(xx))

fig, ax = plt.subplots()
CS1 = ax.contour(xx1, xx2, target_pdf_at_grid.reshape(ngrid,ngrid))
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
h1,_ = CS1.legend_elements()
legend1 = ax.legend([h1[0]], ['target density'])
plt.show()


# + [markdown] id="jgwN-yxSqJH6"
# ## Map training
# ### Defining objective function and its gradient
# Knowing the closed form of the unnormalized target density $\bar{\pi}$, the objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{z})$ that is a good approximation of the target $\pi$.
#
# In order to characterize this posterior density, one method is to build a monotone triangular transport map $T$ such that the KL divergence $D_{KL}(\eta || T^\sharp \pi)$ is minmized. If $T$ is map parameterized by $\mathbf{w}$, the objective function derived from the discrete KL divergence reads:
#
# $$
# J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),
# $$
#
# where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ to the target density $\pi(\mathbf{z})$. The gradient of this objective function reads
#
# $$
# \nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).
# $$

# + [markdown] id="5KqOCRqq54lb"
# The objective function and gradient can be defined using MParT as:

# + executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1660685065327, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="peZCC2Bi6pBb"
# KL divergence objective
def obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    logpdf= target_logpdf(map_of_x)
    log_det = transport_map.LogDeterminant(x)
    return -np.sum(logpdf + log_det)/num_points

# Gradient of unnomalized target density required for gradient objective
def target_grad_logpdf(x):
  grad1 = -x[0,:] + (2*x[0,:]*(x[1,:]-x[0,:]**2))
  grad2 = (x[0,:]**2-x[1,:])
  return np.vstack((grad1,grad2))

# Gradient of KL divergence objective
def grad_obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    sens_vecs = target_grad_logpdf(map_of_x)
    grad_logpdf = transport_map.CoeffGrad(x, sens_vecs)
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x)
    return -np.sum(grad_logpdf + grad_log_det, 1)/num_points


# + [markdown] id="pXYBixXl6ohJ"
# ### Map parameterization

# + [markdown] id="ylbGzU1L97LJ"
# For the parameterization of $T$ we use a total order multivariate expansion of hermite functions. Knowing $T^\text{true}$, any parameterization with total order greater than one will include the true solution of the map finding problem.

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1660685065327, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="goxPI2vPHAlk"
# Set-up first component and initialize map coefficients
map_options = mt.MapOptions()

total_order = 2

# Create dimension 2 triangular map 
transport_map = mt.CreateTriangular(2,2,total_order,map_options)

# + [markdown] id="QJBY9SlnKcDL"
# ### Approximation before optimization
#
# Coefficients of triangular map are set to 0 upon creation.

# + colab={"base_uri": "https://localhost:8080/", "height": 396} executionInfo={"elapsed": 1757, "status": "ok", "timestamp": 1660685067081, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="p9Ce64DDLewW" outputId="e06cf570-5de0-4e50-ce06-2005988c85e8"
# Make reference samples for training
num_points = 10000
z = np.random.randn(2,num_points)

# Make reference samples for testing
test_z = np.random.randn(2,5000)

# Pushed samples
x = transport_map.Evaluate(test_z)

# Before optimization plot
plt.figure()
plt.contour(xx1, xx2, target_pdf_at_grid.reshape(ngrid,ngrid))
plt.scatter(x[0],x[1], facecolor='blue', alpha=0.1, label='Pushed samples')
plt.legend()
plt.show()

# + [markdown] id="lh2eRnPmsoCo"
# At initialization, samples are "far" from being distributed according to the banana distribution.

# + [markdown] id="qSi3SLCkMKwi"
# Initial objective and coefficients:

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 461, "status": "ok", "timestamp": 1660687874229, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="sN38ZQ_tOReR" outputId="784b019e-c75c-4cfd-a4bf-a8c5976bfd1c"
# Print initial coeffs and objective
print('==================')
print('Starting coeffs')
print(transport_map.CoeffMap())
print('Initial objective value: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, test_z)))
print('==================')

# + [markdown] id="uUIHJZYiQ2qH"
# ### Minimization

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 422, "status": "ok", "timestamp": 1660687878949, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="UQPYuHgbMOSN" outputId="5ff0a767-a8b5-4489-8ef8-e716544a626a"
print('==================')
options={'gtol': 1e-4, 'disp': True}
res = minimize(obj, transport_map.CoeffMap(), args=(transport_map, z), jac=grad_obj, method='BFGS', options=options)

# Print final coeffs and objective
print('Final coeffs:')
print(transport_map.CoeffMap())
print('Final objective value: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, test_z)))
print('==================')

# + [markdown] id="rz8sIzXARqhX"
# ### Approximation after optimization

# + [markdown] id="3oZ4tBzvS8gY"
# #### Pushed samples

# + colab={"base_uri": "https://localhost:8080/", "height": 396} executionInfo={"elapsed": 2315, "status": "ok", "timestamp": 1660687289413, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="FD1NdxXDRUtJ" outputId="8dfd0767-ff78-4664-c0a1-8a8d0d749661"
# Pushed samples
x = transport_map.Evaluate(test_z)

# After optimization plot
plt.figure()
plt.contour(xx1, xx2, target_pdf_at_grid.reshape(ngrid,ngrid))
plt.scatter(x[0],x[1], facecolor='blue', alpha=0.1, label='Pushed samples')
plt.legend()
plt.show()


# + [markdown] id="HDg9ajFEje08"
# After optimization, pushed samples $T(z^i)$, $z^i \sim \mathcal{N}(0,I)$ are approximately distributed according to the target $\pi$

# + [markdown] id="vTG5Es2nSfSi"
# #### Variance diagnostic

# + [markdown] id="ZhC7Xt0CkDlK"
# A commonly used accuracy check when facing computation maps from density is the so-called variance diagnostic defined as:
#
# $$ \epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log \frac{\rho}{T^\sharp \bar{\pi}} \right] $$

# + [markdown] id="z5ok9USIlEau"
# This diagnostic is asymptotically equivalent to the minimized KL divergence $D_{KL}(\eta || T^\sharp \pi)$ and should converge to zero when the computed map converge to the true map.

# + [markdown] id="XBFrEH_EmGs1"
# The variance diagnostic can be computed as follow:

# + executionInfo={"elapsed": 459, "status": "ok", "timestamp": 1660686511924, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="GB-RIl_OmL2V"
def variance_diagnostic(tri_map,ref,target_logpdf,x):
  ref_logpdf = ref.logpdf(x.T)
  y = tri_map.Evaluate(x)
  pullback_logpdf = target_logpdf(y) + tri_map.LogDeterminant(x)
  diff = ref_logpdf - pullback_logpdf
  expect = np.mean(diff)
  var = 0.5*np.mean((diff-expect)**2)
  return var


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 384, "status": "ok", "timestamp": 1660686747264, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="nec-_RISnzNz" outputId="d204114b-9427-4f52-d8c6-5e8ceafe226a"
# Reference distribution
ref_distribution = multivariate_normal(np.zeros(2),np.eye(2));

# Compute variance diagnostic
var_diag = variance_diagnostic(transport_map,ref_distribution,target_logpdf,test_z)

# Print final coeffs and objective
print('==================')
print('Variance diagnostic: {:.2E}'.format(var_diag))
print('==================')


# + [markdown] id="272AyrE89BO6"
# #### Pushforward density

# + [markdown] id="EXhbZjtmptiY"
# We can also plot the contour of the unnormalized density $\bar{\pi}$ and the pushforward approximation $T_\sharp \eta$:

# + colab={"base_uri": "https://localhost:8080/", "height": 418} executionInfo={"elapsed": 2659, "status": "ok", "timestamp": 1660687269539, "user": {"displayName": "Paul-Baptiste RUBIO", "userId": "15146079832390040200"}, "user_tz": 240} id="NpngB9t89JR7" outputId="d46cc68b-6d55-43e5-9bd8-7ac291bcab74"
# Pushforward definition
def push_forward_pdf(tri_map,ref,x):
  xinv = tri_map.Inverse(x,x)
  log_det_grad_x_inverse = - tri_map.LogDeterminant(xinv)
  log_pdf = ref.logpdf(xinv.T)+log_det_grad_x_inverse
  return np.exp(log_pdf)

map_approx_grid = push_forward_pdf(transport_map,ref_distribution,xx)

fig, ax = plt.subplots()
CS1 = ax.contour(xx1, xx2, target_pdf_at_grid.reshape(ngrid,ngrid))
CS2 = ax.contour(xx1, xx2, map_approx_grid.reshape(ngrid,ngrid),linestyles='--')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
h1,_ = CS1.legend_elements()
h2,_ = CS2.legend_elements()
legend1 = ax.legend([h1[0], h2[0]], ['Unnormalized target', 'TM approximation'])
plt.show()

