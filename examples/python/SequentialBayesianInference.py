# ---
# jupyter:
#   jupytext:
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

# +
import numpy as np
import scipy 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import os
os.environ['KOKKOS_NUM_THREADS'] = '8'

import mpart as mt
print('Kokkos is using', mt.Concurrency(), 'threads')
plt.rcParams['figure.dpi'] = 110


# -

# ## Generation of training samples
#
# Forward model:

def sigma_eff(sigma_w,sigma_i,z_iw):
    Rz_iw=1/np.sqrt(4*z_iw**2+1);
    out=sigma_i*(1-Rz_iw)+sigma_w*(Rz_iw);
    return out


# +
# Conductivities
sigma_w=2500; 
sigma_i=22;

x = np.linspace(0.5,4);
y = sigma_eff(sigma_w,sigma_i,x);

# -

plt.figure()
plt.plot(x,y)
plt.show()

# +
# Parameters of the Gaussian proposal
mu0 = 2;
std0 = 0.25;

# Measurement noise
std_m = 63

N=20000; # total number of samples
X = mu0 + std0*np.random.randn(1,N);
Y = sigma_eff(sigma_w,sigma_i,X)+std_m*np.random.randn(1,N);

samples = np.vstack((X,Y))

n_train = int(samples.shape[1]*0.8)

samples_train = samples[:,:n_train]
samples_test = samples[:,n_train:]
# -

# plot samples
fig,ax = plt.subplots()
ax.plot(samples_train[0,:],samples_train[1,:],'r.')
plt.show()

# +
# Standardization of samples
C = np.cov(samples_train);
A = np.linalg.inv(np.linalg.cholesky(C));
c = -1 * np.dot(A,np.mean(samples_train,axis=1));

L = mt.AffineMap(np.asfortranarray(A),c)

samples_train_st = L.Evaluate(samples_train)
samples_test_st = L.Evaluate(samples_test)
# -

# Plot standardized samples
fig,ax = plt.subplots()
ax.plot(samples_train_st[0,:],samples_train_st[1,:],'g.')
plt.show()

atm_opts=mt.ATMOptions()
atm_opts.basisLB = -3
atm_opts.basisUB = 3
atm_opts.verbose = 1
atm_opts.maxSize = 15

msets = [mt.MultiIndexSet.CreateTotalOrder(2,1)]
obj = mt.CreateGaussianKLObjective(np.asfortranarray(samples_train_st),np.asfortranarray(samples_test_st),1)
[mset,atm_map] = mt.AdaptiveTransportMap(msets,obj,atm_opts)

# +
from scipy.stats import norm

X_std = atm_map.Evaluate(samples_test)
fig,ax = plt.subplots()
ax.hist(X_std, density=True, bins=30, alpha=0.5, label='Sample Histogram')
x_axis = np.linspace(-4, 4, 100)
pdf = norm.pdf(x_axis)
plt.plot(x_axis, pdf, color='r', label='Normal PDF')
# -

X_std.shape


