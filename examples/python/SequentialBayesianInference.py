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

np.random.seed(8)


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
C = np.cov(samples_train)
A = np.linalg.inv(np.linalg.cholesky(C))
print(A)
C = np.std(samples_train,axis=1)
A = np.array([[1/C[0],0],[0,1/C[1]]])
c = -1 * np.dot(A,np.mean(samples_train,axis=1))

L = mt.AffineMap(np.asfortranarray(A),c)

samples_train_st = L.Evaluate(samples_train)
samples_test_st = L.Evaluate(samples_test)

print(np.mean(samples_train_st))
print(np.mean(samples_test_st))

# -

# Plot standardized samples
fig,ax = plt.subplots()
ax.plot(samples_train_st[0,:],samples_train_st[1,:],'g.')
plt.show()

# +
obj = mt.CreateGaussianKLObjective(np.asfortranarray(samples_train_st),np.asfortranarray(samples_test_st),2)
map_options = mt.MapOptions()
map_options.basisLB = -3
map_options.basisUB = 3
S = mt.CreateTriangular(2,2,2,map_options)

print('train normal')
# Train map
train_options = mt.TrainOptions()
train_options.verbose = 1
train_options.opt_maxtime = 5

# -

mt.TrainMap(S, obj, train_options)

# +
X_std = S.Evaluate(samples_train_st)

plt.figure()
plt.plot(X_std[0,:],X_std[1,:],'r.')
plt.show()

# +
atm_opts=mt.ATMOptions()
atm_opts.basisLB = -3
atm_opts.basisUB = 3
atm_opts.verbose = 1
atm_opts.maxSize = 10
msets = [mt.MultiIndexSet.CreateTotalOrder(1,1),mt.MultiIndexSet.CreateTotalOrder(2,1)]

print('train ATM')

[mset,atm_map] = mt.AdaptiveTransportMap(msets,obj,atm_opts)

# +
X_std2 = atm_map.Evaluate(samples_train_st)

plt.figure()
plt.plot(X_std2[0,:],X_std2[1,:],'r.')
plt.show()
# -


