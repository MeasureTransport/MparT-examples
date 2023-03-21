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

N=20000; # total number of samples
X = mu0 + std0*np.random.randn(1,N);
Y = sigma_eff(sigma_w,sigma_i,X);

samples = np.vstack((X,Y))

n_train = int(samples.shape[1]*0.8)

samples_train = samples[:,:n_train]
samples_test = samples[:,n_train:]


# -

samples_test.shape

opts=mt.ATMOptions()
opts.basisLB = -3
opts.basisUB = 3
opts.verbose = 1

opts

train_options

atm_opts = mt.ATMOptions()

atm_opts


