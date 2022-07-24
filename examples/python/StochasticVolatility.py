import numpy as np
import scipy 
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import multivariate_normal
import time
import os

os.environ['KOKKOS_NUM_THREADS'] = '8'
from mpart import *

print('Kokkos is using', Concurrency(), 'threads')


def generate_SV_samples(d,N):
    # Sample hyper-parameters
    sigma = 0.25
    mu = np.random.randn(1,N)
    phis = 3+np.random.randn(1,N)
    phi = 2*np.exp(phis)/(1+np.exp(phis))-1 #Doesn't seem to follow paper

    # Sample Z0
    Z = np.sqrt(1/(1-phi**2))*np.random.randn(1,N) + mu

    # Sample auto-regressively
    for i in range(d-3):
        Zi = mu + phi * (Z[-1,:] - mu)+sigma*np.random.randn(1,N)
        Z = np.vstack((Z,Zi))

    X = np.vstack((mu,phi,Z))
    return X

X = generate_SV_samples(12,100)

print(X.shape)