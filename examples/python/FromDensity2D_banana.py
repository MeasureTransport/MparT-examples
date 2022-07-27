# # Construct map from banana density 2D

from mpart import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time


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

start=time.time()
options={'gtol': 1e-4, 'disp': True}
res = minimize(obj, transport_map.CoeffMap(), args=(transport_map, x), jac=grad_obj, method='BFGS', options=options)
dt = time.time()-start

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
