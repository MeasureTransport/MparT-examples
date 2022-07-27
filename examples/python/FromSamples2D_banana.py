# # Construct map from banana distribution samples in 2D


from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Make target samples for training
num_points = 1000
z = np.random.randn(2,num_points)
x1 = z[0]
x2 = z[1] + z[0]**2
x = np.vstack([x1,x2])


# Make target samples for testing
test_z = np.random.randn(2,10000)
test_x1 = test_z[0]
test_x2 = test_z[1] + test_z[0]**2
test_x = np.vstack([test_x1,test_x2])


# For plotting and computing reference density 
ref_distribution = multivariate_normal(np.zeros(2),np.eye(2))  #standard normal
t = np.linspace(-5,5,100)
grid = np.meshgrid(t,t)
ref_pdf_at_grid = ref_distribution.pdf(np.dstack(grid))


# Set-up map and initize map coefficients
map_options = MapOptions()
transport_map = CreateTriangular(2,2,2,map_options)
coeffs = np.zeros(transport_map.numCoeffs)
transport_map.SetCoeffs(coeffs)


# KL divergence objective
def obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    ref_logpdf = ref_distribution.logpdf(map_of_x.T)
    log_det = transport_map.LogDeterminant(x)
    return -np.sum(ref_logpdf + log_det)/num_points


def grad_obj(coeffs, transport_map, x):
    transport_map.SetCoeffs(coeffs)
    map_of_x = transport_map.Evaluate(x)
    grad_ref_logpdf = -transport_map.CoeffGrad(x, map_of_x)
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x)
    return -np.sum(grad_ref_logpdf + grad_log_det, 1)/num_points


# Before optimization plot
plt.figure()
plt.contour(*grid, ref_pdf_at_grid)
plt.scatter(test_x[0],test_x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()


# Print initial coeffs and objective
print('Starting coeffs')
print(transport_map.CoeffMap())
print('and error: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, x)))
print('==================')


# Optimize
optimizer_options={'gtol': 1e-4, 'disp': True}
res = minimize(obj, transport_map.CoeffMap(), args=(transport_map, x), jac=grad_obj, method='BFGS', options=optimizer_options)


# Print final coeffs and objective
print('Final coeffs')
print(transport_map.CoeffMap())
print('and error: {:.2E}'.format(obj(transport_map.CoeffMap(), transport_map, x)))
print('==================')


# After optimization plot
map_of_test_x = transport_map.Evaluate(test_x)
plt.figure()
plt.contour(*grid, ref_pdf_at_grid)
plt.scatter(map_of_test_x[0],map_of_test_x[1], facecolor='blue', alpha=0.1, label='Normalized samples')
plt.legend()
plt.show()


# Print statistics of normalized samples (TODO replace with better Gaussianity check)
print('==================')
mean_of_map = np.mean(map_of_test_x,1)
print("Mean of normalized test samples")
print(mean_of_map)
print('==================')
print("Cov of normalized test samples")
cov_of_map = np.cov(map_of_test_x)
print(cov_of_map)