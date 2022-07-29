import numpy as np
from pandas import MultiIndex
import scipy 
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import multivariate_normal
import time
import os

os.environ['KOKKOS_NUM_THREADS'] = '8'
from mpart import *

print('Kokkos is using', Concurrency(), 'threads')

rho1 = multivariate_normal(np.zeros(1),np.eye(1))

def generate_SV_samples(d,N):
    # Sample hyper-parameters
    sigma = 0.25
    mu = np.random.randn(1,N)
    phis = 3+np.random.randn(1,N)
    phi = 2*np.exp(phis)/(1+np.exp(phis))-1 #Doesn't seem to follow paper
    X = np.vstack((mu,phi))
    if d > 2:
        # Sample Z0
        Z = np.sqrt(1/(1-phi**2))*np.random.randn(1,N) + mu

        # Sample auto-regressively
        for i in range(d-3):
            Zi = mu + phi * (Z[-1,:] - mu)+sigma*np.random.randn(1,N)
            Z = np.vstack((Z,Zi))

        X = np.vstack((X,Z))
    return X

def normpdf(x,mu,sigma):
    return  np.exp(-0.5 * ((x - mu)/sigma)**2) / (np.sqrt(2*np.pi) * sigma);

def SV_log_pdf(X):
    sigma = 0.25

    # Extract variables mu, phi and states
    mu = X[0,:]
    phi = X[1,:]
    Z = X[2:,:]

    # Compute density for mu
    piMu = multivariate_normal(np.zeros(1),np.eye(1))
    logPdfMu = piMu.logpdf(mu)
    # Compute density for phi
    phiRef = np.log((1+phi)/(1-phi))
    dphiRef = 2/(1-phi**2)
    piPhi = multivariate_normal(3*np.ones(1),np.eye(1))
    logPdfPhi = piPhi.logpdf(phiRef)+np.log(dphiRef)
    # Add piMu, piPhi to density
    logPdf = np.vstack((logPdfMu,logPdfPhi))

    # Number of time steps
    dz = Z.shape[0]
    if dz > 0:
        # Conditonal density for Z_0
        muZ0 = mu
        stdZ0 = np.sqrt(1/(1-phi**2))
        logPdfZ0 = np.log(normpdf(Z[0,:],muZ0,stdZ0))
        logPdf = np.vstack((logPdf,logPdfZ0))

        # Compute auto-regressive conditional densities for Z_i|Z_{1:i-1}
        for i in range(1,dz):
            meanZi = mu + phi * (Z[i-1,:]-mu)
            stdZi = sigma
            logPdfZi = np.log(normpdf(Z[i,:],meanZi,stdZi))
            logPdf = np.vstack((logPdf,logPdfZi))

    return logPdf


T=20 #number of time steps
d=T+2

N = 10000 #Number of training samples
X = generate_SV_samples(d,N)

# # Plot 1D some 1D marginals
# plt.figure()
# plt.hist(X[0,:].flatten(), 50, facecolor='blue', alpha=0.5, density=True,label=r'$\mu$')
# plt.legend()
# plt.show()

# plt.figure()
# plt.hist(X[1,:].flatten(), 50, facecolor='blue', alpha=0.5, density=True,label=r'$\phi$')
# plt.legend()
# plt.show()

# plt.figure()
# plt.hist(X[-1,:].flatten(), 50, facecolor='blue', alpha=0.5, density=True,label=r'$Z_{'+str(T)+'}$')
# plt.legend()
# plt.show()

# Negative log likelihood objective
def obj(coeffs, tri_map,x):
    """ Evaluates the log-likelihood of the samples using the map-induced density. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Compute the map-induced density at each point 
    map_of_x = tri_map.Evaluate(x)
    rho_of_map_of_x = rho1.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)

    # Return the negative log-likelihood of the entire dataset
    return -np.sum(rho_of_map_of_x + log_det)/num_points
    
def grad_obj(coeffs, tri_map, x):
    """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Evaluate the map
    map_of_x = tri_map.Evaluate(x)

    # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
    grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)

    # Get the gradient of the log determinant with respect to the map coefficients
    grad_log_det = tri_map.LogDeterminantCoeffGrad(x)
    
    return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points


def log_cond_pullback_pdf(triMap,rho,x):
    r = triMap.Evaluate(x)
    log_pdf = rho.logpdf(r.T)+triMap.LogDeterminant(x)
    return log_pdf

def log_cond_composed_pullback_pdf(triMap,mu,L,rho,yx):
    Lyx = mu.reshape(-1,1)+np.dot(L,yx)
    eval = triMap.Evaluate(Lyx)
    log_pdf = rho.logpdf(eval.T)+triMap.LogDeterminant(yx)+np.log(np.linalg.det(Linv))
    return log_pdf

def compute_joint_KL(logPdfTM,logPdfSV):
    KL = np.zeros((logPdfSV.shape[0],))
    for k in range(1,d+1):
        logPdfApp = np.sum(logPdfTM[:k,:],axis=0)
        logPdfTru = np.sum(logPdfSV[:k,:],axis=0)
        KL[k-1] = np.mean(logPdfTru - logPdfApp)
    return KL

Ntest=5000
Xtest = generate_SV_samples(d,Ntest)
logPdfSV = SV_log_pdf(Xtest)

opts = MapOptions()

total_order = 2;
noneLim = NoneLim()
mset_to= MultiIndexSet.CreateTotalOrder(4,total_order,noneLim)
multis1 = MultiIndex([0,3,0,0,0,0])

mset_to +=multis1

multis=np.zeros((mset_to.Size(),6))
for s in range(mset_to.Size()):
    multis_to = np.array([mset_to[s].tolist()])
    print(multis_to)
    multis[s,:2]=multis_to[0,:2]
    multis[s,-2:]=multis_to[0,-2:]

print(multis)

multis = np.arange(0,6).reshape(-1,1)
print(multis.shape)

multis = np.array([[0], [1], [2], [3], [4], [5]])
print(multis.shape)
# mset = mset + multis1