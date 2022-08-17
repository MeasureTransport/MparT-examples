# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
# ---

# + [markdown] id="UK4jDZgoYNEy"
# # Density estimation with sparse transport maps

# + [markdown] id="dJjmXwhl5ycX"
# In this example we demonstrate how MParT can be use to build map with certain sparse structure in order to characterize high dimensional densities with conditional independence.

# + [markdown] id="-M-J-1Ip9GI6"
# ## Imports
# First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.

# +
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm

import os
os.environ['KOKKOS_NUM_THREADS'] = '8'

import mpart as mt
print('Kokkos is using', mt.Concurrency(), 'threads')
plt.rcParams['figure.dpi'] = 110


# + [markdown] id="j5q6kiOc89hO"
# ## Stochastic volatility model

# + [markdown] id="q_jbx-vw_Jyj"
# ### Problem description
#
# The problem considered here is a Markov process that describes the volatility on a financial asset overt time. The model depends on two hyperparamters $\mu$ and $\phi$ and state variable $Z_k$ represents log-volatility at times $k=1,...,T$. The log-volatility follows the order-one autoregressive process:
# $$Z_{k+1} = \mu + \phi(Z_k-\mu) + \epsilon_k, k>1, $$
# where
# $$\mu \sim \mathcal{N}(0,1) $$
# $$ \phi = 2\frac{\exp(\phi^*)}{1+\exp(\phi^*)}, \,\,\, \phi^* \sim \mathcal{N}(3,1)$$
# $$ Z_0 | \mu, \phi \sim \mathcal{N}\left(\mu,\frac{1}{1-\phi^2}\right)$$
#
# The objective is to characterize the joint density of
# $$\mathbf{X}_T = (\mu,\phi,Z_1,...,Z_T), $$
# with $T$ being arbitrarly large. 

# + [markdown] id="3_fpCC6onGoX"
# The conditional independence property for this problem reads
#
# $$ \pi(\mathbf{x}_t|\mathbf{x}_{<t}) = \pi(\mathbf{x}_t|\mathbf{x}_{t-1},\mu,\phi)$$
#
# More details about this problem can be found in [[Baptista et al., 2022]](https://arxiv.org/pdf/2009.10303.pdf).

# + [markdown] id="l-ptbM85KhPr"
# ### Sampling
#
#

# + [markdown] id="pzXh-4UMK_2P"
# Drawing samples $(\mu^i,\phi^i,x_0^i,x_1^i,...,x_T^i)$ can be performed by the following function
# -

def generate_SV_samples(d,N):
    # Sample hyper-parameters
    sigma = 0.25
    mu = np.random.randn(1,N)
    phis = 3+np.random.randn(1,N)
    phi = 2*np.exp(phis)/(1+np.exp(phis))-1
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


# + [markdown] id="Uo8WUYnKp0Q7"
# Set dimension of the problem:
# -

T = 40 #number of time steps including initial condition
d = T+2

# + [markdown] id="V3N7CiDbperf"
# Few realizations of the process look like

# +
Nvisu = 10 #Number of samples
Xvisu = generate_SV_samples(d, Nvisu)

Zvisu = Xvisu[2:,:]

plt.figure()
plt.plot(Xvisu);
plt.xlabel("Days (d)");

# + [markdown] id="zWuJ0dpbqYTC"
# And corresponding realization of hyperparameters
# -

hyper_params = Xvisu[:2,:]
plt.figure()
plt.plot(range(1,11),Xvisu[1,:],label='$\mu$')
plt.plot(range(1,11),Xvisu[2,:],label='$\phi$')
plt.xlabel('Samples')
plt.legend()
plt.show()


# + [markdown] id="F0grplKrri4S"
# ### Probability density function
#

# + [markdown] id="k81EnLHyrqTo"
# The exact log-conditional densities used to define joint density $\pi(\mathbf{x}_T)$ are defined by the following function:
# -

def SV_log_pdf(X):
    
    def normpdf(x,mu,sigma):
        return  np.exp(-0.5 * ((x - mu)/sigma)**2) / (np.sqrt(2*np.pi) * sigma);
    
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


# + [markdown] id="nfecF3RTuh1P"
# ## Transport map training

# + [markdown] id="7XYflgO7ygGY"
# In the following we optimize each map component $S_k$, $k \in \{1,...,T+2\}$:

# + [markdown] id="F4UYJhdkzj6b"
# * For $k=1$, map $S_1$ characterize marginal density $\pi(\mu)$
# * For $k=2$, map $S_2$ characterize conditional density $\pi(\phi|\mu)$
# * For $k=3$, map $S_3$ characterize conditional density $\pi(z_0|\phi,\mu)$
# * For $k>3$, map $S_k$ characterize conditional density $\pi(z_{k-2}|z_{k-3},\phi,\mu)$

# + [markdown] id="1bAkmdgJ1pwW"
# Definition of log-conditional density from map component $S_k$
# -

def log_cond_pullback_pdf(triMap,eta,x):
    r = triMap.Evaluate(x)
    log_pdf = eta.logpdf(r.T)+triMap.LogDeterminant(x)
    return log_pdf


# + [markdown] id="b31RjgM8vA21"
# ### Generating training and testing samples

# + [markdown] id="uB4bPc9eunR6"
# From training samples generated with the known function we compare accuracy of the transport map induced density using different parameterization and a limited number of training samples.

# +
N = 2000 #Number of training samples
X = generate_SV_samples(d, N)

Ntest = 5000 # Number of testing samples
Xtest = generate_SV_samples(d,Ntest)


# + [markdown] id="_uzZrxbjwp-p"
# ### Objective function and gradient

# + [markdown] id="CcEy09YMw2V7"
# We use the minimization of negative log-likelihood to optimize map components.

# + [markdown] id="4gTKOcAjxRYR"
# For map component $k$, the objective function is given by 
#
# $$
# J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left( \log\eta\left(S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right) + \log \frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right)
# $$

# + [markdown] id="EOvAc9SIxTeP"
# and corresponding gradient
# $$
# \nabla_{\mathbf{w}_k}J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left(\left[\nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right]^T \nabla_\mathbf{r}\log \eta \left(S_k
# (\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right) - \frac{\partial \nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k} \left[\frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right]^{-1}\right),
# $$

# +
### Negative log likelihood objective
def obj(coeffs, tri_map,x):
    """ Evaluates the log-likelihood of the samples using the map-induced density. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Compute the map-induced density at each point 
    map_of_x = tri_map.Evaluate(x)

    rho = multivariate_normal(np.zeros(tri_map.outputDim),np.eye(tri_map.outputDim))
    rho_of_map_of_x = rho.logpdf(map_of_x.T)
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


# + [markdown] id="u0JQYqd6xwCR"
# ### Training total order 1 map

# + [markdown] id="bTNK5SNq08Ej"
# Here we use a total order 1 multivariate expansion to parameterize each component $S_k$, $k \in \{1,...,T+2\}$.
# -

opts = mt.MapOptions()
opts.basisType = mt.BasisTypes.HermiteFunctions 

# + [markdown] id="CvdioRau3XAr"
# #### Optimization

# +
# Total order 1 approximation
totalOrder = 1;
logPdfTM = np.zeros((Ntest,))
ListCoeffs_to1=[];
for dk in tqdm(range(1,d+1),desc="Map component"):
    fixed_mset= mt.FixedMultiIndexSet(dk,totalOrder)
    S = mt.CreateComponent(fixed_mset,opts)
    Xtrain = X[:dk,:]
    Xtestk = Xtest[:dk,:]

    ListCoeffs_to1.append(S.numCoeffs)

    options={'gtol': 1e-3, 'disp': False}
    res = minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    
    # Reference density
    eta = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))

    # Compute log-conditional density at testing samples
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,eta,Xtestk)))

logPdfTM_to1=logPdfTM[1:,:]


# + [markdown] id="LvNBKvx62ni6"
# #### Compute KL divergence error
#
# Since we know what the true is for problem we can compute the KL divergence $D_{KL}(\pi(\mathbf{x}_t)||S^\sharp \eta)$ between the map-induced density and the true density.

# +
logPdfSV = SV_log_pdf(Xtest) # true log-pdf

def compute_joint_KL(logPdfSV,logPdfTM):
    KL = np.zeros((logPdfSV.shape[0],))
    for k in range(1,d+1):
        KL[k-1]=np.mean(np.sum(logPdfSV[:k,:],0)-np.sum(logPdfTM[:k,:],0))
    return KL

# Compute joint KL divergence for total order 1 approximation
KL_to1 = compute_joint_KL(logPdfSV,logPdfTM_to1)

# + [markdown] id="2LZMEkKP3buP"
# ### Training total order 2 map

# + [markdown] id="8XCi_k4y3-nh"
# Here we use a total order 2 multivariate expansion to parameterize each component $S_k$, $k \in \{1,...,T+2\}$.

# + [markdown] id="seev6_Fg4WL5"
# #### Optimization
#
# This step can take quite a long time depending of the number of time steps

# +
# Total order 2 approximation
totalOrder = 2;
logPdfTM = np.zeros((Ntest,))
ListCoeffs_to2=[];
for dk in tqdm(range(1,d+1),desc="Map component"):
    fixed_mset= mt.FixedMultiIndexSet(dk,totalOrder)
    S = mt.CreateComponent(fixed_mset,opts)
    Xtrain = X[:dk,:]
    Xtestk = Xtest[:dk,:]

    ListCoeffs_to2.append(S.numCoeffs)

    options={'gtol': 1e-3, 'disp': False}
    res = minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    
    # Reference density
    eta = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))

    # Compute log-conditional density at testing samples
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,eta,Xtestk)))

logPdfTM_to2=logPdfTM[1:,:]


# + [markdown] id="EhyDk50c4VHr"
# #### Compute KL divergence error
# -

# Compute joint KL divergence for total order 2 approximation
KL_to2 = compute_joint_KL(logPdfSV,logPdfTM_to2)

# + [markdown] id="Oey4aiyM4q6I"
# ### Training sparse map

# + [markdown] id="sXsCZxC94xbK"
# Here we use the prior knowledge of the conditional independence property of the target density $\pi(\mathbf{x}_T)$ to parameterize map components with a map structure.

# + [markdown] id="4paQiiRL-RfW"
# #### Prior knowledge used to parameterize map components

# + [markdown] id="z6iq1wFU5UEW"
# From the independence structure mentionned in the problem formulation we have:
#
#
# *   $\pi(\mu,\phi)=\pi(\mu)\pi(\phi)$, meaning $S_2$ only dependes on $\phi$
# *   $\pi(z_{k-2}|z_{k-3},...,z_{0},\phi,\mu)=\pi(z_{k-2}|z_{k-3},\phi,\mu),\,\, k>3$,  meaning $S_k$, only depends on $z_{k-2}$,$z_{k-3}$, $\phi$ and $\mu$
#
#

# + [markdown] id="3Ga4caTs8Gvu"
# Complexity of map component can also be deducted from problem formulation:
#
#
# *   $\pi(\mu)$ being a normal distribution, $S_1$ should be of order 1.
# *  $\pi(\phi)$ is non-Gaussian such that $S_2$ should be nonlinear.
# *  $\pi(z_{k-2}|z_{k-3},\phi,\mu)$ can be represented by a total order 2 parameterization due to the linear autoregressive model.
#
#
#

# + [markdown] id="dWbkOtyA_vDz"
# Hence multi-index sets used for this problem are:
#
#
# *   $k=1$: 1D expansion of order $\geq$ 1
# *   $k=2$: 1D expansion (depending on last component) of high order $>1$
# *   $k=3$: 3D expansion of total order 2
# *   $k>3$: 4D expansion (depending on first two and last two components) of total order 2
#
#

# + [markdown] id="NHy0d80v-f61"
# #### Optimization

# +
totalOrder = 2;
logPdfTM = np.zeros((Ntest,))
ListCoeffs_sa=[];

# MultiIndexSet for map S_k, k>3
mset_to= mt.MultiIndexSet.CreateTotalOrder(4,totalOrder,mt.NoneLim()) 

maxOrder=9 # order for map S_2
for dk in tqdm(range(1,d+1),desc="Map component"):
    if dk == 1:
        fixed_mset= mt.FixedMultiIndexSet(1,totalOrder)
        S = mt.CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1)
    elif dk == 2:
        fixed_mset= mt.FixedMultiIndexSet(1,maxOrder)
        S = mt.CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1) 
    elif dk==3:
        fixed_mset= mt.FixedMultiIndexSet(dk,totalOrder)
        S = mt.CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]
    else:
        multis=np.zeros((mset_to.Size(),dk))
        for s in range(mset_to.Size()):
            multis_to = np.array([mset_to[s].tolist()])
            multis[s,:2]=multis_to[0,:2]
            multis[s,-2:]=multis_to[0,-2:]
        mset = mt.MultiIndexSet(multis)
        fixed_mset = mset.fix(True)
        S = mt.CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]

    ListCoeffs_sa.append(S.numCoeffs)

    options={'gtol': 1e-3, 'disp': False}
    res = minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    rho = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))    
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,rho,Xtestk)))
logPdfTM_sa=logPdfTM[1:,:]

# + [markdown] id="KC9spVVxB9iV"
# #### Compute KL divergence error
# -

# Compute joint KL divergence
KL_sa = compute_joint_KL(logPdfSV,logPdfTM_sa)

# + [markdown] id="mnvo24bVCFs6"
# ## Compare approximations

# + [markdown] id="ikaECsqkCLOi"
# ### KL divergence
# -

# Compare map approximations 
fig, ax = plt.subplots()
ax.plot(range(1,d+1),KL_to1,'-o',label='Total order 1')
ax.plot(range(1,d+1),KL_to2,'-o',label='Total order 2')
ax.plot(range(1,d+1),KL_sa,'-o',label='Sparse MultiIndexSet')
ax.set_yscale('log')
ax.set_xlabel('d')
ax.set_ylabel('$D_{KL}(\pi(\mathbf{x}_t)||S^\sharp \eta)$')
plt.legend()
plt.show()

# + [markdown] id="kY4MM1QyDB8C"
# Usually increasing map complexity will improve map approximation. However when the number of parameters increases too much compared to the number of samples, computed map overfits the data which lead to worst approximation. This overfitting can be seen in this examples when looking at the total order 2 approximation that slowly loses accuracy when the dimension increases. Total order 2 approximation while performing better than order 1 for low dimension perform worst when dimension is greater than ~27. approximation thatn total order 1 with dimension greater than 27.
#
# Using sparse multi-index sets help reduces the increase of parameters when the dimension increases leading to better approximation for all dimensions.

# + [markdown] id="-I6-fheDCQkc"
# ### Map coefficients

# + [markdown] id="QL-g57KVevEO"
# To complement observations made above, we visualize the number of parameters (polyniomal coefficients) for each map parameterization.
# -

fig, ax =plt.subplots()
ax.plot(range(1,d+1),ListCoeffs_to1,'-o',label='Total order 1')
ax.plot(range(1,d+1),ListCoeffs_to2,'-o',label='Total order 2')
ax.plot(range(1,d+1),ListCoeffs_sa,'-o',label='Sparse MultiIndexSet')
ax.set_xlabel('d')
ax.set_ylabel('# coefficients')
plt.legend()
plt.show()

# + [markdown] id="g3t2RU6qfJHe"
# We can observe the exponential growth of the number coefficients for the total order 2 approximation. Chosen sparse multi-index sets have a fixed number of parameters which become smaller than the number of parameters of the total order 1 approximation when dimension is about 15. 

# + [markdown] id="ISjr27_zgr_O"
# Using less parameters help error scaling with the number of dimension but also computation time for the optimization and the computation time when evaluating the transport map. 
