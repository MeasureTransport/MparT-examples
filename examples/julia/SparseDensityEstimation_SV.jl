### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 8ea66ca7-981c-47d6-8513-099aed421e01
using Pkg; Pkg.activate()

# ╔═╡ baab6a84-23c0-11ed-3f3b-01e3ad086ae7
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL, ProgressLogging, Colors, CairoMakie

# ╔═╡ caac1a76-abf2-4ab0-96f9-07310c79628f
ENV["KOKKOS_NUM_THREADS"] = 10

# ╔═╡ e20f5bba-894b-4c9b-a362-f6afe3b8dc18
Base.compilecache(Base.PkgId(MParT))

# ╔═╡ baab6b24-23c0-11ed-0602-bdb17d65ca9b
md"""
# Density estimation with sparse transport maps
"""

# ╔═╡ baab6b7e-23c0-11ed-2af3-c9584eeff5fd
md"""
In this example we demonstrate how MParT can be use to build map with certain sparse structure in order to characterize high dimensional densities with conditional independence.
"""

# ╔═╡ baab6b88-23c0-11ed-021d-8d0a20edce42
md"""
## Imports
First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.
"""

# ╔═╡ baae2daa-23c0-11ed-2739-85c0abcd5341
md"""
## Stochastic volatility model
"""

# ╔═╡ baae2dd2-23c0-11ed-00f8-6d0035b69fe4
md"""
### Problem description

The problem considered here is a Markov process that describes the volatility on a financial asset overt time. The model depends on two hyperparamters $\mu$ and $\phi$ and state variable $Z_k$ represents log-volatility at times $k=1,...,T$. The log-volatility follows the order-one autoregressive process:
```math
Z_{k+1} = \mu + \phi(Z_k-\mu) + \epsilon_k, k>1, 
```
where
```math
\mu \sim \mathcal{N}(0,1) 
```
```math
 \phi = 2\frac{\exp(\phi^*)}{1+\exp(\phi^*)}, \,\,\, \phi^* \sim \mathcal{N}(3,1)
```
```math
 Z_0 | \mu, \phi \sim \mathcal{N}\left(\mu,\frac{1}{1-\phi^2}\right)
```

The objective is to characterize the joint density of
```math
\mathbf{X}_T = (\mu,\phi,Z_1,...,Z_T), 
```
with $T$ being arbitrarly large.
"""

# ╔═╡ baae2eae-23c0-11ed-0fc3-a3c61caa6f21
md"""
The conditional independence property for this problem reads

```math
 \pi(\mathbf{x}_t|\mathbf{x}_{<t}) = \pi(\mathbf{x}_t|\mathbf{x}_{t-1},\mu,\phi)
```

More details about this problem can be found in [[Baptista et al., 2022]](https://arxiv.org/pdf/2009.10303.pdf).
"""

# ╔═╡ baae2ed4-23c0-11ed-2dab-4dca84f92e8b
md"""
### Sampling


"""

# ╔═╡ baae2eea-23c0-11ed-31cc-29625e1c33e0
md"""
Drawing samples $(\mu^i,\phi^i,x_0^i,x_1^i,...,x_T^i)$ can be performed by the following function
"""

# ╔═╡ baae2ef4-23c0-11ed-363f-29c8463b2a72
function generate_SV_samples(d)
    # Sample hyper-parameters
    sigma = 0.25
    mu = randn()
    phis = 3 + randn()
    phi = 2*exp(phis)/(1 + exp(phis)) - 1
    X = zeros(d)
	X[1:2] .= [mu,phi]
    if d  > 2
        # Sample Z0
		Z = zeros(d-2)
        Z[1] = sqrt(1 /(1 - phi^2)) * randn() + mu
		# Sample auto-regressively
        for i in 2:(d-2)
            Z[i] = mu + phi .* (Z[i-1] - mu)+sigma*randn()
		end
		X[3:end] .= Z
	end
	X
end

# ╔═╡ baae8e88-23c0-11ed-2ea4-6d63a48fb345
md"""
Set dimension of the problem:
"""

# ╔═╡ baae8ea8-23c0-11ed-3cd5-e514c1a42f57
begin
T = 10 #number of time steps including initial condition
d = T+2
end

# ╔═╡ baae9894-23c0-11ed-1f4b-dbaa5d0557d2
md"""
Few realizations of the process look like
"""

# ╔═╡ baae98a8-23c0-11ed-271a-f12d2cad5514
begin
	Nvisu = 10 #Number of samples
	Xvisu = reduce(hcat, generate_SV_samples(d) for _ in 1:Nvisu)
	
	Zvisu = Xvisu[3:end,:]
	plt_cols = ["#1f77b4", "#ff7f0e", "#2ca02c",
				"#d62728", "#9467bd", "#8c564b",
				"#e377c2", "#7f7f7f", "#bcbd22",
				"#17becf"]
	fig1 = Figure()
	ax1 = Axis(fig1[1,1], xlabel="Days (d)")
	series!(ax1, Zvisu', color=plt_cols)
	fig1
end

# ╔═╡ baaec170-23c0-11ed-3433-99cd648f9917
md"""
And corresponding realization of hyperparameters
"""

# ╔═╡ baaec184-23c0-11ed-1632-6152101b3a8a
begin
	hyper_params = Xvisu[1:2,:]
	fig2 = Figure()
	ax2 = Axis(fig2[1,1], xlabel="Samples")
	lines!(ax2, 1:Nvisu,Xvisu[2,:],label=L"$\mu$")
	scatter!(ax2, 1:Nvisu,Xvisu[2,:])
	lines!(ax2, 1:Nvisu,Xvisu[3,:],label=L"$\phi$")
	scatter!(ax2, 1:Nvisu,Xvisu[3,:])
	axislegend()
	fig2
end

# ╔═╡ baaee57e-23c0-11ed-36ca-77d6124cb674
md"""
### Probability density function

"""

# ╔═╡ baaee59e-23c0-11ed-3509-c9d5d65b9acb
md"""
The exact log-conditional densities used to define joint density $\pi(\mathbf{x}_T)$ are defined by the following function:
"""

# ╔═╡ baaee5a6-23c0-11ed-10b3-b7ea3dd18770
function SV_log_pdf(X)

    function normpdf(x,mu,sigma)
         exp(-0.5 * ((x - mu)/sigma) .^2) / (sqrt(2*pi) * sigma)
	end

    sigma = 0.25

    # Extract variables mu, phi and states
    mu = X[1]
    phi = X[2]
    Z = X[3:end]

    # Compute density for mu
    piMu = Normal()
    logPdfMu = logpdf(piMu, mu)
    # Compute density for phi
    phiRef = log((1 + phi)./(1 - phi))
    dphiRef = 2/(1 - phi^2)
    piPhi = Normal(3,1)
    logPdfPhi = logpdf(piPhi, phiRef) + log(dphiRef)
    # Add piMu, piPhi to density
    logPdf = zeros(length(X))
	logPdf[1:2] .= [logPdfMu, logPdfPhi]

    # Number of time steps
    dz = length(Z)
    if dz > 0
        # Conditonal density for Z_0
        muZ0 = mu
        stdZ0 = sqrt(1 / (1 - phi^2))
        logPdfZ0 = log(normpdf(Z[1],muZ0,stdZ0))
        logPdf[3] = logPdfZ0

        # Compute auto-regressive conditional densities for Z_i|Z_{1i-1}
        for i in 2:dz
            meanZi = mu + phi * (Z[i-1]-mu)
            stdZi = sigma
            logPdfZi = log(normpdf(Z[i],meanZi,stdZi))
            logPdf[i+2] = logPdfZi
		end
	end
    logPdf
end

# ╔═╡ baafaf7c-23c0-11ed-3d4f-417daa259af1
md"""
## Transport map training
"""

# ╔═╡ baafaf9a-23c0-11ed-0aaa-5158f5057c23
md"""
In the following we optimize each map component $S_k$, $k \in \{1,...,T+2\}$:
"""

# ╔═╡ baafafa6-23c0-11ed-1329-e3c1800b566c
md"""
* For $k=1$, map $S_1$ characterize marginal density $\pi(\mu)$
* For $k=2$, map $S_2$ characterize conditional density $\pi(\phi|\mu)$
* For $k=3$, map $S_3$ characterize conditional density $\pi(z_0|\phi,\mu)$
* For $k>3$, map $S_k$ characterize conditional density $\pi(z_{k-2}|z_{k-3},\phi,\mu)$
"""

# ╔═╡ baafafc2-23c0-11ed-0ab0-1f8eac46d349
md"""
Definition of log-conditional density from map component $S_k$
"""

# ╔═╡ baafafcc-23c0-11ed-2dd3-b56cfcbe5f31
function log_cond_pullback_pdf(tri_map,eta,x)
    r = Evaluate(tri_map, x)
    log_pdf = logpdf(eta, r)+LogDeterminant(tri_map, x)
    log_pdf
end

# ╔═╡ baafc3cc-23c0-11ed-04e4-bd608afe616f
md"""
### Generating training and testing samples
"""

# ╔═╡ baafc3e0-23c0-11ed-12ea-1bbacb622ad0
md"""
From training samples generated with the known function we compare accuracy of the transport map induced density using different parameterization and a limited number of training samples.
"""

# ╔═╡ baafc3ea-23c0-11ed-26b4-a1adec224fc3
begin
	N = 2000 #Number of training samples
	X = reduce(hcat, generate_SV_samples(d) for _ in 1:N)
	
	Ntest = 5000 # Number of testing samples
	Xtest = reduce(hcat, generate_SV_samples(d) for _ in 1:Ntest)
end

# ╔═╡ baaff23e-23c0-11ed-3de6-c31ab40e44eb
md"""
### Objective function and gradient
"""

# ╔═╡ baaff252-23c0-11ed-2fb1-dbfaf5dcca3f
md"""
We use the minimization of negative log-likelihood to optimize map components.
"""

# ╔═╡ baaff25c-23c0-11ed-22dd-63ed96e978a8
md"""
For map component $k$, the objective function is given by

```math
J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left( \log\eta\left(S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right) + \log \frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right)
```
"""

# ╔═╡ baaff2b6-23c0-11ed-21d0-1fc03d600f47
md"""
and corresponding gradient
```math
\nabla_{\mathbf{w}_k}J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left(\left[\nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right]^T \nabla_\mathbf{r}\log \eta \left(S_k
(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right) - \frac{\partial \nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k} \left[\frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right]^{-1}\right),
```
"""

# ╔═╡ baaff2e8-23c0-11ed-0446-e5ad028d990a
begin
	"""
	Evaluates the log-likelihood of the samples using the map-induced density.
	"""
	function obj(coeffs,p)
		tri_map,x = p
	    
	    num_points = size(x,2)
	    SetCoeffs(tri_map, coeffs)
	
	    # Compute the map-induced density at each point
	    map_of_x = Evaluate(tri_map, x)
	    rho = MvNormal(I(outputDim(tri_map)))
	    rho_of_map_of_x = logpdf(rho, map_of_x)
	    log_det = LogDeterminant(tri_map, x)
	
	    # Return the negative log-likelihood of the entire dataset
	    -sum(rho_of_map_of_x + log_det)/num_points
	end
	
	"""
	Returns the gradient of the log-likelihood
	objective wrt the map parameters.
	"""
	function grad_obj(g, coeffs,p)
		tri_map, x = p
	    
	    num_points = size(x,2)
	    SetCoeffs(tri_map, coeffs)
	
	    # Evaluate the map
	    map_of_x = Evaluate(tri_map, x)
	
	    # Now compute the inner product of the
		# map jacobian (\nabla_w S) and the gradient
		# (which is just -S(x) here)
	    grad_rho_of_map_of_x = -CoeffGrad(tri_map, x, map_of_x)
	
	    # Get the gradient of the log determinant
		# with respect to the map coefficients
	    grad_log_det = LogDeterminantCoeffGrad(tri_map, x)
	
	    g .= -vec(sum(grad_rho_of_map_of_x + grad_log_det, dims=2))/num_points
	end

end

# ╔═╡ bab09ad4-23c0-11ed-0a08-f5ecbececd44
md"""
### Training total order 1 map
"""

# ╔═╡ bab09aea-23c0-11ed-3f03-3d18da7b26de
md"""
Here we use a total order 1 multivariate expansion to parameterize each component $S_k$, $k \in \{1,...,T+2\}$.
"""

# ╔═╡ bab09afe-23c0-11ed-25ce-abbc835d5a74
opts = MapOptions(basisType="ProbabilistHermite")

# ╔═╡ bab0a622-23c0-11ed-30f0-eb74b2ff4a55
md"""
#### Optimization
"""

# ╔═╡ bab0a634-23c0-11ed-3d4b-2d171a1a4c29
# function order1Approx()
begin
	# Total order 1 approximation
	totalOrder = 1
	logPdfTM_to1 = zeros(d,Ntest)
	ListCoeffs_to1=zeros(d)
	start1 = time_ns()
	@progress "Map component" for dk in 2:d
	    fixed_mset= FixedMultiIndexSet(dk,totalOrder)
	    S = CreateComponent(fixed_mset,opts)
	    Xtrain = X[1:dk,:]
	    Xtestk = Xtest[1:dk,:]
		p = (S,Xtrain)
		
	    ListCoeffs_to1[dk-1]=numCoeffs(S)
		fcn = OptimizationFunction(obj, grad=grad_obj)
		prob = OptimizationProblem(fcn, CoeffMap(S), p, gtol=1e-3)
	    res = solve(prob, BFGS())
	
	    # Reference density
	    eta = MvNormal(I(outputDim(S)))
	
	    # Compute log-conditional density at testing samples
	    logPdfTM_to1[dk-1,:]=log_cond_pullback_pdf(S,eta,Xtestk)
	end
	end1 = time_ns()
	@info "Took $((end1-start1)*1e-9)s"
end

# ╔═╡ bab119ac-23c0-11ed-0500-5f3f9c280dc8
md"""
#### Compute KL divergence error

Since we know what the true is for problem we can compute the KL divergence $D_{KL}(\pi(\mathbf{x}_t)||S^\sharp \eta)$ between the map-induced density and the true density.
"""

# ╔═╡ bab119ca-23c0-11ed-3a4a-853addf54fa2
begin
logPdfSV = reduce(hcat, SV_log_pdf(xx) for xx in eachcol(Xtest)) # true log-pdf

function compute_joint_KL(logPdfSV,logPdfTM)
    KL = zeros(size(logPdfSV,1))
    for k in 1:d
        KL[k]=mean(sum(logPdfSV[1:k,:],dims=1)-sum(logPdfTM[1:k,:],dims=1))
	end
    KL
end

# Compute joint KL divergence for total order 1 approximation
KL_to1 = compute_joint_KL(logPdfSV,logPdfTM_to1)
end

# ╔═╡ bab14bfc-23c0-11ed-222a-29a7836ec165
md"""
### Training total order 2 map
"""

# ╔═╡ bab14c10-23c0-11ed-3464-e95c62236eaa
md"""
Here we use a total order 2 multivariate expansion to parameterize each component $S_k$, $k \in \{1,...,T+2\}$.
"""

# ╔═╡ bab14c1a-23c0-11ed-3524-7b984a8f1ab2
md"""
#### Optimization

This step can take few minutes depending on the number of time steps set at the definition of the problem.
"""

# ╔═╡ bab14c30-23c0-11ed-259c-adc821842c14
begin
	# Total order 2 approximation
	totalOrder2 = 2
	logPdfTM_to2 = zeros(d,Ntest)
	ListCoeffs_to2=zeros(d)
	start2 = time_ns()
	@progress "Map component" for dk in 2:d
	    fixed_mset= FixedMultiIndexSet(dk,totalOrder2)
	    S = CreateComponent(fixed_mset,opts)
	    Xtrain = X[1:dk,:]
	    Xtestk = Xtest[1:dk,:]
		p = (S,Xtrain)
		
		ListCoeffs_to2[dk-1]=numCoeffs(S)
		fcn = OptimizationFunction(obj, grad=grad_obj)
		prob = OptimizationProblem(fcn, CoeffMap(S), p, gtol=1e-3)
		res = solve(prob, BFGS())
	
	    # Reference density
	    eta = MvNormal(I(outputDim(S)))
	
	    # Compute log-conditional density at testing samples
	    logPdfTM_to2[dk-1,:]=log_cond_pullback_pdf(S,eta,Xtestk)
	end
	end2 = time_ns()
	@info "Took $((end2-start2)*1e-9)s"
end

# ╔═╡ bab1b5e2-23c0-11ed-25c7-f3af826473a8
md"""
#### Compute KL divergence error
"""

# ╔═╡ bab1b5f6-23c0-11ed-0542-c595661d7cee
md"""
Compute joint KL divergence for total order 2 approximation
"""

# ╔═╡ 8fbcbcee-90ae-490e-a867-e7806e0ff434
KL_to2 = compute_joint_KL(logPdfSV,logPdfTM_to2)

# ╔═╡ bab1b600-23c0-11ed-3905-e3dcb5ad077d
md"""
### Training sparse map
"""

# ╔═╡ bab1b60a-23c0-11ed-3713-df191e151f37
md"""
Here we use the prior knowledge of the conditional independence property of the target density $\pi(\mathbf{x}_T)$ to parameterize map components with a map structure.
"""

# ╔═╡ bab1b620-23c0-11ed-203f-7b4c47ef1456
md"""
#### Prior knowledge used to parameterize map components
"""

# ╔═╡ bab1b628-23c0-11ed-2743-6f1fa090a6c4
md"""
From the independence structure mentionned in the problem formulation we have:


*   $\pi(\mu,\phi)=\pi(\mu)\pi(\phi)$, meaning $S_2$ only dependes on $\phi$
*   $\pi(z_{k-2}|z_{k-3},...,z_{0},\phi,\mu)=\pi(z_{k-2}|z_{k-3},\phi,\mu),\,\, k>3$,  meaning $S_k$, only depends on $z_{k-2}$,$z_{k-3}$, $\phi$ and $\mu$


"""

# ╔═╡ bab1b646-23c0-11ed-227f-3fb9a20b4aac
md"""
Complexity of map component can also be deducted from problem formulation:


*   $\pi(\mu)$ being a normal distribution, $S_1$ should be of order 1.
*  $\pi(\phi)$ is non-Gaussian such that $S_2$ should be nonlinear.
*  $\pi(z_{k-2}|z_{k-3},\phi,\mu)$ can be represented by a total order 2 parameterization due to the linear autoregressive model.



"""

# ╔═╡ bab1b664-23c0-11ed-0451-07ce0324b14c
md"""
Hence multi-index sets used for this problem are:


*   $k=1$: 1D expansion of order $\geq$ 1
*   $k=2$: 1D expansion (depending on last component) of high order $>1$
*   $k=3$: 3D expansion of total order 2
*   $k>3$: 4D expansion (depending on first two and last two components) of total order 2


"""

# ╔═╡ bab1b680-23c0-11ed-2152-832d4b0fa3eb
md"""
#### Optimization
"""

# ╔═╡ bab1b68c-23c0-11ed-14e1-61af4bbb8d69
begin
	totalOrder3 = 2
	logPdfTM_sa = zeros(d,Ntest)
	ListCoeffs_sa = zeros(d)
	
	# MultiIndexSet for map S_k, k .>3
	mset_to= CreateTotalOrder(4,totalOrder3)
	
	maxOrder=9 # order for map S_2
	start3 = time_ns()
	@progress "Map component" for dk in 2:d
	    if dk == 2
	        fixed_mset= FixedMultiIndexSet(1,totalOrder3)
	        S = CreateComponent(fixed_mset,opts)
	        Xtrain = reshape(X[dk-1,:], 1, size(X,2))
	        Xtestk = reshape(Xtest[dk-1,:], 1, size(Xtest,2))
		elseif dk == 3
	        fixed_mset= FixedMultiIndexSet(1,maxOrder)
	        S = CreateComponent(fixed_mset,opts)
	        Xtrain = reshape(X[dk-1,:], 1, size(X,2))
	        Xtestk = reshape(Xtest[dk-1,:], 1, size(Xtest,2))
		elseif dk == 4
	        fixed_mset= FixedMultiIndexSet(dk,totalOrder3)
	        S = CreateComponent(fixed_mset,opts)
	        Xtrain = X[1:dk,:]
	        Xtestk = Xtest[1:dk,:]
	    else
	        multis=zeros(Int,Size(mset_to),dk)
	        for s in 1:Size(mset_to)
	            multis_to = mset_to[s]
	            multis[s,1:3]=multis_to[1:3]
	            multis[s,end-1:end]=multis_to[end-1:end]
			end
	        mset = MultiIndexSet(multis)
	        fixed_mset = Fix(mset, true)
	        S = CreateComponent(fixed_mset,opts)
	        Xtrain = X[1:dk,:]
	        Xtestk = Xtest[1:dk,:]
		end
		p = (S,Xtrain)
		
		ListCoeffs_sa[dk-1]=numCoeffs(S)
		fcn = OptimizationFunction(obj, grad=grad_obj)
		prob = OptimizationProblem(fcn, CoeffMap(S), p, gtol=1e-3)
		res = solve(prob, BFGS())

		rho = MvNormal(I(outputDim(S)))    
	    logPdfTM_sa[dk-1,:]=log_cond_pullback_pdf(S,rho,Xtestk)
	end
	end3 = time_ns()
	@info "Took $((end3-start3)*1e-9)s"
end

# ╔═╡ bab29f66-23c0-11ed-3084-19c7dc638ed3
md"""
#### Compute KL divergence error
"""

# ╔═╡ bab29f82-23c0-11ed-1a3e-bf83f75edd2e
md"""
Compute joint KL divergence
"""

# ╔═╡ e3d7c5d6-e1cc-4480-ab53-fe9bc9fd6b29
KL_sa = compute_joint_KL(logPdfSV,logPdfTM_sa)

# ╔═╡ bab29f8e-23c0-11ed-2985-3de6202610b6
md"""
## Compare approximations
"""

# ╔═╡ bab29fa2-23c0-11ed-01f3-f30349beabf6
md"""
### KL divergence
"""

# ╔═╡ bab29fac-23c0-11ed-07aa-1b23ad5890e4
md"""
Compare map approximations
"""

# ╔═╡ 2c143acf-42fd-4716-bc0d-ddfc209fb3e2
begin
	fig3 = Figure()
	ax3 = Axis(fig3[1,1], xlabel="d", ylabel=L"D_{KL}(\pi(\mathbf{x}_t)||S^\sharp \eta)")
	scatter!(ax3,KL_to1, label="Total order 1")
	scatter!(ax3,KL_to2, label="Total order 2")
	scatter!(ax3,KL_sa , label="Sparse MultiIndexSet")
	
	lines!(ax3,KL_to1)
	lines!(ax3,KL_to2)
	lines!(ax3,KL_sa )
	axislegend()
	fig3
end

# ╔═╡ bab29fd4-23c0-11ed-1a95-39be3993e2aa
md"""
Usually increasing map complexity will improve map approximation. However when the number of parameters increases too much compared to the number of samples, computed map overfits the data which lead to worst approximation. This overfitting can be seen in this examples when looking at the total order 2 approximation that rapidly loses accuracy when the dimension increases.

Using sparse multi-index sets help reduces the increase of parameters when the dimension increases leading to better approximation for all dimensions.
"""

# ╔═╡ bab29fe6-23c0-11ed-37ea-41174441ffac
md"""
### Map coefficients
"""

# ╔═╡ bab29ffc-23c0-11ed-2f20-fb9e6d6b2a0a
md"""
To complement observations made above, we visualize the number of parameters (polyniomal coefficients) for each map parameterization.
"""

# ╔═╡ bab2a006-23c0-11ed-327c-7bda74a8d39c
begin
	fig4 = Figure()
	ax4 = Axis(fig4[1,1], xlabel="d", ylabel="# coeffs")
	lines!(ax4,ListCoeffs_to1,label="Total order 1")
	lines!(ax4,ListCoeffs_to2,label="Total order 2")
	lines!(ax4,ListCoeffs_sa,label="Sparse MultiIndexSet")
	scatter!(ax4,ListCoeffs_to1)
	scatter!(ax4,ListCoeffs_to2)
	scatter!(ax4,ListCoeffs_sa)
	axislegend()
	fig4
end

# ╔═╡ bab2c8b8-23c0-11ed-00ac-0fafd36562d9
md"""
We can observe the exponential growth of the number coefficients for the total order 2 approximation. Chosen sparse multi-index sets have a fixed number of parameters which become smaller than the number of parameters of the total order 1 approximation when dimension is 15.
"""

# ╔═╡ bab2c8ce-23c0-11ed-07f7-cf18b49910fc
md"""
Using less parameters helps error scaling with dimension but aslo helps reducing computation time for the optimization and the evaluation the transport maps.
"""

# ╔═╡ Cell order:
# ╠═caac1a76-abf2-4ab0-96f9-07310c79628f
# ╠═8ea66ca7-981c-47d6-8513-099aed421e01
# ╠═baab6a84-23c0-11ed-3f3b-01e3ad086ae7
# ╠═e20f5bba-894b-4c9b-a362-f6afe3b8dc18
# ╟─baab6b24-23c0-11ed-0602-bdb17d65ca9b
# ╟─baab6b7e-23c0-11ed-2af3-c9584eeff5fd
# ╟─baab6b88-23c0-11ed-021d-8d0a20edce42
# ╟─baae2daa-23c0-11ed-2739-85c0abcd5341
# ╟─baae2dd2-23c0-11ed-00f8-6d0035b69fe4
# ╟─baae2eae-23c0-11ed-0fc3-a3c61caa6f21
# ╟─baae2ed4-23c0-11ed-2dab-4dca84f92e8b
# ╟─baae2eea-23c0-11ed-31cc-29625e1c33e0
# ╠═baae2ef4-23c0-11ed-363f-29c8463b2a72
# ╟─baae8e88-23c0-11ed-2ea4-6d63a48fb345
# ╠═baae8ea8-23c0-11ed-3cd5-e514c1a42f57
# ╟─baae9894-23c0-11ed-1f4b-dbaa5d0557d2
# ╠═baae98a8-23c0-11ed-271a-f12d2cad5514
# ╟─baaec170-23c0-11ed-3433-99cd648f9917
# ╠═baaec184-23c0-11ed-1632-6152101b3a8a
# ╟─baaee57e-23c0-11ed-36ca-77d6124cb674
# ╟─baaee59e-23c0-11ed-3509-c9d5d65b9acb
# ╠═baaee5a6-23c0-11ed-10b3-b7ea3dd18770
# ╟─baafaf7c-23c0-11ed-3d4f-417daa259af1
# ╟─baafaf9a-23c0-11ed-0aaa-5158f5057c23
# ╟─baafafa6-23c0-11ed-1329-e3c1800b566c
# ╟─baafafc2-23c0-11ed-0ab0-1f8eac46d349
# ╠═baafafcc-23c0-11ed-2dd3-b56cfcbe5f31
# ╟─baafc3cc-23c0-11ed-04e4-bd608afe616f
# ╟─baafc3e0-23c0-11ed-12ea-1bbacb622ad0
# ╠═baafc3ea-23c0-11ed-26b4-a1adec224fc3
# ╟─baaff23e-23c0-11ed-3de6-c31ab40e44eb
# ╟─baaff252-23c0-11ed-2fb1-dbfaf5dcca3f
# ╟─baaff25c-23c0-11ed-22dd-63ed96e978a8
# ╟─baaff2b6-23c0-11ed-21d0-1fc03d600f47
# ╠═baaff2e8-23c0-11ed-0446-e5ad028d990a
# ╟─bab09ad4-23c0-11ed-0a08-f5ecbececd44
# ╟─bab09aea-23c0-11ed-3f03-3d18da7b26de
# ╠═bab09afe-23c0-11ed-25ce-abbc835d5a74
# ╟─bab0a622-23c0-11ed-30f0-eb74b2ff4a55
# ╠═bab0a634-23c0-11ed-3d4b-2d171a1a4c29
# ╟─bab119ac-23c0-11ed-0500-5f3f9c280dc8
# ╠═bab119ca-23c0-11ed-3a4a-853addf54fa2
# ╟─bab14bfc-23c0-11ed-222a-29a7836ec165
# ╟─bab14c10-23c0-11ed-3464-e95c62236eaa
# ╟─bab14c1a-23c0-11ed-3524-7b984a8f1ab2
# ╠═bab14c30-23c0-11ed-259c-adc821842c14
# ╟─bab1b5e2-23c0-11ed-25c7-f3af826473a8
# ╟─bab1b5f6-23c0-11ed-0542-c595661d7cee
# ╠═8fbcbcee-90ae-490e-a867-e7806e0ff434
# ╟─bab1b600-23c0-11ed-3905-e3dcb5ad077d
# ╟─bab1b60a-23c0-11ed-3713-df191e151f37
# ╟─bab1b620-23c0-11ed-203f-7b4c47ef1456
# ╟─bab1b628-23c0-11ed-2743-6f1fa090a6c4
# ╟─bab1b646-23c0-11ed-227f-3fb9a20b4aac
# ╟─bab1b664-23c0-11ed-0451-07ce0324b14c
# ╟─bab1b680-23c0-11ed-2152-832d4b0fa3eb
# ╠═bab1b68c-23c0-11ed-14e1-61af4bbb8d69
# ╟─bab29f66-23c0-11ed-3084-19c7dc638ed3
# ╟─bab29f82-23c0-11ed-1a3e-bf83f75edd2e
# ╠═e3d7c5d6-e1cc-4480-ab53-fe9bc9fd6b29
# ╟─bab29f8e-23c0-11ed-2985-3de6202610b6
# ╟─bab29fa2-23c0-11ed-01f3-f30349beabf6
# ╟─bab29fac-23c0-11ed-07aa-1b23ad5890e4
# ╠═2c143acf-42fd-4716-bc0d-ddfc209fb3e2
# ╟─bab29fd4-23c0-11ed-1a95-39be3993e2aa
# ╟─bab29fe6-23c0-11ed-37ea-41174441ffac
# ╟─bab29ffc-23c0-11ed-2f20-fb9e6d6b2a0a
# ╠═bab2a006-23c0-11ed-327c-7bda74a8d39c
# ╟─bab2c8b8-23c0-11ed-00ac-0fafd36562d9
# ╟─bab2c8ce-23c0-11ed-07f7-cf18b49910fc
