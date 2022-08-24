### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 954d8538-23bb-11ed-145b-0fad4e80d906
# ╠═╡ show_logs = false
begin
using Pkg
Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")
Pkg.add(["Distributions", "LinearAlgebra", "Statistics", "Optimization", "OptimizationOptimJL", "CairoMakie", "SpecialFunctions"])
end

# ╔═╡ 954d857e-23bb-11ed-379f-bf59db6edf0e
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL, CairoMakie, SpecialFunctions

# ╔═╡ 954d8600-23bb-11ed-3afb-79985341ba63
md"""
# Characterization of Bayesian posterior density
"""

# ╔═╡ 954d8650-23bb-11ed-08e0-35522a6e6112
md"""
The objective of this example is to demonstrate how transport maps can be used to represent posterior distribution of Bayesian inference problems.
"""

# ╔═╡ 954d8658-23bb-11ed-0104-fd19bf60c5af
md"""
## Imports
First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.
"""

# ╔═╡ 955024a2-23bb-11ed-00bf-4b1f8d006e36
md"""
## Problem formulation
"""

# ╔═╡ 955024dc-23bb-11ed-1d86-3b6cc63b7ef9
md"""
### Bayesian inference
"""

# ╔═╡ 955024f0-23bb-11ed-185c-b3f673bbe699
md"""
A way construct a transport map is from an unnormalized density. One situation where we know the probality density function up to a normalization constant is when modeling inverse problems with Bayesian inference.

For an inverse problem, the objective is to characterize the value of some parameters $\boldsymbol{\theta}$ of a given system, knowing some the value of some noisy observations $\mathbf{y}$.

With Bayesian inference, the characterization of parameters $\boldsymbol{\theta}$ is done via a *posterior* density $\pi(\boldsymbol{\theta}|\mathbf{y})$. This density characterizes the distribution of the parameters knowing the value of the observations.

Using Bayes' rule, the posterior can decomposed into the product of two probability densities:

1.   The prior density $\pi(\boldsymbol{\theta})$ which is used to enforce any *a priori* knowledge about the parameters.
2.   The likelihood function $\pi(\mathbf{y}|\boldsymbol{\theta})$. This quantity can be seen as a function of $\boldsymbol{\theta}$ and gives the likelihood that the considered system produced the observation $\mathbf{y}$ for a fixed value of $\boldsymbol{\theta}$. When the model that describes the system is known in closed form, the likelihood function is also knwon in closed form.

Hence, the posterior density reads:

```math
\pi(\boldsymbol{\theta}|\mathbf{y}) = \frac{1}{c} \pi(\mathbf{y}|\boldsymbol{\theta}) \pi(\boldsymbol{\theta})
```

where $c = \int \pi(\mathbf{y}|\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\theta$ is an normalizing constant that ensures that the product of the two quantities is a proper density.  Typically, the integral in this definition cannot be evaluated and $c$ is assumed to be unknown.
"""

# ╔═╡ 9550257c-23bb-11ed-0959-2becda31f4a3
md"""
The objective of this examples is, from the knowledge of $\pi(\mathbf{y}|\boldsymbol{\theta})\pi(\boldsymbol{\theta})$ build a transport map that transports samples from the reference $\eta$ to samples from posterior $\pi(\boldsymbol{\theta}|\mathbf{y})$.
"""

# ╔═╡ 95502586-23bb-11ed-2dcd-7329ebb64ff0
md"""
### Application with the Biochemical Oxygen Demand (BOD) model from [[Sullivan et al., 2010]](https://or.water.usgs.gov/proj/keno_reach/download/chemgeo_bod_final.pdf)

#### Definition

To illustrate the process describe above, we consider the BOD inverse problem described in [[Marzouk et al., 2016]](https://arxiv.org/pdf/1602.05023.pdf).   The goal is to estimate $2$ coefficients in a time-dependent model of oxygen demand, which is used as an indication of biological activity in a water sample.

The time dependent forward model is defined as

```math
\mathcal{B}(t) = A(1-\exp(-Bt))+\mathcal{E},
```

where

```math
\begin{aligned}
\mathcal{E} & \sim \mathcal{N}(0,1e-3)\\
A & = \left[0.4 + 0.4\left(1 + \text{erf}\left(\frac{\theta_1}{\sqrt{2}} \right)\right) \right]\\
B & = \left[0.01 + 0.15\left(1 + \text{erf}\left(\frac{\theta_2}{\sqrt{2}} \right)\right) \right]
\end{aligned}
```

The objective is to characterize the posterior density of parameters $\boldsymbol{\theta}=(\theta_1,\theta_2)$ knowing observation of the system at time $t=\left\{1,2,3,4,5 \right\}$ i.e. $\mathbf{y} = (y_1,y_2,y_3,y_4,y_5) = (\mathcal{B}(1),\mathcal{B}(2),\mathcal{B}(3),\mathcal{B}(4),\mathcal{B}(5))$.
"""

# ╔═╡ 955025ea-23bb-11ed-2bf3-038f89b70ae3
md"""
Definition of the forward model and gradient with respect to $\mathbf{\theta}$:
"""

# ╔═╡ 955025f4-23bb-11ed-2332-7b5583ceb8b4
begin
function forward_model(p1,p2,t)
  A = 0.4+0.4*(1+erf(p1/sqrt(2)))
  B = 0.01+0.15*(1+erf(p2/sqrt(2)))
  out = A*(1-exp.(-B*t))
  out
end

function grad_x_forward_model(p1,p2,t)
  A = 0.4+0.4*(1+erf(p1/sqrt(2)))
  B = 0.01+0.15*(1+erf(p2/sqrt(2)))
  dAdx1 = 0.31954*exp(-0.5*p1 .^2)
  dBdx2 = 0.119683*exp(-0.5*p2 .^2)
  dOutdx1 = dAdx1*(1-exp(-B*t))
  dOutdx2 = t*A*dBdx2*exp(-t*B)
  vcat(dOutdx1,dOutdx2)
end


end

# ╔═╡ 9550755e-23bb-11ed-0a8a-e9249a94e917
md"""
One simulation of the forward model:
"""

# ╔═╡ 9550757c-23bb-11ed-163d-0bc0ea8f5239
begin
t = range(0,10, 100)
fig0 = Figure()
ax0 = Axis(fig0[1,1])
lines!(ax0, t, forward_model.(1, 1, t))
scatter!(ax0, t, forward_model.(1, 1, t));
ax0.xlabel = "t"
ax0.ylabel = "BOD";
fig0
end

# ╔═╡ 955090cc-23bb-11ed-3513-57f02bb737ea
md"""
For this problem, as noise $\mathcal{E}$ is Gaussian and additive, the likelihood function $\pi(\mathbf{y}|\boldsymbol{\theta})$, can be decomposed for each time step as:
```math
\pi(\mathbf{y}|\boldsymbol{\theta}) = \prod_{t}^{5} \pi(y_t|\boldsymbol{\theta}),
```
where
```math
\pi(\mathbf{y}_t|\boldsymbol{\theta})=\frac{1}{\sqrt{0.002.\pi}}\exp \left(-\frac{1}{0.002} \left(y_t - \mathcal{B}(t)\right)^2 \right), t \in \{1,...,5\}.
```
"""

# ╔═╡ 9550911a-23bb-11ed-1e50-c7956d97df01
md"""
Likelihood function and its gradient with respect to parameters:
"""

# ╔═╡ 95509124-23bb-11ed-0191-cbc0b0146cc7
begin
function log_likelihood(std_noise,t,yobs,p1,p2)
  y = forward_model.(p1,p2,t)
  log_lkl = -log.(sqrt(2*pi)*std_noise).-0.5*((y .- yobs)/std_noise) .^2
  log_lkl
end

function grad_x_log_likelihood(std_noise,t,yobs,p1,p2)
  y = forward_model.(p1,p2,t)
  dydx = grad_x_forward_model.(p1,p2,t)
  grad_x_lkl = reduce(vcat,(-1/std_noise .^2)*(y .- yobs).*dydx)
  grad_x_lkl
end


end

# ╔═╡ 9550cb82-23bb-11ed-22db-f925cb8b7d63
md"""
We can then define the unnormalized posterior and its gradient with respect to parameters:
"""

# ╔═╡ 9550cb94-23bb-11ed-3489-3f3450a778a0
begin
function log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,p1,p2)
  log_prior1 = -log(sqrt(2*pi)*std_prior1).-0.5*(p1/std_prior1) .^2
  log_prior2 = -log(sqrt(2*pi)*std_prior2).-0.5*(p2/std_prior2) .^2
  log_posterior = log_prior1+log_prior2
  for (k,t) in enumerate(list_t)
    log_posterior += log_likelihood(std_noise,t,list_yobs[k],p1,p2)
  end
  log_posterior
end

function grad_x_log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,p1,p2)
  grad_x1_prior = -(1/std_prior1 .^2)*(p1)
  grad_x2_prior = -(1/std_prior2 .^2)*(p2)
  grad_x_prior = vcat(grad_x1_prior,grad_x2_prior)
  grad_x_log_posterior = grad_x_prior
  @info "" size(grad_x_log_posterior)
  for (k,t) in enumerate(list_t)
    grad_x_log_posterior += grad_x_log_likelihood(std_noise,t,list_yobs[k],p1,p2)
  end
  grad_x_log_posterior
end


end

# ╔═╡ 955137de-23bb-11ed-08be-77012cac7bd9
md"""
#### Observations

We consider the following realization of observation $\mathbf{y}$:
"""

# ╔═╡ 95513804-23bb-11ed-153a-4f9809a0bc9d
begin
list_t = [1,2,3,4,5]
list_yobs = [0.18,0.32,0.42,0.49,0.54]

std_noise = sqrt(1e-3)
std_prior1 = 1
std_prior2 = 1
end

# ╔═╡ 95515618-23bb-11ed-0b45-cfaf024382b0
md"""
#### Visualization of the **unnormalized** posterior density
"""

# ╔═╡ 9551562c-23bb-11ed-2be1-93c303957fd2
begin
Ngrid = 100
x = range(-0.5, 1.25, Ngrid)
y = range(-0.5, 2.5, Ngrid)
X = repeat(x', Ngrid, 1)
Y = repeat(y, 1, Ngrid)

Z = log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,vec(X),vec(Y))
Z = exp(reshape(Z, Ngrid,Ngrid))


fig1 = Figure()
ax1 = Axis(fig1[1,1], xlabel=L"\theta_1", ylabel=L"\theta_2")
# CS = contour!(ax1, X, Y, Z)
fig1
end

# ╔═╡ 9551a47e-23bb-11ed-0695-979c1f145d52
md"""
Target density for the map from density is non-Gaussian which mean that a non linear map will be required to achieve good approximation.
"""

# ╔═╡ 9551a492-23bb-11ed-20d2-3f8674c9579c
md"""
## Map computation
"""

# ╔═╡ 9551a4a6-23bb-11ed-0117-6bf451a9bc15
md"""
After the definition of the log-posterior and gradient, the construction of the desired map $T$ to characterize the posterior density result in a "classic" map from unnomarlized computation.
"""

# ╔═╡ 9551a4b0-23bb-11ed-31a5-051dcf2c9d0a
md"""
### Definition of the objective function:

Knowing the closed form of unnormalized posterior $\bar{\pi}(\boldsymbol{\theta} |\mathbf{y})= \pi(\mathbf{y}|\boldsymbol{\theta})\pi(\boldsymbol{\theta})$, the objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ that is a good approximation to the posterior $\pi(\boldsymbol{\theta} |\mathbf{y})$.

In order to characterize this posterior density, one method is to build a transport map.

For the map from unnormalized density estimation, the objective function on parameter $\mathbf{w}$ reads:

```math
J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),
```

where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ to the target density $\pi(\mathbf{x})$, which will be the the posterior density.  The gradient of this objective function reads:

```math
\nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).
```
"""

# ╔═╡ 9551a528-23bb-11ed-0b28-fdca4891dd0e
begin
function grad_x_log_target(x)
  out = grad_x_log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,x[1,:],x[2,:])
  out
end

function log_target(x)
  out = log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,x[1,:],x[2,:])
  out
end

function obj(coeffs,p)
	tri_map,x = p
    num_points = size(x,1)
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    rho_of_map_of_x = log_target(map_of_x)
    log_det = LogDeterminant(tri_map, x)
    -sum(rho_of_map_of_x + log_det)/num_points
end

function grad_obj!(g,coeffs,p)
	tri_map, x = p
    num_points = size(x,1)
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    sensi = grad_x_log_target(map_of_x)
	sensi = collect(reshape(sensi, length(sensi), 1))
    grad_rho_of_map_of_x = CoeffGrad(tri_map, x, sensi)
    grad_log_det = LogDeterminantCoeffGrad(tri_map, x)
    g .= -sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points
end


end

# ╔═╡ 955232d6-23bb-11ed-04b0-3956d0902407
begin
#Draw reference samples to define objective
N=10000
Xtrain = randn(2,N)
end

# ╔═╡ 95524424-23bb-11ed-388e-5555245ec846
md"""
#### Map parametrization
"""

# ╔═╡ 95524442-23bb-11ed-210b-9b4243de0dc0
md"""
We use the MParT function `CreateTriangular` to directly create a transport map object of dimension with given total order parameterization.
"""

# ╔═╡ eaeb3764-5d9f-494b-8cad-9031dcb1495a
begin
	# Create transport map
	opts = MapOptions()
	total_order = 3
	tri_map = CreateTriangular(2,2,total_order,opts)
end

# ╔═╡ 95524460-23bb-11ed-0ecc-c5a9fb22102c
md"""
#### Optimization
"""

# ╔═╡ 95524474-23bb-11ed-2396-25b70f66a9aa
begin
# u0 =  CoeffMap(tri_map)
# p = (tri_map, Xtrain)
# fcn = OptimizationFunction(obj, grad=grad_obj!)
# prob = OptimizationProblem(fcn, u0, p, gtol=1e-2)
# res = solve(prob, BFGS())
end

# ╔═╡ 95524e88-23bb-11ed-0a1f-91e9b1718e3c
md"""
## Accuracy checks
"""

# ╔═╡ 95524e92-23bb-11ed-305e-eb877442f4d9
md"""
### Comparing density contours
"""

# ╔═╡ 95524e9c-23bb-11ed-17f6-b3eacfd2065d
md"""
Comparison between contours of the posterior $\pi(\boldsymbol{\theta}|\mathbf{y})$ and conoturs of pushforward density $T_\sharp \eta$.
"""

# ╔═╡ 95524eb2-23bb-11ed-05b5-01cf63f91985
begin
# Pushforward distribution
function push_forward_pdf(tri_map,ref,x)
  xinv = MParT.Inverse(tri_map,x,x)
  log_det_grad_x_inverse = - LogDeterminant(tri_map, xinv)
  log_pdf = logpdf(ref, xinv)+log_det_grad_x_inverse
  exp(log_pdf)
end
end

# ╔═╡ 95526dd2-23bb-11ed-0084-2f374a59fcf3
begin
# Reference distribution
ref = MvNormal(I(2))

xx_eval = vcat(vec(X),vec(Y))
xx_eval = collect(reshape(xx_eval, length(xx_eval), 1))
Z2 = push_forward_pdf(tri_map,ref,xx_eval)
Z2 = reshape(Z2, Ngrid,Ngrid)

fig2 = Figure()
ax2 = Axis(fig2[1,1], xlabel="\theta_1", ylabel="\theta_2")
contour!(ax2, X, Y, Z, label="Unnormalized posterior")
contour!(ax2, X, Y, Z2,linestyles="dashed", label="TM approximation")
axislegend()
fig0


end

# ╔═╡ 9552c4c6-23bb-11ed-1d37-575900099d7c
md"""
### Variance diagnostic

A commonly used accuracy check when facing computation maps from density is the so-called variance diagnostic defined as:

```math
 \epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log \frac{\rho}{T^\sharp \bar{\pi}} \right]
```

This diagnostic is asymptotically equivalent to the minimized KL divergence $D_{KL}(\eta || T^\sharp \pi)$ and should converge to zero when the computed map converge to the theoritical true map.
"""

# ╔═╡ 9552c50c-23bb-11ed-064d-8361c8366bad
begin
function variance_diagnostic(tri_map,ref,target_logpdf,x)
  ref_logpdf = ref.logpdf(x.T)
  y = Evaluate(tri_map, x)
  pullback_logpdf = target_logpdf(y) + tri_map.LogDeterminant(x)
  diff = ref_logpdf - pullback_logpdf
  expect = mean(diff)
  var = 0.5*mean((diff-expect) .^2)
  var
end
end

# ╔═╡ 9552ed34-23bb-11ed-2861-df67d16017ce
begin
# Reference distribution
ref_distribution = MvNormal(I(2));

test_z = randn(2,10000)
# Compute variance diagnostic
var_diag = variance_diagnostic(tri_map,ref,log_target,test_z)

# Print final coeffs and objective
print("==================")
print("Variance diagnostic: {:.2E}".format(var_diag))
print("==================")
end

# ╔═╡ 9553252c-23bb-11ed-37b4-958268509e7c
md"""
### Drawing samples from approximate posterior

Once the transport map from reference to unnormalized posterior is estimated it can be used to sample from the posterior to characterize the Bayesian inference solution.
"""

# ╔═╡ 9553254c-23bb-11ed-1423-9364ea2e42ce
begin
Znew = random.randn(2,5000)
colors = arctan2(Znew[1,:],Znew[0,:])

Xpost = Evaluate(tri_map, Znew)

fig,axs = subplots(ncols=2,figsize=(12,5))
axs[0].scatter(Xpost[0,:],Xpost[1,:], c=colors, alpha=0.2)
axs[0].set_aspect("equal", "box")
axs[0].set_ax0.xlabel = r"$\theta_1$"
axs[0].set_ax0.ylabel = r"$\theta_2$"
axs[0].set_ax0.title = "Approximate Posterior Samples"

axs[1].scatter(Znew[0,:],Znew[1,:], c=colors, alpha=0.2)
axs[1].set_aspect("equal", "box")
axs[1].set_ax0.xlabel = r"$z_1$"
axs[1].set_ax0.ylabel = r"$z_2$"
axs[1].set_ax0.title = "Reference Samples"


fig0
end

# ╔═╡ 95539ba0-23bb-11ed-3d5c-57908b19833e
md"""
Samples can then be used to compute quantity of interest with respect to parameters $\boldsymbol{\theta}$. For example the sample mean:
"""

# ╔═╡ 95539bc6-23bb-11ed-2405-b77739d691a7
begin
X_mean = mean(Xpost,1)
print("Mean a posteriori: $X_mean")
end

# ╔═╡ 9553a5aa-23bb-11ed-164b-e56251b94523
begin
end

# ╔═╡ 9553a5b2-23bb-11ed-3dac-bf9f97829c5b
begin
ax[0].hist(Xpost[0,:], 50, alpha=0.5, density=True)
ax[0].set_ax0.xlabel = r"$\theta_1$"
ax[0].set_ax0.ylabel = r"$\tilde{\pi}(\theta_1)$"
ax[1].hist(Xpost[1,:], 50, alpha=0.5, density=True)
ax[1].set_ax0.xlabel = r"$\theta_2$"
ax[1].set_ax0.ylabel = r"$\tilde{\pi}(\theta_2)$"
fig0

# ╔═╡ Cell order:
# ╠═954d8538-23bb-11ed-145b-0fad4e80d906
# ╠═954d857e-23bb-11ed-379f-bf59db6edf0e
# ╠═954d8600-23bb-11ed-3afb-79985341ba63
# ╠═954d8650-23bb-11ed-08e0-35522a6e6112
# ╠═954d8658-23bb-11ed-0104-fd19bf60c5af
# ╠═955024a2-23bb-11ed-00bf-4b1f8d006e36
# ╠═955024dc-23bb-11ed-1d86-3b6cc63b7ef9
# ╠═955024f0-23bb-11ed-185c-b3f673bbe699
# ╠═9550257c-23bb-11ed-0959-2becda31f4a3
# ╠═95502586-23bb-11ed-2dcd-7329ebb64ff0
# ╠═955025ea-23bb-11ed-2bf3-038f89b70ae3
# ╠═955025f4-23bb-11ed-2332-7b5583ceb8b4
# ╠═9550755e-23bb-11ed-0a8a-e9249a94e917
# ╠═9550757c-23bb-11ed-163d-0bc0ea8f5239
# ╠═955090cc-23bb-11ed-3513-57f02bb737ea
# ╠═9550911a-23bb-11ed-1e50-c7956d97df01
# ╠═95509124-23bb-11ed-0191-cbc0b0146cc7
# ╠═9550cb82-23bb-11ed-22db-f925cb8b7d63
# ╠═9550cb94-23bb-11ed-3489-3f3450a778a0
# ╠═955137de-23bb-11ed-08be-77012cac7bd9
# ╠═95513804-23bb-11ed-153a-4f9809a0bc9d
# ╠═95515618-23bb-11ed-0b45-cfaf024382b0
# ╠═9551562c-23bb-11ed-2be1-93c303957fd2
# ╠═9551a47e-23bb-11ed-0695-979c1f145d52
# ╠═9551a492-23bb-11ed-20d2-3f8674c9579c
# ╠═9551a4a6-23bb-11ed-0117-6bf451a9bc15
# ╠═9551a4b0-23bb-11ed-31a5-051dcf2c9d0a
# ╠═9551a528-23bb-11ed-0b28-fdca4891dd0e
# ╠═955232d6-23bb-11ed-04b0-3956d0902407
# ╠═95524424-23bb-11ed-388e-5555245ec846
# ╠═95524442-23bb-11ed-210b-9b4243de0dc0
# ╠═eaeb3764-5d9f-494b-8cad-9031dcb1495a
# ╠═95524460-23bb-11ed-0ecc-c5a9fb22102c
# ╠═95524474-23bb-11ed-2396-25b70f66a9aa
# ╠═95524e88-23bb-11ed-0a1f-91e9b1718e3c
# ╠═95524e92-23bb-11ed-305e-eb877442f4d9
# ╠═95524e9c-23bb-11ed-17f6-b3eacfd2065d
# ╠═95524eb2-23bb-11ed-05b5-01cf63f91985
# ╠═95526dd2-23bb-11ed-0084-2f374a59fcf3
# ╠═9552c4c6-23bb-11ed-1d37-575900099d7c
# ╠═9552c50c-23bb-11ed-064d-8361c8366bad
# ╠═9552ed34-23bb-11ed-2861-df67d16017ce
# ╠═9553252c-23bb-11ed-37b4-958268509e7c
# ╠═9553254c-23bb-11ed-1423-9364ea2e42ce
# ╠═95539ba0-23bb-11ed-3d5c-57908b19833e
# ╠═95539bc6-23bb-11ed-2405-b77739d691a7
# ╠═9553a5aa-23bb-11ed-164b-e56251b94523
# ╠═9553a5b2-23bb-11ed-3dac-bf9f97829c5b
