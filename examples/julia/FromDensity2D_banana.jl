### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ cf2ac5b0-23bc-11ed-17e3-85cb03d88328
# ╠═╡ show_logs = false
begin
using Pkg
Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")
Pkg.add(["Distributions", "LinearAlgebra", "Statistics", "Optimization", "OptimizationOptimJL", "CairoMakie", "SpecialFunctions"])
end

# ╔═╡ cf2ac5f8-23bc-11ed-26bc-e1b378ddeec9
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL, CairoMakie

# ╔═╡ cf2ac67a-23bc-11ed-0acb-d56c6fe1e2d4
md"""
# Transport Map from density

The objective of this example is to show how a transport map can be build in MParT when the the unnormalized probability density function of the target density is known.
"""

# ╔═╡ cf2ac6ca-23bc-11ed-323a-03b60ad5cab7
md"""
## Problem description

We consider $T(\mathbf{z};\mathbf{w})$ a monotone triangular transport map parameterized by $\mathbf{w}$ (e.g., polynomial coefficients). This map which is invertible and has an invertible Jacobian for any parameter $\mathbf{w}$, transports samples $\mathbf{z}^i$ from the reference density $\eta$ to samples $T(\mathbf{z}^i;\mathbf{w})$ from the map induced density $\tilde{\pi}_\mathbf{w}(\mathbf{z})$ defined as:
```math
 \tilde{\pi}_\mathbf{w}(\mathbf{z}) = \eta(T^{-1}(\mathbf{z};\mathbf{w}))|\text{det } T^{-1}(\mathbf{z};\mathbf{w})|,
```
where $\text{det } T^{-1}$ is the determinant of the inverse map Jacobian at the point $\mathbf{z}$. We refer to $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ as the *map-induced* density or *pushforward distribution* and will commonly interchange notation for densities and measures to use the notation $\tilde{\pi} = T_{\sharp} \eta$.

The objective of this example is, knowing some unnormalized target density $\bar{\pi}$, find the map $T$ that transport samples drawn from $\eta$ to samples drawn from the target $\pi$.
"""

# ╔═╡ cf2b9f50-23bc-11ed-28e4-c7b1e305078d
md"""
## Imports
First, import MParT and other packages used in this notebook.
"""

# ╔═╡ cf2d7028-23bc-11ed-13e7-2911649c99b4
md"""
## Target density and exact map

In this example we use a 2D target density known as the *banana* density where the unnormalized probability density, samples and the exact transport map are known.

The banana density is defined as:
```math
\pi(x_1,x_2) \propto N_1(x_1)\times N_1(x_2-x_1^2)
```
where $N_1$ is the 1D standard normal density.

The exact transport map that transport the 2D standard normal density to $\pi$ is known as:
```math
{T}^\text{true}(z_1,z_2)=
\begin{bmatrix}
z_1\\
z_2 + z_1^2
\end{bmatrix}
```
"""

# ╔═╡ cf2d70d0-23bc-11ed-32dc-9bfca2278411
md"""
Contours of the target density can be visualized as:
"""

# ╔═╡ cf2d70dc-23bc-11ed-3152-750597e92697
begin

# Unnomalized target density required for objective
function target_logpdf(x) 
  rv1 = Normal()
  rv2 = Normal()
  logpdf1 = logpdf.(rv1, x[1,:])
  logpdf2 = logpdf.(rv2, x[2,:]-x[1,:] .^2)
  logpdf_ret = logpdf1 + logpdf2
  logpdf_ret
end

# Gride for plotting
ngrid=100
x1_t = range(-3,3,ngrid)
x2_t = range(-3,7.5,ngrid)
xx1 = vec(repeat(x1_t', ngrid, 1))
xx2 = vec(repeat(x2_t , 1, ngrid))

xx = collect(hcat(xx1, xx2)')

# For plotting and computing densities

target_pdf_at_grid = exp.(target_logpdf(xx))

fig0 = Figure()
ax0 = Axis(fig0[1,1], xlabel=L"x_1", ylabel=L"x_2")
contour!(ax0, xx1, xx2, target_pdf_at_grid, label="target density")
axislegend()
fig0


end

# ╔═╡ cf2e0f56-23bc-11ed-2461-f33568194c75
md"""
## Map training
### Defining objective function and its gradient
Knowing the closed form of the unnormalized target density $\bar{\pi}$, the objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{z})$ that is a good approximation of the target $\pi$.

In order to characterize this posterior density, one method is to build a monotone triangular transport map $T$ such that the KL divergence $D_{KL}(\eta || T^\sharp \pi)$ is minmized. If $T$ is map parameterized by $\mathbf{w}$, the objective function derived from the discrete KL divergence reads:

```math
J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),
```

where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ to the target density $\pi(\mathbf{z})$. The gradient of this objective function reads

```math
\nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).
```
"""

# ╔═╡ cf2e0fd6-23bc-11ed-368b-03cb6689afda
md"""
The objective function and gradient can be defined using MParT as:
"""

# ╔═╡ cf2e888c-23bc-11ed-1b80-85b45122f274
md"""
### Map parameterization
"""

# ╔═╡ cf2e88aa-23bc-11ed-0c95-9f2969dbda47
md"""
For the parameterization of $T$ we use a total order multivariate expansion of hermite functions. Knowing $T^\text{true}$, any parameterization with total order greater than one will include the true solution of the map finding problem.
"""

# ╔═╡ cf2e88b4-23bc-11ed-20cd-7de15cf84efa
begin
# Set-up first component and initialize map coefficients
map_options = MapOptions()

total_order = 2

# Create dimension 2 triangular map 
transport_map = CreateTriangular(2,2,total_order,map_options)
end

# ╔═╡ cf2eaa56-23bc-11ed-36f0-c9a091908b6b
md"""
### Approximation before optimization

Coefficients of triangular map are set to 0 upon creation.
"""

# ╔═╡ cf2eaa6a-23bc-11ed-3e30-dbabffd7d7c2
begin
# Make reference samples for training
num_points = 10000
z = randn(2,num_points)

# Make reference samples for testing
test_z = randn(2,5000)

# Pushed samples
pushed_samples = Evaluate(transport_map, test_z)

# Before optimization plot
fig1 = Figure()
ax1 = Axis(fig1[1,1])
contour!(ax1, xx1, xx2, target_pdf_at_grid, label="Target density")
scatter!(ax1, pushed_samples[1,:],pushed_samples[2,:], facecolor=(:blue,0.1), label="Pushed samples")
axislegend()
fig1
end

# ╔═╡ cf2e0fe2-23bc-11ed-0523-4d37924a653e
begin
# KL divergence objective
function obj(coeffs,p)
	transport_map, x = p
    SetCoeffs(transport_map, coeffs)
    map_of_x = Evaluate(transport_map, x)
    logpdf= target_logpdf(map_of_x)
    log_det = LogDeterminant(transport_map, x)
    -sum(logpdf + log_det)/num_points
end

# Gradient of unnomalized target density required for gradient objective
function target_grad_logpdf(x)
  grad1 = -x[1,:] + (2*x[1,:].*(x[2,:]-x[1,:] .^2))
  grad2 = (x[1,:] .^2-x[2,:])
  collect(hcat(grad1,grad2)')
end

# Gradient of KL divergence objective
function grad_obj!(g, coeffs,p)
	transport_map, x = p
    SetCoeffs(transport_map, coeffs)
    map_of_x = Evaluate(transport_map, x)
    sens_vecs = target_grad_logpdf(map_of_x)
    grad_logpdf = CoeffGrad(transport_map, x, sens_vecs)
    grad_log_det = LogDeterminantCoeffGrad(transport_map, x)
    g .= -sum(grad_logpdf + grad_log_det, dims=2)/num_points
end


end

# ╔═╡ cf2f0410-23bc-11ed-1841-7bb2fa568c4a
md"""
At initialization, samples are "far" from being distributed according to the banana distribution.
"""

# ╔═╡ cf2f0424-23bc-11ed-0a3e-45f85711e4c1
md"""
Initial objective and coefficients:
"""

# ╔═╡ cf2f042e-23bc-11ed-32ae-dd969d3a46cb
begin
	# Print initial coeffs and objective
	println("==================")
	println("Starting coeffs")
	println(CoeffMap(transport_map))
	p_init = (transport_map, test_z)
	println("Initial objective value: $(obj(CoeffMap(transport_map), p_init)))")
	println("==================")
end

# ╔═╡ cf2f044c-23bc-11ed-3931-d946219ee61c
md"""
### Minimization
"""

# ╔═╡ cf2f0456-23bc-11ed-0353-4da660ca53ff
begin
println("==================")
u0 =  CoeffMap(transport_map)
p = (transport_map, z)
fcn = OptimizationFunction(obj, grad=grad_obj!)
prob = OptimizationProblem(fcn, u0, p, gtol=1e-4)
res = solve(prob, BFGS())

# Print final coeffs and objective
println("Final coeffs:")
println(CoeffMap(transport_map))
println("Final objective value: $(obj(CoeffMap(transport_map), transport_map, test_z))")
println("==================")
end

# ╔═╡ cf2f312e-23bc-11ed-0e8c-35b055880c14
md"""
### Approximation after optimization
"""

# ╔═╡ cf2f3142-23bc-11ed-2b9c-93263102060d
md"""
#### Pushed samples
"""

# ╔═╡ cf2f314c-23bc-11ed-344e-e916c1adb459
begin
# Pushed samples
x = Evaluate(transport_map, test_z)

# After optimization plot
fig2 = Figure()
ax2 = Axis(fig2[1,1])
contour!(ax2, xx1, xx2, target_pdf_at_grid, label="Target Density")
scatter!(ax2, x[1,:],x[2,:], facecolor=(:blue,0.1), label="Pushed samples")
axislegend()
fig2
end

# ╔═╡ cf2f6bee-23bc-11ed-37ed-97979e9d0872
md"""
After optimization, pushed samples $T(z^i)$, $z^i \sim \mathcal{N}(0,I)$ are approximately distributed according to the target $\pi$
"""

# ╔═╡ cf2f6c02-23bc-11ed-02dc-89d5a450876c
md"""
#### Variance diagnostic
"""

# ╔═╡ cf2f6c16-23bc-11ed-30eb-69ed4f279765
md"""
A commonly used accuracy check when facing computation maps from density is the so-called variance diagnostic defined as:

```math
 \epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log \frac{\rho}{T^\sharp \bar{\pi}} \right] 
```
"""

# ╔═╡ cf2f6c48-23bc-11ed-09dd-f90cacc8452d
md"""
This diagnostic is asymptotically equivalent to the minimized KL divergence $D_{KL}(\eta || T^\sharp \pi)$ and should converge to zero when the computed map converge to the true map.
"""

# ╔═╡ cf2f6c5c-23bc-11ed-3e65-5fc5b80e0169
md"""
The variance diagnostic can be computed as follow:
"""

# ╔═╡ cf2f6c66-23bc-11ed-34d0-297a00f938c1
function variance_diagnostic(tri_map,ref,target_logpdf,x)
  ref_logpdf = logpdf(ref, x)
  y = Evaluate(tri_map, x)
  pullback_logpdf = target_logpdf(y) + LogDeterminant(tri_map,x)
  diff = ref_logpdf - pullback_logpdf
  expect = mean(diff)
  var = 0.5*mean((diff.-expect) .^2)
  var
end

# ╔═╡ cf2f9664-23bc-11ed-1360-c1f2c4971860
begin
# Reference distribution
ref_distribution = MvNormal(I(2));

# Compute variance diagnostic
var_diag = variance_diagnostic(transport_map,ref_distribution,target_logpdf,test_z)

# Print final coeffs and objective
println("==================")
println("Variance diagnostic: $var_diag)")
println("==================")


end

# ╔═╡ cf2fd3ae-23bc-11ed-368a-9519360368f2
md"""
#### Pushforward density
"""

# ╔═╡ cf2fd3ce-23bc-11ed-2157-e98c1a3486ff
md"""
We can also plot the contour of the unnormalized density $\bar{\pi}$ and the pushforward approximation $T_\sharp \eta$:
"""

# ╔═╡ cf2fd3d6-23bc-11ed-0450-bf5e4a78364f
begin
# Pushforward definition
function push_forward_pdf(tri_map,ref,x)
  xinv = MParT.Inverse(tri_map, x,x)
  log_det_grad_x_inverse = - LogDeterminant(tri_map, xinv)
  log_pdf = logpdf(ref, xinv)+log_det_grad_x_inverse
  exp.(log_pdf)
end

map_approx_grid = push_forward_pdf(transport_map,ref_distribution,xx)

fig3 = Figure()
ax3 = Axis(fig3[1,1], xlabel=L"x_1", ylabel=L"x_2")
CS1 = contour!(ax3, xx1, xx2, target_pdf_at_grid, label="Unnormalized target")
CS2 = contour!(ax3, xx1, xx2, map_approx_grid,linestyle="--", label="TM approximation")
axislegend()
fig3

end

# ╔═╡ Cell order:
# ╠═cf2ac5b0-23bc-11ed-17e3-85cb03d88328
# ╠═cf2ac5f8-23bc-11ed-26bc-e1b378ddeec9
# ╟─cf2ac67a-23bc-11ed-0acb-d56c6fe1e2d4
# ╟─cf2ac6ca-23bc-11ed-323a-03b60ad5cab7
# ╟─cf2b9f50-23bc-11ed-28e4-c7b1e305078d
# ╟─cf2d7028-23bc-11ed-13e7-2911649c99b4
# ╟─cf2d70d0-23bc-11ed-32dc-9bfca2278411
# ╠═cf2d70dc-23bc-11ed-3152-750597e92697
# ╠═cf2e0f56-23bc-11ed-2461-f33568194c75
# ╠═cf2e0fd6-23bc-11ed-368b-03cb6689afda
# ╠═cf2e0fe2-23bc-11ed-0523-4d37924a653e
# ╠═cf2e888c-23bc-11ed-1b80-85b45122f274
# ╠═cf2e88aa-23bc-11ed-0c95-9f2969dbda47
# ╠═cf2e88b4-23bc-11ed-20cd-7de15cf84efa
# ╠═cf2eaa56-23bc-11ed-36f0-c9a091908b6b
# ╠═cf2eaa6a-23bc-11ed-3e30-dbabffd7d7c2
# ╟─cf2f0410-23bc-11ed-1841-7bb2fa568c4a
# ╟─cf2f0424-23bc-11ed-0a3e-45f85711e4c1
# ╠═cf2f042e-23bc-11ed-32ae-dd969d3a46cb
# ╟─cf2f044c-23bc-11ed-3931-d946219ee61c
# ╠═cf2f0456-23bc-11ed-0353-4da660ca53ff
# ╠═cf2f312e-23bc-11ed-0e8c-35b055880c14
# ╠═cf2f3142-23bc-11ed-2b9c-93263102060d
# ╠═cf2f314c-23bc-11ed-344e-e916c1adb459
# ╟─cf2f6bee-23bc-11ed-37ed-97979e9d0872
# ╟─cf2f6c02-23bc-11ed-02dc-89d5a450876c
# ╟─cf2f6c16-23bc-11ed-30eb-69ed4f279765
# ╟─cf2f6c48-23bc-11ed-09dd-f90cacc8452d
# ╟─cf2f6c5c-23bc-11ed-3e65-5fc5b80e0169
# ╠═cf2f6c66-23bc-11ed-34d0-297a00f938c1
# ╠═cf2f9664-23bc-11ed-1360-c1f2c4971860
# ╟─cf2fd3ae-23bc-11ed-368a-9519360368f2
# ╟─cf2fd3ce-23bc-11ed-2157-e98c1a3486ff
# ╠═cf2fd3d6-23bc-11ed-0450-bf5e4a78364f
