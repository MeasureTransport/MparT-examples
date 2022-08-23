### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 8a023ae0-2328-11ed-30ec-e9524212987a
# ╠═╡ show_logs = false
using Pkg; Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")

# ╔═╡ 8a023b6c-2328-11ed-3021-ab8becdac6f1
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL, GLMakie

# ╔═╡ 8a023bee-2328-11ed-1baa-f3f0a59e51a3
md"""
# Monotone least squares
The objective of this example is to show how to build a transport map to solve monotone regression problems using MParT.
## Problem formulation
One direct use of the monotonicity property given by the transport map approximation to model monotone functions from noisy data. This is called isotonic regression and can be solved in our setting by minimizing the least squares objective function

```math
J(\mathbf{w})= \frac{1}{2} \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - y^i \right)^2,
```

where $S$ is a monotone 1D map with parameters (polynomial coefficients) $\mathbf{w}$ and $y^i$ are noisy observations.   To solve for the map parameters that minimize this objective we will use a gradient-based optimizer.  We therefore need the gradient of the objective with respect to the map paramters.  This is given by

```math
\nabla_\mathbf{w} J(\mathbf{w})= \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - y^i \right)^T\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]
```

The implementation of `S(x)` we're using from MParT, provides tools for both evaluating the map to compute  $S(x^i;\mathbf{w})$ but also evaluating computing the action of  $\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]^T$ on a vector, which is useful for computing the gradient.   Below, these features are leveraged when defining an objective function that we then minimize with the BFGS optimizer implemented in `scipy.minimize`.
"""

# ╔═╡ 8a038ca6-2328-11ed-355b-7368255f6159
md"""
## Imports
First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable **before** importing MParT.
"""

# ╔═╡ 8a038cce-2328-11ed-3a5d-f51ead41fcb5
begin
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
os.environ["KOKKOS_NUM_THREADS"] = "8"

import mpart as mt
print("Kokkos is using", Concurrency(), "threads")
rcParams["figure.dpi"] = 110
end

# ╔═╡ 8a09120c-2328-11ed-104a-53ad86a8f0cc
md"""
## Generate training data

### True model

Here we choose to use the step function $H(x)=\text{sgn}(x-2)+1$ as the reference monotone function. It is worth noting that this function is not strictly monotone and piecewise continuous.
"""

# ╔═╡ 8a09125c-2328-11ed-1bcf-cb528c983345
begin
# variation interval
num_points = 1000
xmin, xmax = 0, 4
x = range(xmin, xmax, num_points)
x = collect(reshape(x, 1, length(x)))

y_true = 2*(x .>2)

fig1 = Figure()
ax1 = Axis(fig1[1,1])
ax1.title = "Reference data"
lines!(ax1, vec(x),vec(y_true))
ax1.xlabel = "x"
ax1.ylabel = "H(x)"
fig1
end

# ╔═╡ 8a0997ae-2328-11ed-37c1-99971ac8a095
md"""
### Training data

Training data $y^i$ in the objective defined above are simulated by pertubating the reference data with a white Gaussian noise with a $0.4$ standard deviation.
"""

# ╔═╡ 8a0997e8-2328-11ed-3545-61a6427e2e52
begin
noisesd = 0.4

y_noise = noisesd*randn(1,num_points) 
y_measured = y_true + y_noise

fig2 = Figure()
ax2 = Axis(fig2[1,1])
ax2.title = "Training data"
lines!(ax2, vec(x),vec(y_measured),color=(:orange,0.4),linestyle=:dash,label="measured data")
ax2.xlabel = "x"
ax2.ylabel = "y"
fig2
end

# ╔═╡ 8a0a0e50-2328-11ed-0733-fb30c290803c
md"""
## Map initialization

We use the previously generated data to train the 1D transport map. In 1D, the map complexity can be set via the list of multi-indices. Here, map complexity can be tuned by setting the `max_order` variable.


"""

# ╔═╡ 8a0a0e96-2328-11ed-3767-db8857e61b2c
md"""
### Multi-index set
"""

# ╔═╡ 8a0a0eaa-2328-11ed-275e-65e5725f2f53
begin
# Define multi-index set
max_order = 5
multis = range(0,max_order,6)
multis = Int.(reshape(multis,length(multis),1))
mset = MultiIndexSet(multis)
fixed_mset = Fix(mset, true)

# Set options and create map object
opts = MapOptions(quadMinSub = 4)

monotone_map = CreateComponent(fixed_mset, opts)
end

# ╔═╡ 8a0a8092-2328-11ed-3ad5-6fea5f632927
md"""
### Plot initial approximation
"""

# ╔═╡ 8a0a80bc-2328-11ed-3e6b-693c2e6eb228
begin
# Before optimization
map_of_x_before = Evaluate(monotone_map, x)
error_before = sum((map_of_x_before - y_measured) .^2)/size(x,1)

# Plot data (before and after apart)
fig3 = Figure()
ax3 = Axis(fig3[1,1])
ax3.title = "Starting map error: $error_before"
lines!(ax3, vec(x),vec(y_true),linestyle=:dashdot,label="true data", alpha=0.8)
lines!(ax3, vec(x),vec(y_measured),linestyle=:dashdot,label="measured data",color=(:orange,0.4))
lines!(ax3, vec(x),vec(map_of_x_before),linestyle=:dashdot,label="initial map output", color=(:red,0.8))
ax3.xlabel = "x"
ax3.ylabel = "y"
axislegend()
fig3


end

# ╔═╡ 8a0b3938-2328-11ed-061a-5f617b7a62f5
md"""
Initial map with coefficients set to zero result in the identity map.
"""

# ╔═╡ 8a0b3960-2328-11ed-1dad-7953d9913c83
md"""
## Transport map training
"""

# ╔═╡ 8a0b3974-2328-11ed-013c-8f4311062487
md"""
### Objective function
"""

# ╔═╡ 8a0b3988-2328-11ed-2c66-b158120f183e
begin
# Least squares objective
function objective(coeffs,p)
	monotone_map, x, y_measured = p
    SetCoeffs(monotone_map, coeffs)
    map_of_x = Evaluate(monotone_map, x)
    sum((map_of_x - y_measured) .^2)/size(x,1)
end

# Gradient of objective
function grad_objective(g, coeffs,p)
	monotone_map, x, y_measured = p
    SetCoeffs(monotone_map, coeffs)
    map_of_x = Evaluate(monotone_map, x)
    g .= 2*sum(CoeffGrad(monotone_map, x, map_of_x - y_measured),dims=2)/size(x,1)
end



end

# ╔═╡ 8a0bd2e4-2328-11ed-162c-432c74eec0ea
md"""
#### Optimization
"""

# ╔═╡ 8a0bd30c-2328-11ed-1689-337878a276ee
begin
# Optimize
u0 =  CoeffMap(monotone_map)
p = (monotone_map, x, y_measured)
fcn = OptimizationFunction(objective, grad=grad_objective)
prob = OptimizationProblem(fcn, u0, p, g_tol = 1e-3)
res = solve(prob, BFGS())

# After optimization
map_of_x_after = Evaluate(monotone_map, x)
error_after = objective(CoeffMap(monotone_map), p)
end

# ╔═╡ 8a0c1cf4-2328-11ed-2bee-014573034bae
md"""
### Plot final approximation
"""

# ╔═╡ 8a0c1d1c-2328-11ed-2aaa-31d7b217bdb3
begin
fig4 = Figure()
ax4 = Axis(fig4[1,1])
ax4.title = "Final map error: $error_after"
lines!(ax4, vec(x),vec(y_true),linestyle=:dashdot,label="true data", alpha=0.8)
lines!(ax4, vec(x),vec(y_measured),linestyle=:dashdot,label="noisy data", color=(:orange,0.4))
lines!(ax4, vec(x),vec(map_of_x_after),linestyle=:dashdot,label="final map output", color=(:red,0.8))
ax4.xlabel = "x"
ax4.ylabel = "y"
axislegend()
fig4

# Unlike the true underlying model, map approximation gives a strict coninuous monotone regression of the noisy data.
end

# ╔═╡ Cell order:
# ╠═8a023ae0-2328-11ed-30ec-e9524212987a
# ╠═8a023b6c-2328-11ed-3021-ab8becdac6f1
# ╠═8a023bee-2328-11ed-1baa-f3f0a59e51a3
# ╠═8a038ca6-2328-11ed-355b-7368255f6159
# ╠═8a038cce-2328-11ed-3a5d-f51ead41fcb5
# ╠═8a09120c-2328-11ed-104a-53ad86a8f0cc
# ╠═8a09125c-2328-11ed-1bcf-cb528c983345
# ╠═8a0997ae-2328-11ed-37c1-99971ac8a095
# ╠═8a0997e8-2328-11ed-3545-61a6427e2e52
# ╠═8a0a0e50-2328-11ed-0733-fb30c290803c
# ╠═8a0a0e96-2328-11ed-3767-db8857e61b2c
# ╠═8a0a0eaa-2328-11ed-275e-65e5725f2f53
# ╠═8a0a8092-2328-11ed-3ad5-6fea5f632927
# ╠═8a0a80bc-2328-11ed-3e6b-693c2e6eb228
# ╠═8a0b3938-2328-11ed-061a-5f617b7a62f5
# ╠═8a0b3960-2328-11ed-1dad-7953d9913c83
# ╠═8a0b3974-2328-11ed-013c-8f4311062487
# ╠═8a0b3988-2328-11ed-2c66-b158120f183e
# ╠═8a0bd2e4-2328-11ed-162c-432c74eec0ea
# ╠═8a0bd30c-2328-11ed-1689-337878a276ee
# ╠═8a0c1cf4-2328-11ed-2bee-014573034bae
# ╠═8a0c1d1c-2328-11ed-2aaa-31d7b217bdb3
