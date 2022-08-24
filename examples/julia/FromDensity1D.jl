### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 926055f2-23bc-11ed-3ddc-71fde36a6c16
# ╠═╡ show_logs = false
using Pkg; Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")

# ╔═╡ 92605642-23bc-11ed-3292-795d6fbe0bf5
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL, GLMakie

# ╔═╡ 926056c4-23bc-11ed-051f-5d11cd3b164d
md"""
# Construct map from density

One way to construct a transport map is from an unnormalized density.
"""

# ╔═╡ 92605714-23bc-11ed-301d-1d8c6476ed1f
md"""
First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the `KOKKOS_NUM_THREADS` environment variable before importing MParT.
"""

# ╔═╡ 92605728-23bc-11ed-27e7-ab4d511f8a92
begin
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

import mpart as mt

print("Kokkos is using", Concurrency(), "threads")
end

# ╔═╡ 9262e772-23bc-11ed-062e-b35629702528
md"""
The target distribution is given by $x\sim\mathcal{N}(2, 0.5)$.
"""

# ╔═╡ 9262e7ae-23bc-11ed-09c3-c7e561e39567
begin
num_points = 5000
mu = 2
sigma = .5
x = randn(1,num_points)
x = collect(reshape(x, 1, length(x)))
end

# ╔═╡ 9262fc26-23bc-11ed-22a5-a16caa625403
md"""
As the reference density we choose the standard normal.
"""

# ╔═╡ 9262fc38-23bc-11ed-22ec-7b2712b1066a
begin
reference_density = norm(loc=mu,scale=sigma)
t = range(-3,6,100)
rho_t = reference_density.pdf(t)
end

# ╔═╡ 92630aae-23bc-11ed-2419-5742d1372248
begin
fig1 = Figure()
ax1 = Axis(fig1[1,1])
hist(vec(x), bins="auto", facecolor=(:blue,0.5), density=True, label="Reference samples")
lines!(ax1, t, rho_t,label="Target density")
scatter!(ax1, t, rho_t,label="Target density")
yticks([])
axislegend()
ax1.title = "Before optimization"
fig1
end

# ╔═╡ 92632f84-23bc-11ed-29e2-8527d45df4f5
md"""
Next we create a multi-index set and create a map. Affine transform should be enough to capture the Gaussian target.
"""

# ╔═╡ 92632fa2-23bc-11ed-1471-e57a86ed9827
begin
multis = array([[0], [1]])  # 
mset = MultiIndexSet(multis)
fixed_mset = Fix(mset, True)
end

# ╔═╡ 92634078-23bc-11ed-2f69-9b0e8f468495
md"""
Now we set the map options (default in this case) and initialize the map
"""

# ╔═╡ 926340a0-23bc-11ed-3884-c5e84ce2fd05
begin
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)
end

# ╔═╡ 92634b22-23bc-11ed-12d4-9f74f3d7dee5
md"""
Next we optimize the coefficients of the map by minimizing the Kullback–Leibler divergence between the target and reference density.
"""

# ╔═╡ 92634b2c-23bc-11ed-0e80-03dfa76148a0
begin
function objective(coeffs,p)
	monotoneMap, x, rv = p
    num_points = size(x,0)
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    pi_of_map_of_x = rv.logpdf(map_of_x)
    log_det = monotoneMap.LogDeterminant(x)
    -sum(pi_of_map_of_x + log_det)/num_points
end

function obj(coeffs,p)
	transport_map, x = p
    SetCoeffs(transport_map, coeffs)
    map_of_x = Evaluate(transport_map, x)
    logpdf= target_logpdf(map_of_x)
    log_det = transport_map.LogDeterminant(x)
    -sum(logpdf + log_det)/num_points
end


end

# ╔═╡ 926399e2-23bc-11ed-05a1-a9aa42f3d858
begin
end

# ╔═╡ 926399f6-23bc-11ed-209f-63816e8c6693
begin
print("Starting coeffs")
print(CoeffMap(monotoneMap))
print("and error: {:.2E}".format(objective(CoeffMap(monotoneMap), monotoneMap, x, reference_density)))
res = minimize(objective, CoeffMap(monotoneMap), args=(monotoneMap, x, reference_density), method="Nelder-Mead")
print("Final coeffs")
print(CoeffMap(monotoneMap))
print("and error: {:.2E}".format(objective(CoeffMap(monotoneMap), monotoneMap, x, reference_density)))
end

# ╔═╡ 9263bcec-23bc-11ed-2199-4df8853fdced
md"""
...and plot the results.
"""

# ╔═╡ 9263bd0c-23bc-11ed-3538-2f1e570cb9b3
begin
map_of_x = Evaluate(monotoneMap, x)
fig, axs = subplots(2,1)
axs[0].hist(vec(x), bins="auto", alpha=0.5, density=True, label="Reference samples")
axs[0].hist(vec(map_of_x), bins="auto", facecolor=(:blue,0.5), density=True, label="Mapped samples")
axs[0].lines!(ax1, t,rho_t,label="Target density")
scatter!(ax1, t,rho_t,label="Target density")
axs[0].axislegend()
axs[0].set_yticks([])

axs[1].hist(vec(x), bins="auto", density=True, label="Reference samples", cumulative=True, histtype="step")
axs[1].hist(vec(map_of_x), bins="auto", density=True, label="Mapped samples", cumulative=True, histtype="step")
axs[1].lines!(ax1, t, reference_density.cdf(t), label="Target density")
scatter!(ax1, t, reference_density.cdf(t), label="Target density")
axs[1].axislegend()
axs[1].set_yticks([])

fig.supax1.title = "After optimization"

fig1

# ╔═╡ Cell order:
# ╠═926055f2-23bc-11ed-3ddc-71fde36a6c16
# ╠═92605642-23bc-11ed-3292-795d6fbe0bf5
# ╠═926056c4-23bc-11ed-051f-5d11cd3b164d
# ╠═92605714-23bc-11ed-301d-1d8c6476ed1f
# ╠═92605728-23bc-11ed-27e7-ab4d511f8a92
# ╠═9262e772-23bc-11ed-062e-b35629702528
# ╠═9262e7ae-23bc-11ed-09c3-c7e561e39567
# ╠═9262fc26-23bc-11ed-22a5-a16caa625403
# ╠═9262fc38-23bc-11ed-22ec-7b2712b1066a
# ╠═92630aae-23bc-11ed-2419-5742d1372248
# ╠═92632f84-23bc-11ed-29e2-8527d45df4f5
# ╠═92632fa2-23bc-11ed-1471-e57a86ed9827
# ╠═92634078-23bc-11ed-2f69-9b0e8f468495
# ╠═926340a0-23bc-11ed-3884-c5e84ce2fd05
# ╠═92634b22-23bc-11ed-12d4-9f74f3d7dee5
# ╠═92634b2c-23bc-11ed-0e80-03dfa76148a0
# ╠═926399e2-23bc-11ed-05a1-a9aa42f3d858
# ╠═926399f6-23bc-11ed-209f-63816e8c6693
# ╠═9263bcec-23bc-11ed-2199-4df8853fdced
# ╠═9263bd0c-23bc-11ed-3538-2f1e570cb9b3
