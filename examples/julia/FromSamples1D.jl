using MParT, CxxWrap
using Distributions, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using GLMakie
end

raw"""
    To make skewed and/or non-Gaussian tailed test distributions
    skew \in R, skew > 0 leads to positive (right tilted) skew, skew < 0 leads to negative (left tilted) skew
    tail > 0, tail < 1 leads to light tails, tail > 1 leads to heavy tails.
    skew = 0, tail = 1 leads to affine function x = loc + scale*z
    See for more info: Jones, M. Chris, and Arthur Pewsey. "Sinh-arcsinh distributions." Biometrika 96.4 (2009): 761-780.
"""
function sinharcsinh(z;loc,scale,skew,tail)
    f0 = sinh(tail*asinh(2))
    f = (2/f0)*sinh(tail*(asinh(z) + skew))
    loc + scale*f
end

## Make target samples
num_points = 1000
z = randn(num_points)
x = reshape(sinharcsinh.(z, loc=-1, scale=1, skew=.5, tail=1), 1, num_points)

# For plotting and computing reference density
rv = Normal()
t = range(-3,3,length=100)
rho_t = pdf.(rv, t)

# Before optimization
num_bins = 50
if make_plot
    fig = Figure()
    ax1 = Axis(fig[1,1], title="Before Optimization")
    hist!(ax1, x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Target samples")
    scatter!(ax1, t, rho_t, label="Reference density")
    axislegend(ax1)
    display(fig)
end

## Create multi-index set:
multis = reshape(0:5, 6, 1)
mset = MultiIndexSet(multis)
fixed_mset = Fix(mset, true)

# Set MapOptions and make map
opts = MapOptions(basisType = BasisTypes.HermiteFunctions)
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
function objective(coeffs,_)
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    ref_logpdf_of_map_of_x = logpdf.(rv, map_of_x)
    log_det = LogDeterminant(monotoneMap, x)
    return -sum(ref_logpdf_of_map_of_x[:] + log_det)/num_points
end

u0 = CoeffMap(monotoneMap)
prob = OptimizationProblem(objective, u0, nothing)

## Optimize
println("Starting coeffs")
println(u0)
println("and error: $(objective(u0, nothing))")
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)
println("Final coeffs")
println(u_final)
println("and error: $(objective(u_final, nothing))")

## After optimization plot
map_of_x = Evaluate(monotoneMap, x)
if make_plot
    fig = Figure()
    ax2 = Axis(fig[1,1], title="After Optimization")
    hist!(ax2, map_of_x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Normalized Samples")
    scatter!(ax2, t, rho_t, color=:red, label="Reference Density")
    axislegend(ax2)
    display(fig)
end