include("/home/dannys4/installs/MPART/julia/mpart/MParT.jl")
using .MParT, CxxWrap
using Distributions, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using CairoMakie
end
##

num_points = 5000
mu = 2
sigma = 0.5
x = randn(1,num_points)

rv = Normal(mu, sigma)
t = range(-3,6,length=100)
rho_t = pdf.(rv, t)

num_bins = 50
fig = nothing
if make_plot
    fig = Figure()
    ax1 = Axis(fig[1,1], title="Before Optimization")
    hist!(ax1, x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="ReferenceSamples")
    scatter!(ax1, t, rho_t, color=:red, label="Target Density")
    axislegend(ax1)
end
fig

##
A = reshape(Cint[0,1],2,1) # makes vector into matrix, need Cint for typing
mset = MultiIndexSet(A)
fixed_mset = Fix(mset, true)
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
function objective(coeffs,_)
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    pi_of_map_of_x = logpdf.(rv, map_of_x)
    log_det = LogDeterminant(monotoneMap, x)
    -sum(vec(pi_of_map_of_x) + log_det)/num_points
end

u0 = CoeffMap(monotoneMap)
prob = OptimizationProblem(objective, u0, nothing)

# Optimize
println("Starting Coeffs")
println(u0)
println("And error $(objective(u0,nothing))")
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)
println("Final Coeffs")
println(u_final)
println("And error $(objective(u_final,nothing))")

## After optimization Plot
map_of_x = Evaluate(monotoneMap, x)
if make_plot
    ax2 = Axis(fig[2,1], title="After Optimization")
    hist!(ax2, map_of_x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="OptimizedSamples")
    scatter!(ax2, t, rho_t, color=:red, label="Target Density")
    axislegend(ax2)
end
fig
