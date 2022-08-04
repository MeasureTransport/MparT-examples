using MParT, CxxWrap
using Distributions, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using GLMakie
end
##

num_points = 5000
mu = 2
sigma = 0.5
x = randn(1,num_points)

rv = Normal(mu, sigma)
t = range(-3,6,length=100)
reference_density_pdf = pdf.(rv, t)

num_bins = 50
if make_plot
    fig = Figure()
    ax1 = Axis(fig[1,1], title="Before Optimization")
    hist!(ax1, x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Reference Samples")
    scatter!(ax1, t, reference_density_pdf, color=:red, label="Target Density")
    axislegend(ax1)
    display(fig)
end

##
A = reshape(0:1,2,1) # The ;; makes a single row matrix then transpose for single column
mset = MultiIndexSet(A)
fixed_mset = Fix(mset, true)
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
function objective(coeffs,p)
    monotoneMap, x = p
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    pi_of_map_of_x = logpdf.(rv, map_of_x)
    log_det = LogDeterminant(monotoneMap, x)
    -sum(vec(pi_of_map_of_x) + log_det)/num_points
end

u0 = CoeffMap(monotoneMap)
p = monotoneMap, x
prob = OptimizationProblem(objective, u0, p)

# Optimize
println("Starting Coeffs")
println(u0)
println("And error $(objective(u0,p))")
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)
println("Final Coeffs")
println(u_final)
println("And error $(objective(u_final,p))")

## After optimization Plot
map_of_x = Evaluate(monotoneMap, x)
if make_plot
    fig = Figure()
    ax2 = Axis(fig[1,1], title="After Optimization")
    hist!(ax2, map_of_x[:], bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Optimized Samples")
    scatter!(ax2, t, reference_density_pdf, color=:red, label="Target Density")
    axislegend(ax2)
    display(fig)
end
