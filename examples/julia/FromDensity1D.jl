## It is recommended with this script to use VSCode with Julia extension
# Press shift+alt+enter to run the current block

ENV["KOKKOS_NUM_THREADS"] = 4
using MParT, Distributions, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using GLMakie
    Makie.inline!(true)
end

num_points = 5000
mu = 2
sigma = 0.5
x = randn(1,num_points)

rv = Normal(mu, sigma)
t = range(-3,6,length=100)
reference_density_pdf = pdf.(rv, t)

num_bins = 50
if make_plot
    fig1 = Figure()
    ax1 = Axis(fig1[1,1], title="Before Optimization")
    hist!(ax1, vec(x), bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Reference Samples")
    scatter!(ax1, t, reference_density_pdf, color=:red, label="Target Density")
    axislegend(ax1)
    display(fig1)
end

## Create the MultiIndexSet & Initial Map
A = reshape(0:1,2,1)
mset = MultiIndexSet(A)
fixed_mset = Fix(mset, true)
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

## Define KL divergence objective
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

## Optimize the coefficients of the Map
println("Starting Coeffs")
println(u0)
println("And error $(objective(u0,p))")
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)
println("Final Coeffs")
println(u_final)
println("And error $(objective(u_final,p))")

## Plot the results after optimization
map_of_x = Evaluate(monotoneMap, x)
if make_plot
    fig2 = Figure()
    ax2 = Axis(fig2[1,1], title="After Optimization")
    hist!(ax2, vec(map_of_x), bins=num_bins, color=(:blue,0.5), normalization=:pdf, label="Optimized Samples")
    scatter!(ax2, t, reference_density_pdf, color=:red, label="Target Density")
    axislegend(ax2)
    display(fig2)
end