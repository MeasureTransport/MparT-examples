## It is recommended with this script to use VSCode with Julia extension
# Press shift+alt+enter to run the current block

ENV["KOKKOS_NUM_THREADS"] = 4
using MParT, Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using GLMakie
end

## Geometry
num_points = 1000
xmin, xmax = 0,4
x = collect(reshape(range(xmin, xmax, length=num_points), 1, num_points))

# Measurements
noisesd = 0.4

# Notes: data might not be monotone bc of the noise
# but we assume the true underlying function is monotone
y_true = 2*(x .> 2)
y_noise = noisesd*randn(1,num_points)
y_measured = y_true + y_noise

# Create MultiIndexSet
multis = reshape(0:5, 6, 1)
mset = MultiIndexSet(multis)
fixed_mset = Fix(mset, true)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

## Define Least Squares objective
function objective(coeffs,p)
    monotoneMap, x, y_measured = p
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    norm(map_of_x - y_measured)^2/size(x,2)
end

## Optimization Setup
map_of_x_before = Evaluate(monotoneMap, x)
u0 = CoeffMap(monotoneMap)
p = (monotoneMap, x, y_measured)
error_before = objective(u0,p)
fcn = OptimizationFunction(objective)
prob = OptimizationProblem(fcn, u0, p)

# Plot Before Optimization
if make_plot
    fig1 = Figure()
    ax11 = Axis(fig1[1,1], title = "Starting map error: $error_before")
    scatter!(ax11, vec(x), vec(y_true), alpha=0.8, label="true data")
    scatter!(ax11, vec(x), vec(y_measured), alpha=0.8, label="measured data")
    scatter!(ax11, vec(x), vec(map_of_x_before), alpha=0.8, label="initial map output")
    axislegend(ax11, position=:rb)
    display(fig1)
end

## Optimize
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)

## After Optimization
map_of_x_after = Evaluate(monotoneMap, x)
error_after = objective(u_final, p)
fig1 = fig2 = nothing
if make_plot
    fig2 = Figure()
    ax12 = Axis(fig2[1,1], title = "Starting Map Error: $error_before")
    scatter!(ax12, vec(x), vec(y_measured), alpha=0.8, label="measured data")
    scatter!(ax12, vec(x), vec(y_true), alpha=0.8, label="true data")
    scatter!(ax12, vec(x), vec(map_of_x_before), alpha=0.8, label="initial map output")
    axislegend(ax12, position=:rb)
    ax22 = Axis(fig2[2,1], title = "Final Map Error: $error_after")
    scatter!(ax22, vec(x), vec(y_measured), alpha=0.8, label="measured data")
    scatter!(ax22, vec(x), vec(y_true), alpha=0.8, label="true data")
    scatter!(ax22, vec(x), vec(map_of_x_after), alpha=0.8, label="final map output")
    axislegend(ax22, position=:rb)
    display(fig2)
end
