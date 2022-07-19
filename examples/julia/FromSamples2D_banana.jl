using MParT, CxxWrap
using Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using GLMakie
end
##
num_points = 1000
z = randn(2,num_points)
x1 = z[1,:]
x2 = z[2,:] + z[1,:].^2
x = collect([x1 x2]')

test_n_pts = 10_000
test_z = randn(2,test_n_pts)
test_x1 = test_z[1,:]
test_x2 = test_z[2,:] + test_z[1,:].^2
test_x = collect([test_x1 test_x2]')

# For Plotting and computing reference density
rho = MvNormal(I(2))
t = range(-5,5,length=100)
rho_t = [pdf(rho, [t1,t2]) for t1 in t, t2 in t]

## Set up map and initialize coefficients
opts = MapOptions()
tri_map = CreateTriangular(2,2,2,opts)
coeffs = zeros(numCoeffs(tri_map))

function obj(coeffs, nothing)
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    rho_of_map_of_x = logpdf(rho, map_of_x)
    log_det = LogDeterminant(tri_map, x)
    -sum(rho_of_map_of_x + log_det)/num_points
end

function grad_obj(g, coeffs, nothing)
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    grad_rho_of_map_of_x = -CoeffGrad(tri_map, x, map_of_x)
    grad_log_det = LogDeterminantCoeffGrad(tri_map, x)
    g .= -sum(grad_rho_of_map_of_x + grad_log_det, dims=2)/num_points
end

## Plot before Optimization
map_of_x = Evaluate(tri_map, x)
if make_plot
    fig = Figure()
    ax1 = Axis(fig[1,1], title="Before Optimization")
    contour!(ax1, t, t, rho_t)
    scatter!(ax1, test_x[1,:], test_x[2,:], color=(:blue,0.5), label="Target Samples")
    axislegend(ax1)
    display(fig)
end

u0 = CoeffMap(tri_map)
fcn = OptimizationFunction(obj, grad = grad_obj)
prob = OptimizationProblem(fcn, u0, nothing, g_tol = 1e-16)

## Optimize


println("Starting coeffs")
println(u0)
println("and error: $(obj(u0,nothing))")
println("===================")
@time(sol = solve(prob, BFGS()));
##
u_final = sol.u
SetCoeffs(tri_map, u_final)
println("Final coeffs")
println(u_final)
println("and error: $(obj(u_final,nothing))")
println("===================")
map_of_test_x = Evaluate(tri_map, test_x)
if make_plot
    fig = Figure()
    ax2 = Axis(fig[1,1], title="After Optimization")
    contour!(ax2, t, t, rho_t)
    scatter!(ax2, map_of_test_x[1,:], map_of_test_x[2,:], color=(:blue,0.5), label="Target Samples")
    axislegend(ax2)
    display(fig)
end
mean_of_map = mean(map_of_test_x, dims=2)
cov_of_map = cov(map_of_test_x, dims=2)
println("Mean of map")
println(mean_of_map)
println("Covariance of map")
println(cov_of_map)