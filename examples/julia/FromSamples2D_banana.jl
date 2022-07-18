include("MParT.jl")
##
using .MParT, CxxWrap
using Distributions, LinearAlgebra, Optimization, OptimizationOptimJL

make_plot = true

if make_plot
    using CairoMakie
end
##
num_points = 1000
z = randn(2,num_points)
x1 = z[1,:]
x2 = z[1,:] + z[2,:].^2
x = [x1; x2]

test_n_pts = 10_000
test_z = randn(2,test_n_pts)
test_x1 = test_z[1,:]
test_x2 = test_z[1,:] + test_z[2,:].^2
test_x = [test_x1; test_x2]

# For Plotting and computing reference density
rho = MvNormal(I(2))
t = range(-5,5,length=100)
grid = [[t1,t2] for t1 in t, t2 in t]
rho_t = pdf.(rv, grid)

## Set up map and initialize coefficients