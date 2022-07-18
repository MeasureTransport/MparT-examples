include("MParT.jl")
using .MParT, CxxWrap
using Distributions, Optimization, OptimizationOptimJL

make_plot = false
##

num_points = 5000
mu = 2
sigma = 0.5
x = randn(num_points)

rv = Normal(mu, sigma)
t = range(-3,6,length=100)
rho_t = pdf.(rv, t)

num_bins = 50
fig = nothing
if make_plot

end
fig

##
A = reshape(Cint[0,1],2,1) # makes vector into matrix, need Cint for typing
mset = MultiIndexSet(A)
fixed_mset = Fix(mset, true)
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
function objective(coeffs)
    SetCoeffs(to_base(monotoneMap), coeffs)
    map_of_x = Evaluate(to_base(monotoneMap), x)
    pi_of_map_of_x = logpdf.(rv, map_of_x)
    log_det = LogDeterminant(monotoneMap, x)
    -sum(pi_of_map_of_x + log_det)/num_points
end

# Optimize
println("Starting Coeffs")
println(CoeffMap(to_base(monotoneMap)))
println("And error $(objective(CoeffMap(to_base(monotoneMap))))")
