%% Monotone least squares
% The objective of this example is to show how to build a transport map to solve 
% monotone regression problems using MParT.
%% Problem formulation
% One direct use of the monotonicity property given by the transport map approximation 
% to model monotone functions from noisy data. This is called isotonic regression 
% and can be solved in our setting by minimizing the least squares objective function
% 
% $$J(\mathbf{w})= \frac{1}{2} \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - y^i \right)^2,$$
% 
% where $S$ is a monotone 1D map with parameters (polynomial coefficients) $\mathbf{w}$ 
% and $y^i$ are noisy observations. To solve for the map parameters that minimize 
% this objective we will use a gradient-based optimizer. We therefore need the 
% gradient of the objective with respect to the map paramters. This is given by
% 
% $$\nabla_\mathbf{w} J(\mathbf{w})= \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - 
% y^i \right)^T\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]$$
% 
% The implementation of |S(x)| we're using from MParT, provides tools for both 
% evaluating the map to compute  $S(x^i;\mathbf{w})$ but also evaluating computing 
% the action of  $\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]^T$ on a vector, 
% which is useful for computing the gradient. Below, these features are leveraged 
% when defining an objective function that we then minimize with the BFGS optimizer 
% implemented in |scipy.minimize|.
%% Imports
% First, import MParT and other packages used in this notebook. Note that it 
% is possible to specify the number of threads used by MParT by setting the |KOKKOS_NUM_THREADS| 
% environment variable *before* importing MParT.

addpath(genpath('~/Installations/MParT/matlab'))

sd = 3; rng(sd);

set(0,'DefaultLineLineWidth',1.75)
set(0,'defaultAxesFontSize',14)
set(0,'defaultfigurecolor',[1 1 1])

%% Generate training data
%% True model
% Here we choose to use the step function $H(x)=\text{sgn}(x-2)+1$ as the reference 
% monotone function. It is worth noting that this function is not strictly monotone 
% and piecewise continuous.

% variation interval
num_points = 1000;
xmin = 0;
xmax = 4;
x = linspace(xmin,xmax,num_points);

y_true = 2*(x>2);

figure
title('Reference data')
plot(x,y_true)
xlabel('x')
ylabel('H(x)')
ylim([-0.1,2.1])
%% Training data
% Training data $y^i$ in the objective defined above are simulated by pertubating 
% the reference data with a white Gaussian noise with a $0.4$ standard deviation.

noisesd = 0.4

y_noise = noisesd*randn(1,num_points)
y_measured = y_true + y_noise
%% Map initialization
% We use the previously generated data to train the 1D transport map. In 1D, 
% the map complexity can be set via the list of multi-indices. Here, map complexity 
% can be tuned by setting the |max_order| variable.
%% Multi-index set

% Define multi-index set
% Set options and create map object
%% Plot initial approximation

% Before optimization
% Plot data (before and after apart)

% Initial map with coefficients set to zero result in the identity map.
%% Transport map training
%% Objective function

% Least squares objective
% Gradient of objective
%% Optimization

% Optimize
% After optimization
%% Plot final approximation
% Unlike the true underlying model, map approximation gives a strict coninuous 
% monotone regression of the noisy data.

matlab.internal.liveeditor.openAndConvert('MonotoneLeastSquares.mlx','MonotoneLeastSquares.m')