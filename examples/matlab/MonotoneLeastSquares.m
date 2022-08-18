%% 
% To open this file as live script from the Editor, right-click the document 
% tab, and select *Open MonotoneLeastSquares.mlx as Live Script* from the context 
% menu. Alternatively, go to the *Editor* tab, click *Save*, and select *Save 
% As*. Then, set the *Save as type:* to |MATLAB Live Code Files (*.mlx)| and click 
% *Save*.
%% Montone least squares
% The objective of this example is to show how to build a transport map to solve 
% monotone regression problems using |MParT|.
%% Problem formulation
% One direct use of the monotonicity property given by the transport map approximation 
% to model monotone functions from noisy data. This is called isotonic regression 
% and can be solved in our setting by minimizing the least squares objective function
% 
% $$J(\mathbf{w})= \frac{1}{2} \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - y^i \right)^2$$
% 
% where $S$ is a monotone 1D map with parameters (polynomial coefficients) $\mathbf{w}$ 
% and $y^i$ are noisy observations. To solve for the map parameters that minimize 
% this objective we will use a gradient-based optimizer.  We therefore need the 
% gradient of the objective with respect to the map paramters. This is given by
% 
% $$\nabla_\mathbf{w} J(\mathbf{w})= \sum_{i=1}^N \left(S(x^i;\mathbf{w}) - 
% y^i \right)^T\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]$$
% 
% The implementation of |_S(x)|`_we're using from |MParT|, provides tools for 
% both evaluating the map to compute  $S(x^i;\mathbf{w})$ but also evaluating 
% computing the action of  $\left[\nabla_\mathbf{w}S(x^i;\mathbf{w})\right]^T$ 
% on a vector, which is useful for computing the gradient.   Below, these features 
% are leveraged when defining an objective function that we then minimize with 
% the built-in Matlab optimizer |fminunc|.

clear;
addpath(genpath('~/Installations/MParT/matlab/')) %installation path
%% 
% 

matlab.internal.liveeditor.openAndConvert('MonotoneLeastSquares.mlx','MonotoneLeastSquares.m')