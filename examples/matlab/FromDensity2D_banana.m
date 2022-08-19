%%
%%  Transport Map from density
%
% The objective of this example is to show how a transport map can be build in MParT when the the unnormalized probability density function of the target density is known.
%%  Problem description
%
% We consider $T(\mathbf{z};\mathbf{w})$ a monotone triangular transport map parameterized by $\mathbf{w}$ (e.g., polynomial coefficients). This map which is invertible and has an invertible Jacobian for any parameter $\mathbf{w}$, transports samples $\mathbf{z}^i$ from the reference density $\eta$ to samples $T(\mathbf{z}^i;\mathbf{w})$ from the map induced density $\tilde{\pi}_\mathbf{w}(\mathbf{z})$ defined as:
% $$ \tilde{\pi}_\mathbf{w}(\mathbf{z}) = \eta(T^{-1}(\mathbf{z};\mathbf{w}))|\text{det } T^{-1}(\mathbf{z};\mathbf{w})|,$$
% where $\text{det } T^{-1}$ is the determinant of the inverse map Jacobian at the point $\mathbf{z}$. We refer to $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ as the *map-induced* density or *pushforward distribution* and will commonly interchange notation for densities and measures to use the notation $\tilde{\pi} = T_{*} \eta$.
%
% The objective of this example is, knowing some unnormalized target density $\bar{\pi}$, find the map $T$ that transport samples drawn from $\eta$ to samples drawn from the target $\pi$.
%%  Imports
% First, import MParT and other packages used in this notebook. Note that it is possible to specify the number of threads used by MParT by setting the |KOKKOS_NUM_THREADS| environment variable *before* importing MParT.
%%

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
%%
%%  Target density and exact map
%
% In this example we use a 2D target density known as the *banana* density where the unnormalized probability density, samples and the exact transport map are known.
%
% The banana density is defined as:
% $$
% \pi(x_1,x_2) \propto N_1(x_1)\times N_1(x_2-x_1^2)
% $$
% where $N_1$ is the 1D standard normal density.
%
% The exact transport map that transport the 2D standard normal density to $\pi$ is known as:
% $$
%     {T}^\text{true}(z_1,z_2)=
%     \begin{bmatrix}
% z_1\\
% z_2 + z_1^2
% \end{bmatrix}
% $$
% Contours of the target density can be visualized as:
%%
% Unnomalized target density required for objective

% code 

% code 

% code 

% code 

% code 

% code 

% code 
% Gride for plotting

% code 

% code 

% code 

% code 

% code 
% For plotting and computing densities

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
%%
%%  Map training
%% Defining objective function and its gradient
% Knowing the closed form of the unnormalized target density $\bar{\pi}$, the objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{z})$ that is a good approximation of the target $\pi$.
%
% In order to characterize this posterior density, one method is to build a monotone triangular transport map $T$ such that the KL divergence $D_{KL}(\eta || T^* \pi)$ is minmized. If $T$ is map parameterized by $\mathbf{w}$, the objective function derived from the discrete KL divergence reads:
%
% $$
% J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),
% $$
%
% where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ to the target density $\pi(\mathbf{z})$. The gradient of this objective function reads
%
% $$
% \nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).
% $$
% The objective function and gradient can be defined using MParT as:
%%
% KL divergence objective

% code 

% code 

% code 

% code 

% code 

% code 
% Gradient of unnomalized target density required for gradient objective

% code 

% code 

% code 

% code 
% Gradient of KL divergence objective

% code 

% code 

% code 

% code 

% code 

% code 

% code 
%%
%% Map parameterization
% For the parameterization of $T$ we use a total order multivariate expansion of hermite functions. Knowing $T^\text{true}$, any parameterization with total order greater than one will include the true solution of the map finding problem.
%%
% Set-up first component and initialize map coefficients

% code 

% code 
% Create dimension 2 triangular map

% code 
%%
%% Approximation before optimization
%
% Coefficients of triangular map are set to 0 upon creation.
%%
% Make reference samples for training

% code 

% code 
% Make reference samples for testing

% code 
% Pushed samples

% code 
% Before optimization plot

% code 

% code 

% code 

% code 

% code 
%%
% At initialization, samples are "far" from being distributed according to the banana distribution.
% Initial objective and coefficients:
% Print initial coeffs and objective

% code 

% code 

% code 

% code 

% code 
%% Minimization
%%

% code 

% code 

% code 
% Print final coeffs and objective

% code 

% code 

% code 

% code 
%%
%% Approximation after optimization
%% Pushed samples
%%
% Pushed samples

% code 
% After optimization plot

% code 

% code 

% code 

% code 

% code 
%%
% After optimization, pushed samples $T(z^i)$, $z^i \sim \mathcal{N}(0,I)$ are approximately distributed according to the target $\pi$
%% Variance diagnostic
% A commonly used accuracy check when facing computation maps from density is the so-called variance diagnostic defined as:
%
% $$ \epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log \frac{\rho}{T^* \bar{\pi}} \right] $$
% This diagnostic is asymptotically equivalent to the minimized KL divergence $D_{KL}(\eta || T^* \pi)$ and should converge to zero when the computed map converge to the true map.
% The variance diagnostic can be computed as follow:

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
%%
% Reference distribution

% code 
% Compute variance diagnostic

% code 
% Print final coeffs and objective

% code 

% code 

% code 
%%
%% Pushforward density
% We can also plot the contour of the unnormalized density $\bar{\pi}$ and the pushforward approximation $T_* \eta$:
%%
% Pushforward definition

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
%%
matlab.internal.liveeditor.openAndConvert('FromDensity2D_banana.mlx','FromDensity2D_banana.m')
