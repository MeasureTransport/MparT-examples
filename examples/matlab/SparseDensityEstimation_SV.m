matlab.internal.liveeditor.openAndConvert('SparseDensityEstimation_SV.mlx','SparseDensityEstimation_SV.m')
%% 
%% Density estimation with sparse transport maps
% In this example we demonstrate how MParT can be use to build map with certain 
% sparse structure in order to characterize high dimensional densities with conditional 
% independence.
%% Imports
% First, import |MParT| by adding the path to the installation folder and initialize 
% the |Kokkos| environment. Note that it is possible to specify the number of 
% threads used by |MParT| as an argument of the |KokkosInitialize| function. The 
% number of threads can only be set once per session.

addpath(genpath('~/Installations/MParT/matlab'))
num_threads = 8;
KokkosInitialize(num_threads);
%% 
% Default settings:

sd = 3; rng(sd);

set(0,'DefaultLineLineWidth',1.75)
set(0,'defaultAxesFontSize',12)
set(0,'defaultfigurecolor',[1 1 1])
set(0, 'DefaultAxesBox', 'on');
%% 
% 
%% Stochastic volatility model
%% Problem description
% The problem considered here is a Markov process that describes the volatility 
% on a financial asset overt time. The model depends on two hyperparamters $\mu$ 
% and $\phi$ and state variable $Z_k$ represents log-volatility at times $k=1,...,T$. 
% The log-volatility follows the order-one autoregressive process: 
% 
% $$$Z_{k+1} = \mu + \phi(Z_k-\mu) + \epsilon_k, k>1, $$$ 
% 
% where 
% 
% $$$\mu \sim \mathcal{N}(0,1)  $$$ 
% 
% $$$ \phi = 2\frac{\exp(\phi^*)}{1+\exp(\phi^*)}, \,\,\, \phi^* \sim \mathcal{N}(3,1)$$
% 
% $$$ Z_0 | \mu, \phi \sim \mathcal{N}\left(\mu,\frac{1}{1-\phi^2}\right)$$
% 
% The objective is to characterize the joint density of $$\mathbf{X}_T = (\mu,\phi,Z_1,...,Z_T),    
% $$ with $T$ being arbitrarly large. The conditional independence property for 
% this problem reads
% 
% $$ \pi(\mathbf{x}_t|\mathbf{x}_{<t}) = \pi(\mathbf{x}_t|\mathbf{x}_{t-1},\mu,\phi)$$
% 
% More details about this problem can be found in <https://arxiv.org/pdf/2009.10303.pdf%3E 
% Baptista et al., 2022>.
%% Sampling
% Drawing samples $(\mu^i,\phi^i,x_0^i,x_1^i,...,x_T^i)$ can be performed by 
% the following function

% code 
% Sample hyper-parameters

% code 

% code 

% code 

% code 

% code 

% code 
% Sample Z0

% code 
% Sample auto-regressively

% code 

% code 

% code 

% code 

% code 
% Set dimension of the problem:

% code 

% code 
% Few realizations of the process look like
%% 
% code 

% code 

% code 

% code 

% code 

% code
%% 
% And corresponding realization of hyperparameters

% code 

% code 

% code 

% code 

% code 

% code 

% code
%% Probability density function
% The exact log-conditional densities used to define joint density $\pi(\mathbf{x}_T)$ 
% are defined by the following function:

% code 

% code 

% code 

% code 
% Extract variables mu, phi and states

% code 

% code 

% code 
% Compute density for mu

% code 

% code 
% Compute density for phi

% code 

% code 

% code 

% code 
% Add piMu, piPhi to density

% code 
% Number of time steps

% code 

% code 
% Conditonal density for Z_0

% code 

% code 

% code 

% code 
% Compute auto-regressive conditional densities for Z_i|Z_{1:i-1}

% code 

% code 

% code 

% code 

% code 

% code
%% Transport map training
% In the following we optimize each map component $S_k$, $k \in \{1,...,T+2\}$: 
% * For $k=1$, map $S_1$ characterize marginal density $\pi(\mu)$ * For $k=2$, 
% map $S_2$ characterize conditional density $\pi(\phi|\mu)$ * For $k=3$, map  
% $S_3$ characterize conditional density $\pi(z_0|\phi,\mu)$ * For $k>3$, map  
% $S_k$ characterize conditional density $\pi(z_{k-2}|z_{k-3},\phi,\mu)$ Definition 
% of log-conditional density from map component $S_k$

% code 

% code 

% code 

% code
%% Generating training and testing samples
% From training samples generated with the known function we compare accuracy 
% of the transport map induced density using different parameterization and a 
% limited number of training samples.

% code 

% code 

% code 

% code
%% 
% 
%% Objective function and gradient
% We use the minimization of negative log-likelihood to optimize map components. 
% For map component $k$, the objective function is given by
% 
% $$J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left( \log\eta\left(S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right)  
% + \log \frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right)$$
% 
% and corresponding gradient
% 
% $$\nabla_{\mathbf{w}_k}J_k(\mathbf{w}_k) = - \frac{1}{N}\sum_{i=1}^N \left(\left[\nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right]^T 
% \nabla_\mathbf{r}\log \eta \left(S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)\right) 
% - \frac{\partial \nabla_{\mathbf{w}_k}S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial 
% x_k} \left[\frac{\partial S_k(\mathbf{x}_{1:k}^i;\mathbf{w}_k)}{\partial x_k}\right]^{-1}\right),$$
%% Negative log likelihood objective

% code 

% code 

% code 

% code 
% Compute the map-induced density at each point

% code 

% code 

% code 

% code 
% Return the negative log-likelihood of the entire dataset

% code 

% code 

% code 

% code 

% code 
% Evaluate the map

% code 
% Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)

% code 
% Get the gradient of the log determinant with respect to the map coefficients

% code 

% code
%% 
% 
%% Training total order 1 map
% Here we use a total order 1 multivariate expansion to parameterize each component  
% $S_k$, $k \in \{1,...,T+2\}$.

% code 

% code
%% Optimization
% Total order 1 approximation

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
% Reference density

% code 
% Compute log-conditional density at testing samples

% code 

% code
%% 
% 
%% Compute KL divergence error
% Since we know what the true is for problem we can compute the KL divergence  
% $D_{KL}(\pi(\mathbf{x}_t)||S^* \eta)$ between the map-induced density and the 
% true density.

% code 

% code 

% code 

% code 

% code 

% code 
% Compute joint KL divergence for total order 1 approximation

% code
%% 
% 
%% Training total order 2 map
% Here we use a total order 2 multivariate expansion to parameterize each component  
% $S_k$, $k \in \{1,...,T+2\}$.
%% Optimization
% This step can take quite a long time depending of the number of time steps
% 
% Total order 2 approximation

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
% Reference density

% code 
% Compute log-conditional density at testing samples

% code 

% code
%% 
% 
%% Compute KL divergence error
% Compute joint KL divergence for total order 2 approximation

% code
%% Training sparse map
% Here we use the prior knowledge of the conditional independence property of 
% the target density $\pi(\mathbf{x}_T)$ to parameterize map components with a 
% map structure.
%% Prior knowledge used to parameterize map components
% From the independence structure mentionned in the problem formulation we have:
%% 
% * $\pi(\mu,\phi)=\pi(\mu)\pi(\phi)$, meaning $S_2$ only dependes on $\phi$
% * $\pi(z_{k-2}|z_{k-3},...,z_{0},\phi,\mu)=\pi(z_{k-2}|z_{k-3},\phi,\mu),\,\,    
% k>3$, meaning $S_k$, only depends on $z_{k-2}$, $z_{k-3}$ , $\phi$ and $\mu$
%% 
% Complexity of map component can also be deducted from problem formulation:
%% 
% * $\pi(\mu)$ being a normal distribution, $S_1$ should be of order 1.
% * $\pi(\phi)$ is non-Gaussian such that $S_2$ should be nonlinear.
% * $\pi(z_{k-2}|z_{k-3},\phi,\mu)$ can be represented by a total order 2 parameterization 
% due to the linear autoregressive model.
%% 
% Hence multi-index sets used for this problem are:
%% 
% * $k=1$: 1D expansion of order $\geq$ 1
% * $k=2$: 1D expansion (depending on last component) of high order $>1$
% * $k=3$: 3D expansion of total order 2
% * $k>3$: 4D expansion (depending on first two and last two components) of 
% total order 2
%% Optimization

% code 

% code 

% code 
% MultiIndexSet for map S_k, k>3

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

% code 

% code 

% code 

% code 

% code
%% 
% 
%% Compute KL divergence error
% Compute joint KL divergence

% code
%% Compare approximations
%% KL divergence
% Compare map approximations

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
% Usually increasing map complexity will improve map approximation. However when the number of parameters increases too much compared to the number of samples, computed map overfits the data which lead to worst approximation. This overfitting can be seen in this examples when looking at the total order 2 approximation that slowly loses accuracy when the dimension increases. Total order 2 approximation while performing better than order 1 for low dimension perform worst when dimension is greater than ~27. approximation thatn total order 1 with dimension greater than 27.
%
% Using sparse multi-index sets help reduces the increase of parameters when the dimension increases leading to better approximation for all dimensions.
%% Map coefficients
% To complement observations made above, we visualize the number of parameters 
% (polyniomal coefficients) for each map parameterization.

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
% We can observe the exponential growth of the number coefficients for the total order 2 approximation. Chosen sparse multi-index sets have a fixed number of parameters which become smaller than the number of parameters of the total order 1 approximation when dimension is about 15.
% Using less parameters help error scaling with the number of dimension but also computation time for the optimization and the computation time when evaluating the transport map.
%% Custom functions needed for this example
% 

function X = generate_SV_samples(d,N)
    % Sample hyper-parameters
    sigma = 0.25;
    mu = randn(1,N);
    phis = 3+randn(1,N);
    phi = 2*exp(phis)./(1+exp(phis))-1;
    X = [mu;phi];
    if d > 2
        % Sample Z0
        Z = sqrt(1./(1-phi.^2))*randn(1,N) + mu;
        % Sample auto-regressively
        for i=1:d-3
            Zi = mu + phi.*(Z(end,:)-mu)+sigma*randn(1,N);
            Z = [Z;Zi];
        end
        X = [X;Z];
    end
end
%% 
% 

function logPdf=SV_log_pdf(X)
    %conditional log-pdf for the SV problem

    sigma = 0.25;
    % Extract variables mu, phi and states
    mu = X(1,:);
    phi = X(2,:);
    Z = X(3:end,:);

    % Compute density for mu
    logPdfMu = log(normpdf(mu));
    % Compute density for phi
    phiRef = log((1+phi)./(1-phi));
    dphiRef = 2./(1-phi.^2);
    logPdfPhi = normpdf((phiRef)+log(dphiRef),3,1);
    % Add piMu, piPhi to density
    logPdf = [logPdfMu;logPdfPhi];

    % Number of time steps
    dz = size(Z,1);
    if dz > 0
        % Conditonal density for Z_0
        muZ0 = mu;
        stdZ0 = sqrt(1./(1-phi.^2));
        logPdfZ0 = log(normpdf(Z(1,:),muZ0,stdZ0));
        logPdf = [logPdf;logPdfZ0];

        % Compute auto-regressive conditional densities for Z_i|Z_{1:i-1}
        for i=2:dz
            meanZi = mu + phi .*(Z(i-1,:)-mu);
            stdZi = sigma;
            logPdfZi = log(normpdf(Z(i,:),meanZi,stdZi));
            logPdf = [logPdf;logPdfZi];
        end
    end
end
%% 
% 

 function log_pdf=log_cond_pullback_pdf(triMap,x)
    %log-conditonal pullback density
    r = triMap.Evaluate(x);
    log_pdf = log(mvnpdf(r'))+triMap.LogDeterminant(x)';
 end 
%% 
% 

function [L,dwL]=objective(coeffs,tri_map,x)
% Negative log likelihood objective
num_points = size(x,2);
tri_map.SetCoeffs(coeffs);

% Compute the map-induced density at each point
map_of_x = tri_map.Evaluate(x);
ref_of_map_of_x = log(normpdf(map_of_x));
log_det = tri_map.LogDeterminant(x);

% negative log-likelihood of the entire dataset
L = - sum(ref_of_map_of_x + log_det')/num_points;

if (nargout > 1)
    % Compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
    grad_ref_of_map_of_x = -tri_map.CoeffGrad(x,map_of_x);

    % Get the gradient of the log determinant with respect to the map coefficients
    grad_log_det = tri_map.LogDeterminantCoeffGrad(x);

    % Gradient of the negative log-likelihood
    dwL = - sum(grad_ref_of_map_of_x + grad_log_det,2)/num_points;
end
end