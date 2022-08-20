%% Density estimation with sparse transport maps
%
% In this example we demonstrate how MParT can be use to build map with certain 
% sparse structure in order to characterize high dimensional densities with conditional 
% independence.
% 
% _The rendered live script version of this example can be obtain by doing "Open 
% as Live Script” from the Current Folder Browser or document tab or by doing 
% "Save as ..." and select the "MATLAB Live Code Files (*.mlx)" format._
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
%% Stochastic volatility model
%% Problem description
% The problem considered here is a Markov process that describes the volatility 
% on a financial asset overt time. The model depends on two hyperparamters $\mu$ 
% and $\phi$ and state variable $Z_k$ represents log-volatility at times $k=1,...,T$. 
% The log-volatility follows the order-one autoregressive process: 
% 
% $$Z_{k+1} = \mu + \phi(Z_k-\mu) + \epsilon_k, k>1,$$
% 
% where 
% 
% $$ \phi = 2\frac{\exp(\phi^*)}{1+\exp(\phi^*)}, \,\,\, \phi^* \sim \mathcal{N}(3,1)$$
% 
% $$ Z_0 | \mu, \phi \sim \mathcal{N}\left(\mu,\frac{1}{1-\phi^2}\right)$$
% 
% The objective is to characterize the joint density of $$\mathbf{X}_T = (\mu,\phi,Z_1,...,Z_T),        
% with $T$ being arbitrarly large. The conditional independence property for this 
% problem reads
% 
% $$ \pi(\mathbf{x}_t|\mathbf{x}_{<t}) = \pi(\mathbf{x}_t|\mathbf{x}_{t-1},\mu,\phi)$$
% 
% More details about this problem can be found in <https://arxiv.org/pdf/2009.10303.pdf%3E 
% Baptista et al., 2022>.
%% Sampling
% Drawing samples $(\mu^i,\phi^i,x_0^i,x_1^i,...,x_T^i)$ can be performed by 
% function |generate_SV_samples| defined at the end of the script.
% 
% Set dimension of the problem:

T = 30; %Number of time steps including initial condition
d = T+2;
%% 
% Few realizations of the process look like

Nvisu = 10; %Number of samples
Xvisu = generate_SV_samples(d,Nvisu);

Zvisu = Xvisu(3:end,:);

figure
plot(1:T,Zvisu)
xlim([1 T])
xlabel('Days (d)')
%% 
% And corresponding realization of hyperparameters

hyper_params = Xvisu(1:2,:);

figure
hold on
plot(1:Nvisu,Xvisu(1,:))
plot(1:Nvisu,Xvisu(2,:))
xlim([1 Nvisu])
xlabel('Samples')
legend('\mu','\phi')
%% Probability density function
% The exact log-conditional densities used to define joint density $\pi(\mathbf{x}_T)$ 
% are defined by the |SV_log_pdf| function.
%% Transport map training
% In the following we optimize each map component $S_k$, $k \in \{1,...,T+2\}$: 
%% 
% * For $k=1$, map $S_1$ characterize marginal density $\pi(\mu)$
% * For $k=2$, map $S_2$ characterize conditional density $\pi(\phi|\mu)$
% * For $k=3$, map  $S_3$ characterize conditional density $\pi(z_0|\phi,\mu)$
% * For $k>3$, map  $S_k$ characterize conditional density $\pi(z_{k-2}|z_{k-3},\phi,\mu)$ 
%% 
% Knowing $S_k$, the map induced log-density is defined by the function |log_cond_pullback_pdf|.
%% Generating training and testing samples
% From training samples generated with the known function we compare accuracy 
% of the transport map induced density using different parameterization and a 
% limited number of training samples.

N = 2000; %Number of training samples
X = generate_SV_samples(d,N);

Ntest = 5000; %Number of testing samples
Xtest = generate_SV_samples(d,Ntest);
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
% 
% These functions are defined in the |objective| function.
%% Training total order 1 map
% Here we use a total order 1 multivariate expansion to parameterize each component  
% $S_k$, $k \in \{1,...,T+2\}$.

opts = MapOptions;
opts.basisType = BasisTypes.HermiteFunctions;
%% Optimization
% Total order 1 approximation

totalOrder = 1;
logPdfTM_to1 = zeros(d,Ntest);
ListCoeffs_to1=zeros(1,d);

disp('===== Total Order 1 approximation =====')
fprintf('Map component:')
for dk=1:d
    fprintf('%d ', dk);
    fixed_mset = FixedMultiIndexSet(dk,totalOrder);
    S = CreateComponent(fixed_mset,opts);
    Xtrain = X(1:dk,:);
    Xtestk = Xtest(1:dk,:);

    ListCoeffs_to1(dk) = S.numCoeffs;

    % Optimize
    obj = @(w) objective(w,S,Xtrain);
    w0 = S.Coeffs();
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'none');
    [~] = fminunc(obj, w0, options);
    logPdfTM_to1(dk,:) = log_cond_pullback_pdf(S,Xtestk);
end
fprintf('\n');
%% Compute KL divergence error
% Since we know what the true is for problem we can compute the KL divergence  
% $D_{KL}(\pi(\mathbf{x}_t)||S^* \eta)$ between the map-induced density and the 
% true density.

logPdfSV = SV_log_pdf(Xtest); % true log-pdfs

% Compute joint KL divergence for total order 1 approximation
KL_to1 = compute_joint_KL(logPdfSV,logPdfTM_to1);
%% Training total order 2 map
% Here we use a total order 2 multivariate expansion to parameterize each component  
% $S_k$, $k \in \{1,...,T+2\}$.
%% Optimization
% This step can take few minutes depending on the number of time steps set at the definition of the problem 

totalOrder = 2;
logPdfTM_to2 = zeros(d,Ntest);
ListCoeffs_to2=zeros(1,d);
disp('===== Total Order 2 approximation =====')
fprintf('Map component:')
for dk=1:d
    fprintf('%d ', dk);
    fixed_mset = FixedMultiIndexSet(dk,totalOrder);
    S = CreateComponent(fixed_mset,opts);
    Xtrain = X(1:dk,:);
    Xtestk = Xtest(1:dk,:);

    ListCoeffs_to2(dk) = S.numCoeffs;

    % Optimize
    obj = @(w) objective(w,S,Xtrain);
    w0 = S.Coeffs();
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'none');
    [~] = fminunc(obj, w0, options);

    logPdfTM_to2(dk,:) = log_cond_pullback_pdf(S,Xtestk);
end
fprintf('\n');
%% Compute KL divergence error
% Compute joint KL divergence for total order 2 approximation

% Compute joint KL divergence for total order 2 approximation
KL_to2 = compute_joint_KL(logPdfSV,logPdfTM_to2);
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

totalOrder = 2;
logPdfTM_sa = zeros(d,Ntest);
ListCoeffs_sa = zeros(1,d);

% MultiIndexSet for map S_k, k>3
mset_to = MultiIndexSet.CreateTotalOrder(4,totalOrder);

maxOrder = 9; %order for map S_2
disp('===== Sparse map approximation =====')
fprintf('Map component:')
for dk = 1:d
    fprintf('%d ', dk);
    if dk==1
        fixed_mset = FixedMultiIndexSet(1,totalOrder);
        S = CreateComponent(fixed_mset,opts);
        Xtrain = X(dk,:);
        Xtestk = Xtest(dk,:);
    elseif dk==2
        fixed_mset = FixedMultiIndexSet(1,maxOrder);
        S = CreateComponent(fixed_mset,opts);
        Xtrain = X(dk,:);
        Xtestk = Xtest(dk,:);
    elseif dk==3
        fixed_mset = FixedMultiIndexSet(dk,totalOrder);
        S = CreateComponent(fixed_mset,opts);
        Xtrain = X(1:dk,:);
        Xtestk = Xtest(1:dk,:);
    else
        multis=zeros(mset_to.Size(),dk);
        for s=1:mset_to.Size()
            multis_to_mt = mset_to{s};
            multis_to = multis_to_mt.Vector();
            multis(s,1:2)=multis_to(1:2);
            multis(s,end-1:end)=multis_to(3:4);
        end
        mset = MultiIndexSet(multis);
        fixed_mset = mset.Fix();
        S = CreateComponent(fixed_mset,opts);
        Xtrain = X(1:dk,:);
        Xtestk = Xtest(1:dk,:);
    end

    ListCoeffs_sa(dk)=S.numCoeffs;

    % Optimize
    obj = @(w) objective(w,S,Xtrain);
    w0 = S.Coeffs();
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'none');
    [~] = fminunc(obj, w0, options);

    logPdfTM_sa(dk,:) = log_cond_pullback_pdf(S,Xtestk);
end
fprintf('\n');
%% Compute KL divergence error
% Compute joint KL divergence

% Compute joint KL divergence for sparse map approximation
KL_sa = compute_joint_KL(logPdfSV,logPdfTM_sa);
%% Compare approximations
%% KL divergence
% Compare map approximations

figure
hold on
plot(1:d,KL_to1,'-o')
plot(1:d,KL_to2,'-o')
plot(1:d,KL_sa,'-o')
set(gca, 'YScale', 'log')
xlabel('d')
ylabel('$D_{KL}(\pi(\mathbf{x}_t)||S^*\eta)$',Interpreter='latex')
legend('Total order 1','Total order 2','Sparse MultiIndexSet')
%% 
% Usually increasing map complexity will improve map approximation. However 
% when the number of parameters increases too much compared to the number of samples, 
% computed map overfits the data which lead to worst approximation. This overfitting 
% can be seen in this examples when looking at the total order 2 approximation 
% that rapidly loses accuracy when the dimension increases. 
% 
% Using sparse multi-index sets help reduces the increase of parameters when 
% the dimension increases leading to better approximation for all dimensions.
%% Map coefficients
% To complement observations made above, we visualize the number of parameters 
% (polyniomal coefficients) for each map parameterization.

figure
hold on
plot(1:d,ListCoeffs_to1,'-o')
plot(1:d,ListCoeffs_to2,'-o')
plot(1:d,ListCoeffs_sa,'-o')
xlabel('d')
ylabel('# coefficients')
legend('Total order 1','Total order 2','Sparse MultiIndexSet')
%% 
% We can observe the exponential growth of the number coefficients for the total 
% order 2 approximation. Chosen sparse multi-index sets have a fixed number of 
% parameters which become smaller than the number of parameters of the total order 
% 1 approximation when dimension is 15.
% 
% Using less parameters helps error scaling with dimension but aslo helps reducing 
% computation time for the optimization and the evaluation the transport maps.
%% Custom functions needed for this example

function X = generate_SV_samples(d,N)
% Sample hyper-parameters
sigma = 0.25;
mu = randn(1,N);
phis = 3+randn(1,N);
phi = 2*exp(phis)./(1+exp(phis))-1;
X = [mu;phi];
if d > 2
    % Sample Z0
    Z = sqrt(1./(1-phi.^2)).*randn(1,N) + mu;
    % Sample auto-regressively
    for i=1:d-3
        Zi = mu + phi.*(Z(end,:)-mu)+sigma*randn(1,N);
        Z = [Z;Zi];
    end
    X = [X;Z];
end
end

function logPdf=SV_log_pdf(X)
%conditional log-pdf for the SV problem

sigma = 0.25;
% Extract variables mu, phi and states
mu = X(1,:);
phi = X(2,:);
Z = X(3:end,:);

% Compute density for mu
logPdfMu = normlogpdf(mu);
% Compute density for phi
phiRef = log((1+phi)./(1-phi));
dphiRef = 2./(1-phi.^2);
logPdfPhi = normlogpdf(phiRef,3,1)+log(dphiRef);
% Add piMu, piPhi to density
logPdf = [logPdfMu;logPdfPhi];

% Number of time steps
dz = size(Z,1);
if dz > 0
    % Conditonal density for Z_0
    muZ0 = mu;
    stdZ0 = sqrt(1./(1-phi.^2));
    logPdfZ0 = normlogpdf(Z(1,:),muZ0,stdZ0);
    logPdf = [logPdf;logPdfZ0];

    % Compute auto-regressive conditional densities for Z_i|Z_{1:i-1}
    for i=2:dz
        meanZi = mu + phi .*(Z(i-1,:)-mu);
        stdZi = sigma;
        logPdfZi = normlogpdf(Z(i,:),meanZi,stdZi);
        logPdf = [logPdf;logPdfZi];
    end
end
end

function log_pdf=log_cond_pullback_pdf(triMap,x)
%log-conditonal pullback density
r = triMap.Evaluate(x);
log_pdf = normlogpdf(r)+triMap.LogDeterminant(x)';
end

function logpdf=normlogpdf(varargin)
    x = varargin{1};
    if nargin ==1
    logpdf = -0.5*(log(2*pi)+x.^2);
    else
     logpdf = -log(sqrt(2*pi)*varargin{3})-0.5*((x-varargin{2})/varargin{3}).^2;
    end
end

function [L,dwL]=objective(coeffs,tri_map,x)

    % Negative log likelihood objective
    num_points = size(x,2);
    tri_map.SetCoeffs(coeffs);
    
    % Compute the map-induced density at each point
    map_of_x = tri_map.Evaluate(x);
    ref_of_map_of_x = normlogpdf(map_of_x);
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

function KL = compute_joint_KL(logPdfSV,logPdfTM)
% compute the KL divergence between true joint density and map
% approximation
d = size(logPdfSV,1);
KL = zeros(1,d);
for k=1:d
    KL(k)=mean(sum(logPdfSV(1:k,:),1)-sum(logPdfTM(1:k,:),1));
end
end