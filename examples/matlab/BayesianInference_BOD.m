%% Characterization of Bayesian posterior density
%
% The objective of this example is to demonstrate how transport maps can be 
% used to represent posterior distribution of Bayesian inference problems.
%
% _The rendered live script version of this example can be obtain by doing "Open 
% as Live Script” from the Current Folder Browser or document tab or by doing 
% "Save as ..." and select the "MATLAB Live Code Files (*.mlx)" format._
%
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
%% Problem formulation
%% Bayesian inference
% A way construct a transport map is from an unnormalized density. One situation 
% where we know the probality density function up to a normalization constant 
% is when modeling inverse problems with Bayesian inference.
% 
% For an inverse problem, the objective is to characterize the value of some 
% parameters $\mathbf{\theta}$ of a given system, knowing some the value of some 
% noisy observations $\mathbf{y}$.
% 
% With Bayesian inference, the characterization of parameters $\mathbf{\theta}$ 
% is done via a *posterior* density $\pi(\mathbf{\theta}|\mathbf{y})$. This density 
% characterizes the distribution of the parameters knowing the value of the observations.
% 
% Using Bayes' rule, the posterior can decomposed into the product of two probability 
% densities:
% 
% 1. The prior density $\pi(\mathbf{\theta})$ which is used to enforce any *a 
% priori* knowledge about the parameters. 
% 
% 2. The likelihood function $\pi(\mathbf{y}|\mathbf{\theta})$. This quantity 
% can be seen as a function of $\mathbf{\theta}$ and gives the likelihood that 
% the considered system produced the observation $\mathbf{y}$ for a fixed value 
% of $\mathbf{\theta}$. When the model that describes the system is known in closed 
% form, the likelihood function is also knwon in closed form.
% 
% Hence, the posterior density reads:
% 
% $$\pi(\mathbf{\theta}|\mathbf{y}) = \frac{1}{c} \pi(\mathbf{y}|\mathbf{\theta}) 
% \pi(\mathbf{\theta})$$
% 
% where $c = \int \pi(\mathbf{y}|\mathbf{\theta}) \pi(\mathbf{\theta}) \text{d}\mathbf{\theta}$ 
% is an normalizing constant that ensures that the product of the two quantities 
% is a proper density. Typically, the integral in this definition cannot be evaluated 
% and $c$ is assumed to be unknown. The objective of this examples is, from the 
% knowledge of $\pi(\mathbf{y}|\mathbf{\theta})\pi(\mathbf{\theta})$ build a transport 
% map that transports samples from the reference $\eta$ to samples from posterior  
% $\pi(\mathbf{\theta}|\mathbf{y})$.
%% Application with the Biochemical Oxygen Demand (BOD) model from <https://or.water.usgs.gov/proj/keno_reach/download/chemgeo_bod_final.pdf Sullivan et al., 2010>
%% Definition
% To illustrate the process describe above, we consider the BOD inverse problem 
% described in <https://arxiv.org/pdf/1602.05023.pdf Marzouk et al., 2016>. The 
% goal is to estimate $2$ coefficients in a time-dependent model of oxygen demand, 
% which is used as an indication of biological activity in a water sample.
% 
% The time dependent forward model is defined as
% 
% $$\mathcal{B}(t) = A(1-\exp(-Bt))+\mathcal{E},$$
% 
% where
% 
% $$\mathcal{E} \sim \mathcal{N}(0,1e-3)$$
% 
% $$A  = \left[0.4 + 0.4\left(1 + \text{erf}\left(\frac{\theta_1}{\sqrt{2}} 
% \right)\right) \right]$$
% 
% $$B  = \left[0.01 + 0.15\left(1 + \text{erf}\left(\frac{\theta_2}{\sqrt{2}} 
% \right)\right) \right]$$
% 
% The objective is to characterize the posterior density of parameters $\mathbf{\theta}=(\theta_1,\theta_2)$ 
% knowing observation of the system at time $t=\left\{1,2,3,4,5 \right\}$ i.e.  
% $\mathbf{y} = (y_1,y_2,y_3,y_4,y_5) = (\mathcal{B}(1),\mathcal{B}(2),\mathcal{B}(3),\mathcal{B}(4),\mathcal{B}(5))$.
% 
% Closed forms of the forward model and gradient with respect to $\mathbf{\theta}$ 
% can easily be derived and are provided at end of the script by the |fowad_model| 
% and |grad_x_forward_model| functions.
% 
% One simulation of the forward model:

t = linspace(0,10,100);
figure
plot(t,forward_model(1,1,t));
xlabel('t')
ylabel('BOD')
%% 
% For this problem, as noise $\mathcal{E}$ is Gaussian and additive, the likelihood 
% function $\pi(\mathbf{y}|\mathbf{\theta})$, can be decomposed for each time 
% step as:
% 
% $$\pi(\mathbf{y}|\mathbf{\theta}) = \prod_{t=1}^{5} \pi(y_t|\mathbf{\theta}), 
% $$
% 
% where
% 
% $$\pi(\mathbf{y}_t|\mathbf{\theta})=\frac{1}{\sqrt{0.002.\pi}}\exp \left(-\frac{1}{0.002} 
% \left(y_t - \mathcal{B}(t)\right)^2 \right), t \in \{1,...,5\}.$$
% 
% Likelihood function and its gradient with respect to parameters are defined 
% in functions |log_likelihood| and |grad_x_log_likelihood|.
% 
% We can then define the unnormalized posterior and its gradient with respect 
% to parameters with functions |log_posterior| and |grad_x_log_posterior|.
%% Observations
% We consider the following realization of observation $\mathbf{y}$:

list_t = 1:5;
list_yobs = [0.18 0.32 0.42 0.49 0.54];

std_noise = sqrt(1e-3);
std_prior1 = 1;
std_prior2 = 1;
%% Visualization of the *unnormalized* posterior density

Ngrid = 100;
x = linspace(-0.5,1.25);
y = linspace(-0.5,2.5);
[X,Y] = meshgrid(x,y);

Z = log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,X(:),Y(:));
Z = reshape(exp(Z),Ngrid,Ngrid);

figure
contour(X,Y,Z)
xlabel('\theta_1')
ylabel('\theta_2')
legend('Unnormalized posterior')
%% 
% Target density for the map from density is non-Gaussian which mean that a 
% non linear map will be required to achieve good approximation.
%% Map computation
% After the definition of the log-posterior and gradient, the construction of 
% the desired map $T$ to characterize the posterior density result in a "classic" 
% map from unnomarlized computation.
%% Definition of the objective function:
% Knowing the closed form of unnormalized posterior $\bar{\pi}(\mathbf{\theta}      
% |\mathbf{y})= \pi(\mathbf{y}|\mathbf{\theta})\pi(\mathbf{\theta})$, the objective 
% is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ that 
% is a good approximation to the posterior $\pi(\mathbf{\theta} |\mathbf{y})$.
% 
% In order to characterize this posterior density, one method is to build a 
% transport map.
% 
% For the map from unnormalized density estimation, the objective function on 
% parameter $\mathbf{w}$ reads:
% 
% $$J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) 
% + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, 
% \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),$$
% 
% where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ 
% to the target density $\pi(\mathbf{x})$, which will be the the posterior density. 
% The gradient of this objective function reads:
% 
% $$\nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w} 
% T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) 
% + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), 
% \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d).$$

%Draw reference samples to define objective
N = 10000;
z = randn(2,N);
%% Map parametrization
% We use the MParT function |CreateTriangular| to directly create a transport 
% map object of dimension with given total order parameterization.

% Create transport map
opts = MapOptions;
total_order = 3;
tri_map = CreateTriangular(2,2,total_order,opts);
%% Optimization

log_target = @(x) log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,x(1,:),x(2,:));
grad_x_log_target = @(x) grad_x_log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,x(1,:),x(2,:));

% Optimize
obj = @(w) objective(log_target,grad_x_log_target,w,tri_map,z);
w0 = tri_map.Coeffs();


options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'none');
[~] = fminunc(obj, w0, options);
%% Accuracy checks
%% Comparing density contours
% Comparison between contours of the posterior $\pi(\mathbf{\theta}|\mathbf{y})$ 
% and conoturs of pushforward density $T_* \eta$. Pushforward distribution

xx_eval= [X(:)';Y(:)'];

Z2 = push_forward_pdf(tri_map,xx_eval);
Z2 = reshape(Z2,Ngrid,Ngrid);

figure
hold on
contour(X,Y,Z)
contour(X,Y,Z2,'LineStyle','--')
xlabel('\theta_1')
ylabel('\theta_2')
legend('Unnormalized posterior','TM approximation')
%% Variance diagnostic
% A commonly used accuracy check when facing computation maps from density is 
% the so-called variance diagnostic defined as:
% 
% $$ \epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log \frac{\rho}{T^* 
% \bar{\pi}} \right] $$
% 
% This diagnostic is asymptotically equivalent to the minimized KL divergence  
% $D_{KL}(\eta || T^* \pi)$ and should converge to zero when the computed map 
% converge to the theoritical true map.

test_z = randn(2,10000);

% Compute variance diagnostic
var_diag = variance_diagonostic(log_target,tri_map,test_z);

% Print final coeffs and objective
disp('==================')
disp(['Variance diagnostic: ',num2str(var_diag)])
disp('==================')
%% 
% 
%% Drawing samples from approximate posterior
% Once the transport map from reference to unnormalized posterior is estimated 
% it can be used to sample from the posterior to characterize the Bayesian inference 
% solution.

Znew = randn(2,5000);
colors = atan2(Znew(1,:),Znew(2,:));

Xpost = tri_map.Evaluate(Znew);

figure
subplot(1,2,1)
scatter(Xpost(1,:),Xpost(2,:),[],colors,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
xlabel('\theta_1')
ylabel('\theta_2')
xlim([-1,3.5])
axis('equal')
title('Approximate Posterior Samples')
subplot(1,2,2)
scatter(Znew(1,:),Znew(2,:),[],colors,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
xlabel('\theta_1')
ylabel('\theta_2')
title('Reference Samples')
axis('equal')
%% 
% Samples can then be used to compute quantity of interest with respect to parameters  
% $\mathbf{\theta}$. For example the sample mean:

X_mean = mean(Xpost,2);
disp('Mean a posteriori: ')
disp((X_mean))
%% 
% Samples can also be used to study parameters marginals. Here are the one-dimensional 
% marginals histograms:

figure
subplot(1,2,1)
histogram(Xpost(1,:),'Normalization','pdf')
xlabel('\theta_1')
ylabel('$\tilde{\pi}(\theta_1)$','Interpreter','Latex')
subplot(1,2,2)
histogram(Xpost(2,:),'Normalization','pdf')
xlabel('\theta_2')
ylabel('$\tilde{\pi}(\theta_2)$','Interpreter','Latex')
%% Custom functions needed for this example

function out=forward_model(p1,p2,t)
% Forward model
A = 0.4+0.4*(1+erf(p1/sqrt(2)));
B = 0.01+0.15*(1+erf(p2/sqrt(2)));
out = A.*(1-exp(-B*t));
end

function dout = grad_x_forward_model(p1,p2,t)
% Gradient of forward model w.r.t parameters
A = 0.4+0.4*(1+erf(p1/sqrt(2)));
B = 0.01+0.15*(1+erf(p2/sqrt(2)));
dAdx1 = 0.31954*exp(-0.5*p1.^2);
dBdx2 = 0.119683*exp(-0.5*p2.^2);
dOutdx1 = dAdx1.*(1-exp(-B*t));
dOutdx2 = t*A.*dBdx2.*exp(-t*B);
dout= [dOutdx1;dOutdx2];
end

function log_lkl=log_likelihood(std_noise,t,yobs,p1,p2)
%log-likelihood function
y = forward_model(p1,p2,t);
log_lkl = - log(sqrt(2*pi)*std_noise)-0.5*((y-yobs)/std_noise).^2;
end

function grad_x_lkl=grad_x_log_likelihood(std_noise,t,yobs,p1,p2)
%gradient log-likelihood function w.r.t parameters
y = forward_model(p1,p2,t);
dydx = grad_x_forward_model(p1,p2,t);
grad_x_lkl = (-1/std_noise.^2)*(y - yobs).*dydx;
end
function log_post=log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,p1,p2)
  log_prior1 = -log(sqrt(2*pi)*std_prior1)-0.5*(p1/std_prior1).^2;
  log_prior2 = -log(sqrt(2*pi)*std_prior2)-0.5*(p2/std_prior2).^2;
  log_post = log_prior1+log_prior2;
  for k=1:length(list_t)
    log_post = log_post + log_likelihood(std_noise,list_t(k),list_yobs(k),p1,p2);
  end
end

function grad_x_log_post=grad_x_log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,p1,p2)
  grad_x1_prior = -(1/std_prior1.^2)*(p1);
  grad_x2_prior = -(1/std_prior2.^2)*(p2);
  grad_x_prior = [grad_x1_prior;grad_x2_prior];
  grad_x_log_post = grad_x_prior;
  for k=1:length(list_t)
    grad_x_log_post = grad_x_log_post + grad_x_log_likelihood(std_noise,list_t(k),list_yobs(k),p1,p2);
  end
end


function [L,dwL]=objective(log_target,grad_x_log_target,coeffs,transport_map,x)
% KL divergence objective
num_points = size(x,2);
transport_map.SetCoeffs(coeffs);
map_of_x = transport_map.Evaluate(x);
%logpdf = log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,map_of_x(1,:),map_of_x(2,:));
logpdf = log_target(map_of_x);
log_det = transport_map.LogDeterminant(x);
L = - sum(logpdf + log_det')/num_points;
% Gradient of KL divergence objective
if (nargout > 1)
    %sens_vecs = grad_x_log_posterior(std_noise,std_prior1,std_prior2,list_t,list_yobs,map_of_x(1,:),map_of_x(2,:));
    sens_vecs = grad_x_log_target(map_of_x);
    grad_logpdf = transport_map.CoeffGrad(x,sens_vecs);
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x);
    dwL = - sum(grad_logpdf + grad_log_det,2)/num_points;
end
end

% variance diagnostic
function var = variance_diagonostic(log_target,tri_map,x)
    ref_logpdf = log(mvnpdf(x'));
    y = tri_map.Evaluate(x);
    pullback_logpdf = log_target(y) + tri_map.LogDeterminant(x)';
    diff = ref_logpdf' - pullback_logpdf;
    expect = mean(diff);
    var = 0.5*mean((diff-expect).^2);
end
function pdf=push_forward_pdf(tri_map,x)
    % pushforward density
    xinv = tri_map.Inverse(x,x);
    log_det_grad_x_inverse = - tri_map.LogDeterminant(xinv);
    log_pdf = log(mvnpdf(xinv'))+log_det_grad_x_inverse;
    pdf = exp(log_pdf);
end