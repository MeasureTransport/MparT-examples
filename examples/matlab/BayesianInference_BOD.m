%% Characterization of Bayesian posterior density
% The objective of this example is to demonstrate how transport maps can be 
% used to represent posterior distribution of Bayesian inference problems.
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
%% Application with the Biochemical Oxygen Demand (BOD) model from <http://%3Chttps//or.water.usgs.gov/proj/keno_reach/download/chemgeo_bod_final.pdf Sullivan et al., 2010>
%% Definition
% To illustrate the process describe above, we consider the BOD inverse problem 
% described in <http://%3Chttps//arxiv.org/pdf/1602.05023.pdf Marzouk et al., 
% 2016>. The goal is to estimate $2$ coefficients in a time-dependent model of 
% oxygen demand, which is used as an indication of biological activity in a water 
% sample.
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
% Definition of the forward model and gradient with respect to $\mathbf{\theta}$:

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
% One simulation of the forward model:

% code 

% code 

% code 

% code 

% code 

%% 
% For this problem, as noise $\mathcal{E}$ is Gaussian and additive, the likelihood 
% function $\pi(\mathbf{y}|\mathbf{\theta})$, can be decomposed for each time 
% step as:
% 
% $$\pi(\mathbf{y}|\mathbf{\theta}) = \prod_{t}^{5} \pi(y_t|\mathbf{\theta}), 
% $$
% 
% where
% 
% $$\pi(\mathbf{y}_t|\mathbf{\theta})=\frac{1}{\sqrt{0.002.\pi}}\exp \left(-\frac{1}{0.002} 
% \left(y_t - \mathcal{B}(t)\right)^2 \right), t \in \{1,...,5\}.$$
% 
% Likelihood function and its gradient with respect to parameters:
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
% We can then define the unnormalized posterior and its gradient with respect 
% to parameters:

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
%% Observations
% We consider the following realization of observation $\mathbf{y}$:

% code 

% code 

% code 

% code 

% code
%% 
% 
%% Visualization of the *unnormalized* posterior density

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
%Draw reference samples to define objective

% code 

% code
%% Map parametrization
% We use the MParT function |CreateTriangular| to directly create a transport 
% map object of dimension with given total order parameterization.
% 
% Create transport map

% code 

% code 

% code
%% Optimization

% code 

% code
%% Accuracy checks
%% Comparing density contours
% Comparison between contours of the posterior $\pi(\mathbf{\theta}|\mathbf{y})$ 
% and conoturs of pushforward density $T_* \eta$. Pushforward distribution

% code 

% code 

% code 

% code 

% code
%% 
% Reference distribution

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

% code 
% Compute variance diagnostic

% code 
% Print final coeffs and objective

% code 

% code 

% code
%% 
% 
%% Drawing samples from approximate posterior
% Once the transport map from reference to unnormalized posterior is estimated 
% it can be used to sample from the posterior to characterize the Bayesian inference 
% solution.

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
% Samples can then be used to compute quantity of interest with respect to parameters  
% $\mathbf{\theta}$. For example the sample mean:

% code 

% code 
% Samples can also be used to study parameters marginas. Here are the one-dimensional marginals histograms:

% code 

% code 

% code 

% code 

% code 

% code 

% code 

% code 
matlab.internal.liveeditor.openAndConvert('BayesianInference_BOD.mlx','BayesianInference_BOD.m')