%% Transport Map from density
%
% The objective of this example is to show how a transport map can be build 
% in MParT when the the unnormalized probability density function of the target 
% density is known.
%
% _The rendered live script version of this example can be obtain by doing "Open 
% as Live Script” from the Current Folder Browser or document tab or by doing 
% "Save as ..." and select the "MATLAB Live Code Files (*.mlx)" format._
%
%% Problem description
% We consider $T(\mathbf{z};\mathbf{w})$ a monotone triangular transport map 
% parameterized by $\mathbf{w}$ (e.g., polynomial coefficients). This map which 
% is invertible and has an invertible Jacobian for any parameter $\mathbf{w}$, 
% transports samples $\mathbf{z}^i$ from the reference density $\eta$ to samples  
% $T(\mathbf{z}^i;\mathbf{w})$ from the map induced density $\tilde{\pi}_\mathbf{w}(\mathbf{z})$ 
% defined as: $$ \tilde{\pi}_\mathbf{w}(\mathbf{z}) = \eta(T^{-1}(\mathbf{z};\mathbf{w}))|\text{det  
% } T^{-1}(\mathbf{z};\mathbf{w})|,$$ where $\text{det } T^{-1}$ is the determinant 
% of the inverse map Jacobian at the point $\mathbf{z}$. We refer to $\tilde{\pi}_{\mathbf{w}}(\mathbf{x})$ 
% as the *map-induced* density or *pushforward distribution* and will commonly 
% interchange notation for densities and measures to use the notation $\tilde{\pi}  
% = T_{*} \eta$.
% 
% The objective of this example is, knowing some unnormalized target density  
% $\bar{\pi}$, find the map $T$ that transport samples drawn from $\eta$ to samples 
% drawn from the target $\pi$.
%% Imports
% First, import |MParT| by adding the path to the installation folder and initialize 
% the |Kokkos| environment. Note that it is possible to specify the number of 
% threads used by |MParT| as an argument of the |KokkosInitialize| function. The 
% number of threads can only be set once per session.

addpath(genpath('your/MParT/install/path'))
num_threads = 8;
KokkosInitialize(num_threads);
%% 
% Default settings:

sd = 3; rng(sd);

set(0,'DefaultLineLineWidth',1.75)
set(0,'defaultAxesFontSize',12)
set(0,'defaultfigurecolor',[1 1 1])
set(0, 'DefaultAxesBox', 'on');
%% Target density and exact map
% In this example we use a 2D target density known as the _banana_ density where 
% the unnormalized probability density, samples and the exact transport map are 
% known.
% 
% The banana density is defined as: $$ \pi(x_1,x_2) \propto N_1(x_1)\times N_1(x_2-x_1^2)  
% $$ where $N_1$ is the 1D standard normal density.
% 
% The exact transport map that transport the 2D standard normal density to $\pi$ 
% is known as: $$$
% 
% $${T}^\text{true}(z_1,z_2)= \left[ \matrix{z_1 \cr z_2 + z_1^2 } \right]$$
% 
% Contours of the target density can be visualized as:

 % Grid for plotting 

ngrid=100;
x1_t = linspace(-3,3,ngrid);
x2_t = linspace(-3,7.5,ngrid);
[xx1,xx2] = meshgrid(x1_t,x2_t);

xx = [xx1(:)';xx2(:)'];

% Target contours
target_pdf_at_grid = exp(target_logpdf(xx));

figure
contour(xx1,xx2,reshape(target_pdf_at_grid,ngrid,ngrid))
xlabel('x_1')
ylabel('x_2')
legend('target density')
%% Map training
%% Defining objective function and its gradient
% Knowing the closed form of the unnormalized target density $\bar{\pi}$, the 
% objective is to find a map-induced density $\tilde{\pi}_{\mathbf{w}}(\mathbf{z})$ 
% that is a good approximation of the target $\pi$.
% 
% In order to characterize this posterior density, one method is to build a 
% monotone triangular transport map $T$ such that the KL divergence $D_{KL}(\eta  
% || T^* \pi)$ is minmized. If $T$ is map parameterized by $\mathbf{w}$, the objective 
% function derived from the discrete KL divergence reads:
% 
% $$J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right) 
% + \log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right), \,\,\, 
% \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d),$$
% 
% where $T$ is the transport map pushing forward the standard normal $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ 
% to the target density $\pi(\mathbf{z})$. The gradient of this objective function 
% reads
% 
% $$ \nabla_\mathbf{w} J(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N \left( \nabla_\mathbf{w}  
% T(\mathbf{z}^i;\mathbf{w}).\nabla_\mathbf{x}\log\pi\left(T(\mathbf{z}^i;\mathbf{w})\right)  
% + \nabla_{\mathbf{w}}\log  \text{det }\nabla_\mathbf{z} T(\mathbf{z}^i;\mathbf{w})\right),  
% \,\,\, \mathbf{z}^i \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d). $$ The objective 
% function and gradient are defined at the end of the script.
%% Map parameterization
% For the parameterization of $T$ we use a total order multivariate expansion 
% of hermite functions. Knowing $T^\text{true}$, any parameterization with total 
% order greater than one will include the true solution of the map finding problem.

% Set-up first component and initialize map coefficients
map_options = MapOptions;

total_order = 2;
 
% Create dimension 2 triangular map
transport_map = CreateTriangular(2,2,total_order,map_options);
%% 
% 
%% Approximation before optimization
% Coefficients of triangular map are set to 0 upon creation.

% Make reference samples for training
num_points = 10000;
z = randn(2,num_points);

% Make reference samples for testing
test_z = randn(2,5000);

% Pushed samples
x = transport_map.Evaluate(test_z);

% Before optimization plot
figure
hold on
contour(xx1,xx2,reshape(target_pdf_at_grid,ngrid,ngrid))
scatter(x(1,:),x(2,:),'MarkerFaceColor',[0 0.4470 0.7410],'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
legend('Unnormalizerd target','Pushed samples')
%% 
% At initialization, samples are "far" from being distributed according to the 
% banana distribution. 
% 
% Initial objective and coefficients: 

% Print initial coeffs and objective
obj_init = objective(transport_map.CoeffMap(), transport_map, test_z);
disp('==================')
disp('Starting coeffs')
disp(transport_map.CoeffMap());
disp(['Initial objective value: ',num2str(obj_init)])
disp('==================')
%% Minimization

disp('==================')

% Optimize
obj = @(w) objective(w,transport_map,z);
w0 = transport_map.Coeffs();

options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'none');
[~] = fminunc(obj, w0, options);

% Print final coeffs and objective
obj_final = objective(transport_map.CoeffMap(), transport_map, test_z);
disp('==================')
disp('Final coeffs')
disp(transport_map.CoeffMap());
disp(['Final objective value: ',num2str(obj_final)])
disp('==================')
%% 
% 
%% Approximation after optimization
%% Pushed samples

% Pushed samples
x = transport_map.Evaluate(test_z);

% After optimization plot
figure
hold on
contour(xx1,xx2,reshape(target_pdf_at_grid,ngrid,ngrid))
scatter(x(1,:),x(2,:),'MarkerFaceColor',[0 0.4470 0.7410],'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
xlabel('x_1')
ylabel('x_2')
legend('Unnormalizerd target','Pushed samples')
%% 
% After optimization, pushed samples $T(z^i)$, $z^i \sim \mathcal{N}(0,I)$ are 
% approximately distributed according to the target $\pi$
%% Variance diagnostic
% A commonly used accuracy check when facing computation maps from density is 
% the so-called variance diagnostic defined as:
% 
% $$$$$$$\epsilon_\sigma = \frac{1}{2} \mathbb{V}\text{ar}_\rho \left[ \log 
% \frac{\rho}{T^* \bar{\pi}} \right]$$
% 
% This diagnostic is asymptotically equivalent to the minimized KL divergence  
% $D_{KL}(\eta || T^* \pi)$ and should converge to zero when the computed map 
% converge to the true map. 

% Compute variance diagnostic
var_diag = variance_diagonostic(transport_map,test_z);

% Print variance diagnostic
disp('==================')
disp(['Variance diagnostic: ',num2str(var_diag)])
disp('==================')
%% Pushforward density
% We can also plot the contour of the unnormalized density $\bar{\pi}$ and the 
% pushforward approximation $T_* \eta$:

map_approx_grid = push_forward_pdf(transport_map,xx);

figure
hold on
contour(xx1,xx2,reshape(target_pdf_at_grid,ngrid,ngrid))
contour(xx1,xx2,reshape(map_approx_grid,ngrid,ngrid),'LineStyle','--')
xlabel('x_1')
ylabel('x_2')
legend('Unnormalized target','TM approximation')

%% Custom functions needed for this example

function logpdf = target_logpdf(x)
% definition of the banana unnormalized density
logpdf1 = -0.5*log(2*pi)-0.5*x(1,:).^2;
y2 = (x(2,:)-x(1,:).^2);
logpdf2 = -0.5*log(2*pi)-0.5*y2.^2;
logpdf = logpdf1 + logpdf2;
end
% Gradient of unnomalized target density required for gradient objective
function grad_logpdf = target_grad_logpdf(x)
  grad1 = -x(1,:) + (2*x(1,:).*(x(2,:)-x(1,:).^2));
  grad2 = (x(1,:).^2-x(2,:));
  grad_logpdf = [grad1;grad2];
end
function [L,dwL]=objective(coeffs,transport_map,x)
% KL divergence objective
num_points = size(x,2);
transport_map.SetCoeffs(coeffs);
map_of_x = transport_map.Evaluate(x);
logpdf = target_logpdf(map_of_x);
log_det = transport_map.LogDeterminant(x);

L = - sum(logpdf + log_det')/num_points;

% Gradient of KL divergence objective
if (nargout > 1)
    sens_vecs = target_grad_logpdf(map_of_x);
    grad_logpdf = transport_map.CoeffGrad(x,sens_vecs);
    grad_log_det = transport_map.LogDeterminantCoeffGrad(x);
    dwL = - sum(grad_logpdf + grad_log_det,2)/num_points;
end
end
% variance diagnostic
function var = variance_diagonostic(tri_map,x)
    ref_logpdf = log(mvnpdf(x'));
    y = tri_map.Evaluate(x);
    pullback_logpdf = target_logpdf(y) + tri_map.LogDeterminant(x)';
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