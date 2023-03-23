clear
close all

addpath('~/Installations/MParT/matlab/')
KokkosInitialize(8)

sigma_w=2500;
sigma_i=22;

mu0 = 2;
std0 = 0.25;

N=20000;
X = mu0 + std0*randn(1,20000);

std_m = 63;
Y = sigma_eff(sigma_w,sigma_i,X)+std_m*randn(1,20000);

samples = [X;Y];

figure
plot(samples(1,:),samples(2,:),'r.')

%% Split training/test samples
n_train = floor(N*0.8);
samples_train = samples(:,1:n_train);
samples_test = samples(:,n_train+1:end);

%% Standardization
C = cov(samples');
A = inv(chol(C,"lower"));
c = -1 * A * mean(samples,2);

L = AffineMap(A,c);

samples_train_st = L.Evaluate(samples_train);
samples_test_st = L.Evaluate(samples_test);

figure
plot(samples_train_st(1,:),samples_train_st(2,:),'g.')

%% Definition of the minimization problem
obj = GaussianKLObjective(samples_train_st,samples_test_st,2);
atm_opts = ATMOptions();
atm_opts.basisLB = -3;
atm_opts.basisUB = 3;
atm_opts.verbose = 1;
atm_opts.maxSize = 3;

msets = [MultiIndexSet.CreateTotalOrder(1,1), MultiIndexSet.CreateTotalOrder(2,1)];

%% Minimization via ATM

[S] = AdaptiveTransportMap(msets,obj,atm_opts);



function [out] = sigma_eff(sigma_w,sigma_i,z_iw)
    Rz_iw=1./sqrt(4*z_iw.^2+1);
    out=sigma_i.*(1-Rz_iw)+sigma_w.*(Rz_iw);
end