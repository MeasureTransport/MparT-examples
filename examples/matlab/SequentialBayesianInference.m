clear
close all

rng(4)

addpath(genpath('~/Installations/MParT/matlab/'))
addpath(genpath('~/Documents/code/ATM/adaptivetransportmaps'))
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
atm_opts.maxSize = 20;
atm_opts.maxDegrees = MultiIndex([10,10]);

msets = [MultiIndexSet.CreateTotalOrder(1,1),MultiIndexSet.CreateTotalOrder(2,1)];

%% Minimization via ATM
[S] = AdaptiveTransportMap(msets,obj,atm_opts);


%% Normality test

LS = ComposedMap([L,S]);

X = LS.Evaluate(samples_train);

figure
hold on
plot(X(1,:),X(2,:),'.b')
axis equal

%% Observations
N=40;
List_obs=sigma_eff(sigma_w,sigma_i,2)+randn(N,1);


ref2 = IndependentProductDistribution(repmat({Normal()},1,2));
CM=load('offline_map_2_25.mat').CM;
L_M={CM.S{1}.S CM.S{2}.S};
yobs=List_obs(1);
PB_off=ComposedPullbackDensity(L_M,ref2);

lkl = LikelihoodFunction(PB_off, yobs);

xlin = linspace(1,3,100);

x_jt=[xlin', repmat(yobs,100,1)];

y = PB_off.evaluate(x_jt);
y2 = LS.Evaluate(x_jt');


Gy=LS.Gradient(x_jt',[zeros(1,100);ones(1,100)]);



DY = gradient(Y(1,:),X(1,:));


figure
plot(xlin,y(:,2))
hold on
plot(xlin,y2(2,:))
title('S(X)')

figure
plot(xlin,lkl.log(xlin'))
title('Log-likelihood')



function [out] = sigma_eff(sigma_w,sigma_i,z_iw)
    Rz_iw=1./sqrt(4*z_iw.^2+1);
    out=sigma_i.*(1-Rz_iw)+sigma_w.*(Rz_iw);
end

function [log] = LikelihoodFunction_M(S,yobs,X)   
    
    dobs = length(yobs);
    dtheta = S.outputDim - dobs;
    x_jt=[X;repmat(yobs,1,size(X,1))];
    
    SX = S.Evaluate(x_jt);
    Gy=LS.Gradient(x_jt,[zeros(dtheta,100);ones(dobs,100)]);
    log=normlogpdf(SX(end-dobs:end,:));
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
