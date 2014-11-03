function [logpost,dlogpost] = SEGammaHyperPosterior(hyp,x,y)

if nargin > 1 % data available: evaluate log likelihood
    [lik,dlik] = gp(hyp,@infExact,[],@covSEard,@likGauss,x,y);
else
    lik = 0; dlik.cov = []; dlik.lik = [];
end

% evaluate log prior
log_ell  = hyp.cov(1:end-1);
log_the2 = 2 * hyp.cov(end);
log_sig  = hyp.lik;

N = size(hyp.cov,1) - 1;
a = 0.05 * ones(N,1); % hyperhyp.cov.a_ell;
b = 100   * ones(N,1); % hyperhyp.cov.b_ell;
c = 0.01; % hyperhyp.cov.a_theta;
d = 100;   % hyperhyp.cov.b_theta;
s = 0.005; % hyperhyp.lik.s;
m = 10; % hyperhyp.lik.m;

% hyperprior
logprior = sum(log_ell .* a - exp(log_ell) ./ b) ... % gamma prior over length scales
    + log_the2 .* c - exp(log_the2) ./ d ...     % gamma prior over signal variance
    + log_sig .* s - exp(log_sig) ./ m;        % gamma prior over noise stddev

% gradient
dlogprior = [a - exp(log_ell) ./ b; ...
    2 * c - 2 * exp(log_the2) ./ d; ...
    s - exp(log_sig) ./ m];

% log posterior is log lik + log prior
logpost = lik - logprior;

dlogpost.cov = dlik.cov - dlogprior(1:end-1);
dlogpost.lik = dlik.lik - dlogprior(end);
dlogpost.mean = [];


%% unit test:
% N = 3;
% x.cov = randn(N+1,1);
% x.lik = randn();
% e = 1.0e-6; dt = zeros(N+1,1);
% [f,df] = SEGammaPrior(x); 
% for i = 1:N+1
%     y = x; y.cov(i) = y.cov(i) + e; f1 = SEGammaPrior(y);
%     y = x; y.cov(i) = y.cov(i) - e; f2 = SEGammaPrior(y);
%     dt(i) = (f1 - f2) / (2*e);
% end
% y = x; y.lik = y.lik + e; f1 = SEGammaPrior(y);
% y = x; y.lik = y.lik - e; f2 = SEGammaPrior(y);
% dt(N+2) = (f1 - f2) / (2*e);
% [df dt]