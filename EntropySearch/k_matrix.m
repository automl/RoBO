function K = k_matrix(GP,x,z)
% Kernel gram matrix, without outputs for the derivatives

switch nargin
  case 2
    N = numel(x);
    if ~GP.deriv
      K = feval(GP.covfunc{:},GP.hyp.cov,x); % + exp(2*GP.hyp.lik)*eye(N);
    else
      K = feval(GP.covfunc{:},GP.hyp.cov,x);
      K_dx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);
      K_dx2 = feval(GP.covfunc_dxdz{:},GP.hyp.cov,x);
      K = [K    -K_dx;
           K_dx K_dx2]'; % + exp(2*GP.hyp.lik)*eye(2*N);
    end
  case 3
    if isempty(z)
      K = [];
    elseif ~GP.deriv
      K = (feval(GP.covfunc{:},GP.hyp.cov,x,z))';
    else
      K = feval(GP.covfunc{:},GP.hyp.cov,x,z);
      K_dx = feval(GP.covfunc_dx{:},GP.hyp.cov,x,z);
      K = [K; K_dx]';
    end
end
end

