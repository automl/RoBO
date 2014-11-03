function K = k_matrix_full(GP,x,z)
% Kernel gram matrix, with outputs for the derivatives
switch nargin
  case 2
    N = size(x,1);
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
    if isempty(z) || isempty(x)
      K = zeros(size(z,1),size(x,1));
    elseif ~GP.deriv
      K = (feval(GP.covfunc{:},GP.hyp.cov,x,z))';
    else
      K = feval(GP.covfunc{:},GP.hyp.cov,x,z);
      K_dx = feval(GP.covfunc_dx{:},GP.hyp.cov,x,z);
      K_dx2 = feval(GP.covfunc_dxdz{:},GP.hyp.cov,x,z);
      K = [K    -K_dx;  % why the minus? This is a special aspect of SE?
           K_dx K_dx2]'; % why no transpose on the K? So size(K(x,y)) = [size(y),size(x)] ? That seems odd.
    end
end
end

