function innovation_function = GP_innovation_local(GP,zbel)
% what is the belief over zbel, und how does it change when we evaluate anywhere
% in zeval?
%
% Philipp Hennig & Christian Schuler, August 2011

if GP.poly >=0 || GP.deriv
    error 'current implementation is for standard GPs only. No fancy stuff yet'.
end

K   = feval(GP.covfunc{:},GP.hyp.cov,GP.x) + exp(2*GP.hyp.lik) * eye(size(GP.x,1));
cK  = chol(K);
kbX = feval(GP.covfunc{:},GP.hyp.cov,zbel,GP.x);

innovation_function = @(x) efficient_innovation(x,cK,kbX,GP,zbel);

end

function [Lx,dLxdx] = efficient_innovation(x,cK,kbX,GP,zbel)
if size(x,1) > 1 
    error 'this function is for only single data-point evaluations.'
end

if isempty(GP.x)
    % kernel values
    kbx  = feval(GP.covfunc{:},GP.hyp.cov,zbel,x);
    kxx  = feval(GP.covfunc{:},GP.hyp.cov,x);
    
    % derivatives of kernel values
    dkxb = feval(GP.covfunc_dx{:},GP.hyp.cov,x,zbel);
    dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);
    
    dkxb = reshape(dkxb,[size(dkxb,2),size(dkxb,3)]);
    dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);
    
    % terms of the innovation
    sloc   = sqrt(kxx);
    proj   = kbx;
    
    dvloc  = dkxx;
    dproj  = dkxb;
    
    % innovation, and its derivative
    Lx     = proj ./ sloc;
    dLxdx  = dproj ./ sloc - 0.5 * bsxfun(@times,proj,dvloc) ./ sloc.^3;
    return
end

% kernel values
kbx  = feval(GP.covfunc{:},GP.hyp.cov,zbel,x);
kXx  = feval(GP.covfunc{:},GP.hyp.cov,GP.x,x);
kxx  = feval(GP.covfunc{:},GP.hyp.cov,x) + exp(GP.hyp.lik * 2);

% derivatives of kernel values
dkxb = feval(GP.covfunc_dx{:},GP.hyp.cov,x,zbel);
dkxX = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x);
dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);

dkxb = reshape(dkxb,[size(dkxb,2),size(dkxb,3)]);
dkxX = reshape(dkxX,[size(dkxX,2),size(dkxX,3)]);
dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);

% terms of the innovation
sloc   = sqrt(kxx - kXx' * (cK \ (cK' \ kXx)));
proj   = kbx - kbX * (cK \ (cK' \ kXx));

dvloc  = (dkxx' - 2 * dkxX' * (cK \ (cK' \ kXx)))';
dproj  = dkxb - kbX * (cK \ (cK' \ dkxX));

% innovation, and its derivative
Lx     = proj ./ sloc;
dLxdx  = dproj ./ sloc - 0.5 * bsxfun(@times,proj,dvloc) ./ sloc.^3;
end