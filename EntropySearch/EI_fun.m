function fx = EI_fun(GP,xmin,xmax,invertsign)
% returns a function, which returns the expected improvement and its derivative,
% with respect to x. (EI has a surprisingly simple derivative).
%
% Philipp Hennig, August 2011

if nargin < 4
    invertsign = false;
end

if GP.deriv; 
    error 'this code is for GPs without derivative observations only.'; 
end

if ~isempty(GP.x)
    alpha = GP.cK \ (GP.cK' \ GP.y);
    m     = feval(GP.covfunc{:},GP.hyp.cov,GP.x) * alpha;
    fmin  = min(m);
else
    alpha = [];
    fmin  = +Inf;
end

fx    = @(x) EI_f(x,alpha,GP,fmin,xmin,xmax,invertsign);
end

function [f,df] = EI_f(x,alpha,GP,fm,xmin,xmax,invertsign)
if invertsign; sign = -1; else sign = 1; end
D = size(x,2);
if size(x,1) > 1; error 'only single inputs allowed'; end;
if any(x<xmin) || any(x>xmax)
    f  = 0;
    df = zeros(size(x,2),1);
%     fprintf 'projecting back. check that this works.'
%     df = (x < xmin) - (x > xmax); % project back towards good regions.
%     df = sign * df';
    return;
end

if isempty(GP.x)
    kxx  = feval(GP.covfunc{:},GP.hyp.cov,x);
    dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);
    dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);
    
    s    = sqrt(kxx);
    dsdx = 0.5 / s * dkxx';
    
    z    = fm / s;                          % assuming zero mean
    
    phi   = exp(-0.5 * z.^2) ./ sqrt(2*pi); % faster than Matlabs normpdf
    Phi   = 0.5 * erfc(-z ./ sqrt(2));      % faster than Matlabs normcdf

    
    f  = sign * (fm * Phi + s * phi);
    df = sign * (dsdx * phi);
    return;
end

% kernel values
kXx  = feval(GP.covfunc{:},GP.hyp.cov,GP.x,x);
kxx  = feval(GP.covfunc{:},GP.hyp.cov,x);

% derivatives of kernel values
dkxX = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x);
dkxx = feval(GP.covfunc_dx{:},GP.hyp.cov,x);

dkxX = reshape(dkxX,[size(dkxX,2),size(dkxX,3)]);
dkxx = reshape(dkxx,[size(dkxx,2),size(dkxx,3)]);

m    = kXx' * alpha;
dmdx = (dkxX' * alpha);
s    = sqrt(kxx - kXx' * (GP.cK \ (GP.cK' \ kXx)));
dsdx = zeros(D,1);
for d = 1:D
    dsdx(d) = 0.5 / s * (dkxx(1,d) - 2 * dkxX(:,d)' * (GP.cK \ (GP.cK' \ kXx)));
end

z    = (fm - m) ./ s;

phi   = exp(-0.5 * z.^2) ./ sqrt(2*pi); % faster than Matlabs normpdf
Phi   = 0.5 * erfc(-z ./ sqrt(2));      % faster than Matlabs normcdf

f  = sign * ((fm - m) * Phi + s * phi);
df = sign * (-dmdx    * Phi + dsdx * phi);

if sign * f < 0 
    f  = 0;
    df = zeros(size(x,2),1);
end
end