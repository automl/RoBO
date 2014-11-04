function K = covSEard_dx_MD(hyp, x, z)
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sqrt(sf2)) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(diag(1./ell)*x');
  else                                                   % cross covariances Kxz
    K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
  end
end
K0 = sf2*exp(-K/2);

if xeqz
    p = reshape(x',[1,size(x,2),size(x,1)]);
else
    p = reshape(z',[1,size(z,2),size(z,1)]);
end
D_idj = bsxfun(@minus,x,p);
T_idj = bsxfun(@rdivide,D_idj,reshape(ell.^2,[1,D,1]));
K     = -bsxfun(@times,T_idj,reshape(K0,[size(K0,1),1,size(K0,2)]));
K     = permute(K,[1,3,2]);