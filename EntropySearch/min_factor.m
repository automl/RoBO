function [logZ,dlogZdMu,dlogZdSigma,dlogZdMudMu,mvmin,dMdMu,dMdSigma,dVdSigma] = min_factor(Mu,Sigma,k,gam,MPrec)
% probability for the k-th element to be the smallest of the D elements
% over which Mu,Sigma, define a joint Gaussian probability
% also returns Gaussian approximation for the marginal distribution of
% x if it is in fact the minimum. (mean, variance)
%
% Philipp Hennig, June 2011

D = length(Mu);

if nargin < 4; gam = 1; end

% messages (in natural parameters):
logS = zeros(D-1,1); % normalization constant (determines zeroth moment)
MP   = zeros(D-1,1); % mean times precision (determines first moment)
P    = zeros(D-1,1); % precision (determines second moment)    
% marginal:
M    = Mu;
V    = Sigma;

%% EP iteration
count = 0;
while true
    count = count + 1;
    Diff = 0;
    for i = 1:D - 1
        if i < k; l = i; else l = i + 1; end % which comparison?
        [M,V,P(i),MP(i),logS(i),diff] = lt_factor(k,l,M,V,MP(i),P(i),gam);
        if isnan(diff); break; end % found impossible combination
        Diff    = Diff + abs(diff);
    end
    if isnan(diff); break; end
    if count > 50
        disp('EP iteration ran over iteration limit. Stopped.')
        break;
    end
    if Diff < 1.0e-3
        %disp(num2str(count));
        break;
    end
end

% global Mo Vo Po MPo logSo
% keyboard
% Mo = M; Vo = V; Po = P; MPo = MP; logSo = logS;

if isnan(diff)
    logZ  = -inf;
    dlogZdMu = zeros(D,1);
    dlogZdSigma = zeros(0.5*(D*(D+1)),1);
    dlogZdMudMu = zeros(D,D);
    mvmin = [Mu(k),Sigma(k,k)];
    dMdMu = zeros(1,D);
    dMdSigma = zeros(1,0.5*(D*(D+1)));
    dVdSigma = zeros(1,0.5*(D*(D+1)));
else
  %% evaluate log Z:
  C = eye(D) ./ sqrt(2); C(k,:) = -1/sqrt(2); C(:,k) = [];
  R       = bsxfun(@times,sqrt(P'),C);
  r       = sum(bsxfun(@times,MP',C),2);
  mpm     = MP.* MP ./ P;
  mpm(MP==0) = 0;
  mpm     = sum(mpm);
  s       = sum(logS);

  IRSR    = (eye(D-1) + R' * Sigma * R);
  rSr     = r' * Sigma * r;
  A       = R * (IRSR \ R');
  A       = 0.5 * (A' + A); % ensure symmetry.
  b       = (Mu + Sigma * r);
  Ab      = A * b;
  dts     = logdet(IRSR);
  logZ    = 0.5 * (rSr - b' * Ab - dts) + Mu' * r + s - 0.5 * mpm;
  if(logZ == inf); keyboard; end

  %% derivatives of log Z, if necessary
  if nargout > 2

      btA = b' * A;

      dlogZdMu    = r - Ab;
      dlogZdMudMu = -A;
      
%       dlogZdSigma = zeros(0.5*(D*(D+1)),1);    
%       i = 0;
%       for col = 1:D
%           for row = 1:col
%               i = i + 1;
%   %             if col == row; trAS = A(row,row); else trAS = A(col,row); end
%   %             % note that A is not rectangular!
%   %             dlogZdSigma(i) = -0.5 * trAS - r(row) * Ab(col) ...
%   %                 + 0.5 * btA(row) * Ab(col) + 0.5 * r(row) * r(col);
%               if col == row
%                  dlogZdSigma(i) = -0.5 * A(row,row) - r(row) * Ab(col) ...
%                      + 0.5 * btA(row) * Ab(col) + 0.5 * r(row) * r(col);               
%               else
%                  dlogZdSigma(i) = -0.5 * (A(col,row) + A(row,col)) - r(row) * Ab(col) - r(col) * Ab(row) ...
%                      + 0.5 * b' * A(:,row) * A(col,:) * b + 0.5 * b' * A(:,row) * A(col,:) * b + r(row) * r(col);                
%               end
%           end
%       end
      
      dlogZdSigma = -A - 2*r*Ab' + r*r' + btA'*Ab';
      dlogZdSigma = 0.5*(dlogZdSigma+dlogZdSigma'-diag(diag(dlogZdSigma)));
      dlogZdSigma = dlogZdSigma(logical(triu(ones(D,D))));
  end
  % dS(logical(triu(ones(D,D)))) = dlogZdSigma;
  % dS = dS' + dS - diag(diag(dS));
  
  if nargout > 4
      if nargin < 5;
          MPrec = Sigma \ Mu;
      end
      % V = Sigma - Sigma * A * Sigma;
      % M = V * (MPrec + r);
      mvmin = [M(k),V(k,k)];
      if nargout > 5          
          VSigmaI = eye(D) - Sigma * R * (IRSR \ R');
          dMdMu = VSigmaI(k,:);
          
          VSIVSI   = etprod('ijkl',VSigmaI,'ki',VSigmaI,'lj');
          
          dVdSigma = VSIVSI(:,:,k,k);
          %dVdSigma = VSigmaI(k,:)' * VSigmaI(k,:);
          dVdSigma = dVdSigma(logical(triu(ones(D,D))))';
          
          dMdSigma = etprod('ijk',VSIVSI,'ijkl',MPrec + r,'l');
          dMdSigma = dMdSigma(:,:,k) - VSigmaI(k,:)' * MPrec';
          dMdSigma = dMdSigma(logical(triu(ones(D,D))))';
      end
  end
end
