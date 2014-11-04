function [logP,dlogPdMu,dlogPdSigma,dlogPdMudMu,logPf] = joint_min(Mu,Sigma,silent)
% distribution (discrete) over elements x, over which Mu,Sigma define a
% joint Gaussian, being the smallest of all x, respectively.

% addpath './logsumexp/';
if ismac
    addpath '~/Documents/MATLAB/control/code/util/tprod/'
elseif isunix
    addpath '~/Documents/MATLAB/tprod'
end

if ~exist('silent','var')
  silent = 0;
  fprintf 'evaluating p(x_min)'
end

gam   = 1;

D     = length(Mu);
logP  = zeros(D,1);
logPf = zeros(D,2); % mu(x|x=xmin) , sigma2(x|x=xmin);

if nargout == 1        % no derivatives required
    for k = 1 : D
        %if mod(k,10) == 0; disp(['dimension ', num2str(k)]); end
        if ~silent
          if mod(k,10) == 0; fprintf('#'); end
        end
        logP(k) = min_factor(Mu,Sigma,k,gam);
    end
    logP(isinf(logP)) = -500;
    logP  = logP - logsumexp(logP);
    
elseif nargout < 5     % with derivatives
    dlogPdMu    = zeros(D,D);
    dlogPdSigma = zeros(D,0.5 * D * (D+1));
    dlogPdMudMu= zeros(D,D,D);
    for k = 1 : D
        %if mod(k,10) == 0; disp(['dimension ', num2str(k)]); end
        if ~silent
          if mod(k,10) == 0; fprintf('#'); end
        end
        %[logP(k),dlPdM,dlPdS,dlPdMdM] = min_factor_with_messages(Mu,Sigma,k,gam);
        [logP(k),dlPdM,dlPdS,dlPdMdM] = min_factor(Mu,Sigma,k,gam);
        dlogPdMu(k,:)    = dlPdM';
        dlogPdSigma(k,:) = dlPdS';
        dlogPdMudMu(k,:,:) = dlPdMdM;
    end
        
    logP(isinf(logP)) = -500;
    % re-normalize at the end, to smooth out numerical imbalances:
    logPold        = logP;
    dlogPdMuold    = dlogPdMu;
    dlogPdSigmaold = dlogPdSigma;
    dlogPdMudMuold = dlogPdMudMu;

    Z     = sum(exp(logPold));
    logP  = logP - logsumexp(logP);

    % adjust derivatives, too. This is a bit tedious.
    Zm    = sum(bsxfun(@times,exp(logPold),dlogPdMuold)) ./ Z; 
    Zs    = sum(bsxfun(@times,exp(logPold),dlogPdSigmaold)) ./ Z; 

    dlogPdMu    = bsxfun(@minus,dlogPdMuold,Zm);
    dlogPdSigma = bsxfun(@minus,dlogPdSigmaold,Zs);

    %ff   = dlogPdMuold' * diag(exp(logPold)) * dlogPdMuold ./ Z;    
    %gg   = etprod('ij',dlogPdMudMuold,'kij',exp(logPold),'k') ./ Z;
    ff   = etprod('kij',dlogPdMuold,'ki',dlogPdMuold,'kj');
    gg   = etprod('ij',dlogPdMudMuold+ff,'kij',exp(logPold),'k') ./ Z;
    Zij  = Zm' * Zm;
    adds = reshape(-gg+Zij,[1,D,D]);

    dlogPdMudMu = bsxfun(@plus,dlogPdMudMuold,adds);
elseif nargout > 4 % with belief over value of minimum
    dlogPdMu    = zeros(D,D);
    dlogPdSigma = zeros(D,0.5 * D * (D+1));
    dlogPdMudMu= zeros(D,D,D);
    for k = 1 : D
        %if mod(k,10) == 0; disp(['dimension ', num2str(k)]); end
        if ~silent
          if mod(k,10) == 0; fprintf('#'); end
        end
        %[logP(k),dlPdM,dlPdS,dlPdMdM] = min_factor_with_messages(Mu,Sigma,k,gam);
        [logP(k),dlPdM,dlPdS,dlPdMdM,logPf(k,:)] = min_factor(Mu,Sigma,k,gam);
        dlogPdMu(k,:)    = dlPdM';
        dlogPdSigma(k,:) = dlPdS';
        dlogPdMudMu(k,:,:) = dlPdMdM;
    end
        
    logP(isinf(logP)) = -500;
    % re-normalize at the end, to smooth out numerical imbalances:
    logPold        = logP;
    dlogPdMuold    = dlogPdMu;
    dlogPdSigmaold = dlogPdSigma;
    dlogPdMudMuold = dlogPdMudMu;

    Z     = sum(exp(logPold));
    logP  = logP - logsumexp(logP);

    % adjust derivatives, too. This is a bit tedious.
    Zm    = sum(bsxfun(@times,exp(logPold),dlogPdMuold)) ./ Z; 
    Zs    = sum(bsxfun(@times,exp(logPold),dlogPdSigmaold)) ./ Z; 

    dlogPdMu    = bsxfun(@minus,dlogPdMuold,Zm);
    dlogPdSigma = bsxfun(@minus,dlogPdSigmaold,Zs);

    %ff   = dlogPdMuold' * diag(exp(logPold)) * dlogPdMuold ./ Z;    
    %gg   = etprod('ij',dlogPdMudMuold,'kij',exp(logPold),'k') ./ Z;
    ff   = etprod('kij',dlogPdMuold,'ki',dlogPdMuold,'kj');
    gg   = etprod('ij',dlogPdMudMuold+ff,'kij',exp(logPold),'k') ./ Z;
    Zij  = Zm' * Zm;
    adds = reshape(-gg+Zij,[1,D,D]);

    dlogPdMudMu = bsxfun(@plus,dlogPdMudMuold,adds);   
end

if(any(isnan(logP))); keyboard; end
