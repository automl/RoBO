function dH_fun = dH_MC_local(zbel,GP,logP,dlogPdM,dlogPdV,ddlogPdMdM,T,lmb,xmin,xmax,invertsign,LossFunc)
% this function constructs another function, which maps from locations in
% evaluation space to expected changes in the metric entropy of logP.
%
% Philipp Hennig, August 2011

% if nargin < 11
%     invertsign = false;
% end

W = randn(1,T);        % draw samples only once, to ensure smooth derivatives
%W = zeros(1,T);
%fprintf('dont forget to comment out the zeros!')


%H = - sum(exp(logP) .* (logP + lmb));                         % current Entropy 
L = GP_innovation_local(GP,zbel);                         % innovation function

dH_fun = @(x) dHdx_local(x,logP,dlogPdM,dlogPdV,ddlogPdMdM,lmb,W,L,xmin,xmax,invertsign,LossFunc,zbel);
end

function [dH,ddHdx] = dHdx_local(x,logP,dlogPdM,dlogPdV,ddlogPdMdM,lmb,W,L,xmin,xmax,invertsign,LossFunc,zbel)
if any(x<xmin) || any(x>xmax)
    dH    = eps;
    ddHdx = zeros(size(x,2),1);
    return;
end

if size(x,1) > 1; error 'dHdx_local is only for single x inputs'; end

N         = numel(logP);                        % number of locations in belief
D         = size(x,2);
T         = size(W,2);
[Lx,~  ]  = L(x);                                         % evaluate innovation
dMdx      = Lx;                                  % innovation function for mean
dVdx      = -Lx * Lx';                     % innovation function for covariance
dVdx      = dVdx(logical(triu(ones(N,N)))); 

dMM       = dMdx * dMdx';
trterm    = sum(sum(bsxfun(@times,ddlogPdMdM,reshape(dMM,[1,size(dMM)])),3),2);
detchange = dlogPdV * dVdx + 0.5 * trterm;       % deterministic part of change
stochange = (dlogPdM * dMdx) * W;                   % stochastic part of change
lPred     = bsxfun(@plus,logP + detchange,stochange);      % predicted new logP
lselP     = logsumexp(lPred,1);
lPred     = bsxfun(@minus,lPred,lselP);                             % normalise
%dHp       = -sum(exp(lPred) .* bsxfun(@plus,lPred,lmb),1) - H; % @minus? If you change it, change it above in H, too!
dHp       = feval(LossFunc{:},logP,lmb,lPred,zbel);
dH        = mean(dHp);

if invertsign; dH = -dH; end;

if ~isreal(dH); keyboard; end;

% numerical derivative, because the re-normalization makes analytical
% derivatives unstable.
if nargout > 1
    e     = 1.0e-5;
    ddHdx = zeros(D,1);
    for d = 1:D
        y = x; y(d) = y(d) + e;
        [Ly,~]    = L(y);                                         % evaluate innovation
        dMdy      = Ly;                                  % innovation function for mean
        dVdy      = -Ly * Ly';                     % innovation function for covariance
        dVdy      = dVdy(logical(triu(ones(N,N))));
        
        dMM       = dMdy * dMdy';
        trterm    = sum(sum(bsxfun(@times,ddlogPdMdM,reshape(dMM,[1,size(dMM)])),3),2);
        detchange = dlogPdV * dVdy + 0.5 * trterm;       % deterministic part of change
        stochange = (dlogPdM * dMdy) * W;                   % stochastic part of change
        lPred     = bsxfun(@plus,logP + detchange,stochange);      % predicted new logP
        lselP     = logsumexp(lPred,1);
        lPred     = bsxfun(@minus,lPred,lselP);                             % normalise
        %dHp       = -sum(exp(lPred) .* bsxfun(@plus,lPred,lmb),1) - H; % @minus? If you change it, change it above in H, too!
        dHp       = feval(LossFunc{:},logP,lmb,lPred,zbel);
        dHy1      = mean(dHp);
        
        y = x; y(d) = y(d) - e;
        [Ly,~]    = L(y);                                         % evaluate innovation
        dMdy      = Ly;                                  % innovation function for mean
        dVdy      = -Ly * Ly';                     % innovation function for covariance
        dVdy      = dVdy(logical(triu(ones(N,N))));
        
        dMM       = dMdy * dMdy';
        trterm    = sum(sum(bsxfun(@times,ddlogPdMdM,reshape(dMM,[1,size(dMM)])),3),2);
        detchange = dlogPdV * dVdy + 0.5 * trterm;       % deterministic part of change
        stochange = (dlogPdM * dMdy) * W;                   % stochastic part of change
        lPred     = bsxfun(@plus,logP + detchange,stochange);      % predicted new logP
        lselP     = logsumexp(lPred,1);
        lPred     = bsxfun(@minus,lPred,lselP);                             % normalise
        %dHp       = -sum(exp(lPred) .* bsxfun(@plus,lPred,lmb),1) - H; % @minus? If you change it, change it above in H, too!
        dHp       = feval(LossFunc{:},logP,lmb,lPred,zbel);
        dHy2      = mean(dHp);
        
        ddHdx(d)  = (dHy1 - dHy2) ./ (2*e);
        if invertsign; ddHdx = -ddHdx; end;
    end
end

% numerical derivative, because the re-normalization makes analytical
% derivatives unstable.
% if nargout > 1                                                      % derivative
%     LdL       = bsxfun(@times,Lx,reshape(dLx,[1,size(dLx)]));
%     LdL       = - LdL - permute(LdL,[2,1,3]);
%     ddVdxdx   = zeros(0.5 * N * (N+1),D);
%     for d = 1:D
%         tmp          = LdL(:,:,d);
%         ddVdxdx(:,d) = tmp(logical(triu(ones(N,N)))); % correct.
%     end
%     ddMdxdx   = dLx;    % correct.
%     dtrtermdx = etprod('pk',ddlogPdMdM,'pij',-LdL,'ijk'); % correct.
%     
%     dHpdx     = -(exp(lPred) .* bsxfun(@plus,1 + lPred,lmb))';% T x N % correct
%     dPdMdMdxx = bsxfun(@times,reshape(dlogPdM * ddMdxdx,[N,1,D]),reshape(W,[1,T,1])); % correct
%     ddetch    = dlogPdV * ddVdxdx + 0.5 * dtrtermdx;
%     keyboard;
%     normdet   = sum(exp(lPred) .* ddetch,1) % need further corrections here.:
%     dzlogpi/dx = dlogpi/dx - sum_j(exp(logpj) dlogpj/x / sum_j(exp(logpj));
%     ddetch    = bsxfun(@minus,ddetch,normdet);
%     dstoch    = bsxfun(@minus,dPdMdMdxx,sum(dPdMdMdxx,1));
%     ddHdxT    = dHpdx * (ddetch) + ...
%             reshape(sum(bsxfun(@times,reshape(dHpdx',[N,T,1]),dstoch),1),[T,D]);
%     % etprod('td',dHpdx,'ti',dPdMdMdxx,'itd'); 
%     ddHdx     = mean(ddHdxT,1)';
% end
end