function [Mnew,Vnew,pnew,mpnew,logS,d] = lt_factor(s,l,M,V,mp,p,gam) 
% 'less-than' factor. c_i = delta_{il} - delta_{is}

persistent sq2;
sq2 = sqrt(2);

% rank 1 projected cavity parameters
cVc   = ( V(l,l) - 2 * V(s,l) + V(s,s) ) / 2;
Vc    = (V(:,l) - V(:,s) ) / sq2;
cM    = (M(l) - M(s)) / sq2;
cVnic = max([cVc / (1 - p * cVc),0]); 
cmni  = cM + cVnic * ( p * cM - mp);

% rank 1 calculation: step factor
z     = cmni / sqrt(cVnic);
[e,lP,exit_flag]= log_relative_Gauss(z);
switch exit_flag
  case 0
    alpha = e / sqrt(cVnic);
    %beta  = alpha * (alpha + cmni / cVnic);
    %r     = beta * cVnic / (1 - cVnic * beta);
    beta  = alpha * (alpha * cVnic + cmni);
    r     = beta / (1 - beta);

    % new message
    pnew  = r / cVnic; 
    mpnew = r * ( alpha + cmni / cVnic ) + alpha;

    % update terms
    dp    = max([-p + eps,gam * (pnew - p)]); % at worst, remove message
    dmp   = max([-mp + eps,gam * (mpnew- mp)]);
    d     = max([dmp dp]); % for convergence measures

    pnew  = p  + dp;
    mpnew = mp + dmp;

    % project out to marginal
    Vnew  = V - dp / (1 + dp * cVc) * (Vc * Vc');
    Mnew  = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

    if any(isnan(Vnew)); keyboard; end
    % if z < -30; keyboard; end

    % normalization constant
    %logS  = lP - 0.5 * (log(beta) - log(pnew)) + (alpha * alpha) / (2*beta);
    
    % there is a problem here, when z is very large
    logS  = lP - 0.5 * (log(beta) - log(pnew) - log(cVnic)) + (alpha * alpha) / (2*beta) * cVnic;
    
  case -1 % impossible combination
    d = nan;
    
    Mnew  = 0;
    Vnew  = 0;
    pnew  = 0;    
    mpnew = 0;
    logS  = -Inf;
  case 1 % uninformative message
    d     = 0;
    % remove message from marginal:
    % new message
    pnew  = 0; 
    mpnew = 0;

    % update terms
    dp    = -p; % at worst, remove message
    dmp   = -mp;
    d     = max([dmp dp]); % for convergence measures

    % project out to marginal
    Vnew  = V - dp / (1 + dp * cVc) * (Vc * Vc');
    Mnew  = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

    logS  = 0;
end


function [e,logPhi,exit_flag] = log_relative_Gauss(z)

persistent l2p
l2p = log(2) + log(pi);

if z < -6 
    e = 1;
    logPhi = -1.0e12;
    exit_flag = -1;
elseif z > 6 % this gives zero precision messages.
    e = 0;
    logPhi = 0;
    exit_flag = 1;
else
    logphi = -0.5 * (z * z + l2p);           % faster than Matlabs normpdf
    logPhi = log(0.5 * erfc(-z ./ sqrt(2))); % faster than Matlabs normcdf
    e      = exp(logphi - logPhi);
    exit_flag = 0;
end
