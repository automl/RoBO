function out = EntropySearch(in)
% probabilistic line search algorithm that adapts it search space
% stochastically, by sampling search points from their marginal probability of
% being smaller than the current best function guess.
%
% (C) Philipp Hennig & Christian Schuler, August 2011

fprintf 'starting entropy search.\n'

%% fill in default values where possible
if ~isfield(in,'likfunc'); in.likfunc = @likGauss; end; % noise type
if ~isfield(in,'poly'); in.poly = -1; end; % polynomial mean? 
if ~isfield(in,'log'); in.log = 0; end;  % logarithmic transformed observations?
if ~isfield(in,'with_deriv'); in.with_deriv = 0; end; % derivative observations?
if ~isfield(in,'x'); in.x = []; end;  % prior observation locations
if ~isfield(in,'y'); in.y = []; end;  % prior observation values
if ~isfield(in,'T'); in.T = 200; end; % number of samples in entropy prediction
if ~isfield(in,'Ne'); in.Ne = 10; end; % number of restart points for search
if ~isfield(in,'Nb'); in.Nb = 50; end; % number of representers
if ~isfield(in,'LossFunc'); in.LossFunc = {@LogLoss}; end;
if ~isfield(in,'PropFunc'); in.PropFunc = {@EI_fun}; end;
in.D = size(in.xmax,2); % dimensionality of inputs (search domain)

% the following should be provided by the user
%in.covfunc      = {@covSEard};       % GP kernel
%in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use
%covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard,
%covRQard, respectively.
%in.hyp          = hyp;  % hyperparameters, with fields .lik and .cov
%in.xmin         = xmin; % lower bounds of rectangular search domain
%in.xmax         = xmax; % upper bounds of rectangular search domain
%in.MaxEval      = H;    % Horizon (number of evaluations allowed)
%in.f            = @(x) f(x) % handle to objective function

%% set up
GP              = struct;
GP.covfunc      = in.covfunc;
GP.covfunc_dx   = in.covfunc_dx;
%GP.covfunc_dx   = in.covfunc_dx;
%GP.covfunc_dxdz = in.covfunc_dxdz;
GP.likfunc      = in.likfunc;
GP.hyp          = in.hyp;
GP.res          = 1;
GP.deriv        = in.with_deriv;
GP.poly         = in.poly;
GP.log          = in.log;
%GP.SampleHypers = in.SampleHypers;
%GP.HyperSamples = in.HyperSamples;
%GP.HyperPrior   = in.HyperPrior;

GP.x            = in.x;
GP.y            = in.y;
%GP.dy           = in.dy;
GP.K            = [];
GP.cK           = [];

D = in.D;
S0= 0.5 * norm(in.xmax - in.xmin);

%% iterations
converged = false;
numiter   = 0;
MeanEsts  = zeros(0,D);
MAPEsts   = zeros(0,D);
BestGuesses= zeros(0,D);
while ~converged && (numiter < in.MaxEval)
    numiter = numiter + 1;
    fprintf('\n');
    disp(['iteration number ' num2str(numiter)])
%     try
       
        % sample belief and evaluation points
        [zb,lmb]   = SampleBeliefLocations(GP,in.xmin,in.xmax,in.Nb,BestGuesses,in.PropFunc);
        
        [Mb,Vb]    = GP_moments(GP,zb);
        
        % belief over the minimum on the sampled set
        [logP,dlogPdM,dlogPdV,ddlogPdMdM] = joint_min(Mb,Vb);        % p(x=xmin)
        
        out.Hs(numiter) = - sum(exp(logP) .* (logP + lmb));       % current Entropy
        
        % store the best current guess as start point for later optimization.
        [~,bli] = max(logP + lmb);
        % is this far from all the best guesses? If so, then add it in.
        ell = exp(GP.hyp.cov(1:D))';
        if isempty(BestGuesses)
            BestGuesses(1,:) = zb(bli,:);
        else
            dist = min(sqrt(sum(bsxfun(@minus,zb(bli,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D));
            if dist > 0.1
                BestGuesses(size(BestGuesses,1)+1,:) = zb(bli,:);
            end
        end
        
        dH_fun     = dH_MC_local(zb,GP,logP,dlogPdM,dlogPdV,ddlogPdMdM,in.T,lmb,in.xmin,in.xmax,false,in.LossFunc);
        dH_fun_p   = dH_MC_local(zb,GP,logP,dlogPdM,dlogPdV,ddlogPdMdM,in.T,lmb,in.xmin,in.xmax,true,in.LossFunc);
        % sample some evaluation points. Start with the most likely min in zb.
        [~,mi]     = max(logP);
        xx         = zb(mi,:);
        Xstart     = zeros(in.Ne,D);
        Xend       = zeros(in.Ne,D);
        Xdhi       = zeros(in.Ne,1);
        Xdh        = zeros(in.Ne,1);
        fprintf('\n sampling start points for search for optimal evaluation points\n')
        xxs = zeros(10*in.Ne,D);
        for i = 1:10 * in.Ne
            if mod(i,10) == 1 && i > 1; xx = in.xmin + (in.xmax - in.xmin) .* rand(1,D); end;
            xx     = Slice_ShrinkRank_nolog(xx,dH_fun_p,S0,true);
            xxs(i,:) = xx;
            if mod(i,10) == 0; Xstart(i/10,:) = xx; Xdhi(i/10) = dH_fun(xx); end
        end
        
        % optimize for each evaluation point:
        fprintf('local optimizations of evaluation points\n')
        for i = 1:in.Ne
            [Xend(i,:),Xdh(i)] = fmincon(dH_fun,Xstart(i,:),[],[],[],[],in.xmin,in.xmax,[], ...
                optimset('MaxFunEvals',20,'TolX',eps,'Display','off','GradObj','on'));
            
        end
        % which one is the best?
        [xdhbest,xdhbv]   = min(Xdh);
                
        fprintf('evaluating function \n')
        xp                = Xend(xdhbv,:);
        yp                = in.f(xp);
        
        GP.x              = [GP.x ; xp ];
        GP.y              = [GP.y ; yp ];
        %GP.dy             = [GP.dy; dyp];
        GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
        GP.cK             = chol(GP.K);
        
        MeanEsts(numiter,:) = sum(bsxfun(@times,zb,exp(logP)),1);
        [~,MAPi]            = max(logP + lmb);
        MAPEsts(numiter,:)  = zb(MAPi,:);
        
        fprintf('finding current best guess\n')
        [out.FunEst(numiter,:),FunVEst] = FindGlobalGPMinimum(BestGuesses,GP,in.xmin,in.xmax);
        % is the new point very close to one of the best guesses?
        [cv,ci] = min(sum(bsxfun(@minus,out.FunEst(numiter,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D);
        if cv < 2.5e-1 % yes. Replace it with this improved guess
            BestGuesses(ci,:)  = out.FunEst(numiter,:);
        else % no. Add it to the best guesses
            BestGuesses(size(BestGuesses,1)+1,:) = out.FunEst(numiter,:);
        end
        
        % optimize hyperparameters
        if in.LearnHypers
            minimizeopts.length    = 10;
            minimizeopts.verbosity = 1;
            GP.hyp = minimize(GP.hyp,@(x)in.HyperPrior(x,GP.x,GP.y),minimizeopts);
            GP.K   = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
            GP.cK  = chol(GP.K);
            fprintf 'hyperparameters optimized.'
            display(['length scales: ', num2str(exp(GP.hyp.cov(1:end-1)'))]);
            display([' signal stdev: ', num2str(exp(GP.hyp.cov(end)))]);
            display([' noise stddev: ', num2str(exp(GP.hyp.lik))]);
        end
        
        out.GPs{numiter} = GP;
%     catch error
%         if numiter > 1
%             out.FunEst(numiter,:) = out.FunEst(numiter-1,:);
%         end
%         fprintf('error occured. evaluating function at random location \n')
%         xp                = in.xmin + (in.xmax - in.xmin) .* rand(1,D);
%         yp                = in.f(xp);
%         
%         GP.x              = [GP.x ; xp ];
%         GP.y              = [GP.y ; yp ];
%         %GP.dy             = [GP.dy; dyp];
%         GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
%         GP.cK             = chol(GP.K);
%         
%         MeanEsts(numiter,:) = sum(bsxfun(@times,zb,exp(logP)),1);
%         [~,MAPi]            = max(logP + lmb);
%         MAPEsts(numiter,:)  = zb(MAPi,:);
%         
%         out.errors{numiter} = error;
%     end
end

%% construct output
out.GP       = GP;
out.MeanEsts = MeanEsts;
out.MAPEsts  = MAPEsts;
out.logP     = logP;
