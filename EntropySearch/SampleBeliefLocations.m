function [zb,mb] = SampleBeliefLocations(GP,xmin,xmax,Nb,BestGuesses,propfun)
% samples belief locations according to x ~ p(f(x) < min f(GP.x))
% returns zbelief, zeval, and the measure mb propto p(f(x) < min f(GP.x))
%
% Philipp Hennig, August 2011

fprintf 'generating belief and evaluation points.\n'

if isempty(GP.x)  % if there are no previous evaluations, just sample uniformly
    D  = size(xmax,2);
    zb = bsxfun(@plus,bsxfun(@times,(xmax - xmin),rand(Nb,D)),xmin);
    mb = -log(prod(xmax-xmin)) * ones(Nb,1);
    return;
end

D  = size(GP.x,2);

%Ky = GP.cK \ (GP.cK' \ y_vector(GP));             % pre-compute for ProbImprove
%MinGuess = min(k_matrix(GP,GP.x,GP.x) * Ky);      % construct current best guess
%logProbImprove = @(x) ProbImprove(x,GP,Ky,MinGuess,xmin,xmax);

%EI       = EI_fun(GP,xmin,xmax);
%EI        = PI_fun(GP,xmin,xmax);
EI = feval(propfun{:},GP,xmin,xmax);

%xx = xmin + (xmax - xmin) .* rand(1,D);
d0 = norm(xmax - xmin) ./ 2;               % step size for the slice sampler

zb = zeros(Nb,D);                                        % fill zb with samples
mb = zeros(Nb,1);

numblock = floor(Nb / 10); % number of batches for re-starts.

% restarts = zeros(numblock,D);
% ell = exp(GP.hyp.cov(1:D))';
% for i = 1:numblock
%     start = xmin + (xmax - xmin) .* rand(1,D);
%     [xend,vend] = fmincon(EI,start,[],[],[],[],xmin,xmax,[], ...
%             optimset('MaxFunEvals',20,'TolX',eps,'Display','off','GradObj','on'));
%     % distance to previous restarts:
%     if i > 1
%         [mdist,imd] = min(sqrt(sum(bsxfun(@minus,bsxfun(@rdivide,restarts(1:i-1,:),ell),xend./ell).^2,2)./D));
%         if mdist < 0.1
%             fprintf '.'
%             xend = start;
%         end
%     end
%     restarts(i,:) = xend;
% end

restarts = zeros(numblock,D);
restarts(1:min(numblock,size(BestGuesses,1)),:) = ...
    BestGuesses(max(size(BestGuesses,1)-numblock+1,1):end,:);
restarts(min(numblock,size(BestGuesses,1)) + 1:numblock,:) = ...
    bsxfun(@plus,xmin,bsxfun(@times,(xmax - xmin),rand(numel(min(numblock,size(BestGuesses,1)) + 1:numblock),D)));

xx = restarts(1,:);
subsample = 20;
for i = 1 : subsample * Nb              % sub-sample by factor of 10 to improve mixing
    if mod(i,subsample*10) == 0 && i / (subsample*10) < numblock
        xx = restarts(i/(subsample*10) + 1,:);
    end
    %xx = Slice_ShrinkRank(xx,logProbImprove,d0,true);
    xx = Slice_ShrinkRank_nolog(xx,EI,d0,true);
    if mod(i,subsample) == 0
        zb(i / subsample,:) = xx;
        emb         = EI(xx);
        mb(i / subsample)  = log(emb);
    end
end