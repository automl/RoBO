function [x,f] = FindGlobalGPMinimum(BestGuesses,GP,xmin,xmax)

[N,D] = size(BestGuesses);

X = zeros(N+10,D);
F = zeros(N+10,1);
alpha = (GP.cK \ (GP.cK' \ GP.y));

for i = 1 : size(BestGuesses,1)
    [X(i,:),F(i)]   = fmincon(@(x)GPmeanderiv(x,GP,alpha),BestGuesses(i,:),[],[],[],[],xmin,xmax,[], ...
        optimset('MaxFunEvals',100,'TolX',eps,'Display','off','GradObj','on'));
    
    %[X(i,:),F(i)]   = fminbnd(@(x)GPmeanderiv(x,GP,alpha),xmin,xmax, ...
    %    optimset('MaxFunEvals',100,'TolX',eps,'Display','off'));
end
for i = size(BestGuesses,1) + 1: size(BestGuesses,1) + 10
    start = xmin + (xmax-xmin) .* rand(1,D);
    [X(i,:),F(i)]   = fmincon(@(x)GPmeanderiv(x,GP,alpha),start,[],[],[],[],xmin,xmax,[], ...
        optimset('MaxFunEvals',100,'TolX',eps,'Display','off','GradObj','on'));
end

[f,xi] = min(F);
x = X(xi,:);
end

function [f,df] = GPmeanderiv(x,GP,alpha)

kx  = feval(GP.covfunc{:},GP.hyp.cov,x,GP.x);
dkx = feval(GP.covfunc_dx{:},GP.hyp.cov,x,GP.x);

f  = kx * alpha;
df = zeros(size(x,2),1);
for d = 1:size(x,2)
    df(d) = dkx(:,:,d) * alpha;
end

end