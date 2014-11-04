function dHp = LogLoss(logP,lmb,lPred,~,~,~)

H   = - sum(exp(logP) .* (logP + lmb));           % current Entropy 
dHp = -sum(exp(lPred) .* bsxfun(@plus,lPred,lmb),1) - H; % @minus? If you change it, change it above in H, too!