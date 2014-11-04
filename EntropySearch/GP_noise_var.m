function noise_var = GP_noise_var(GP,y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(GP,'log')
  GP.log = 0;
end

if GP.log
  noise_var = exp(2*GP.hyp.lik)*ones(numel(y),1) ./ exp(2*y);
else
  noise_var = exp(2*GP.hyp.lik)*ones(numel(y),1);
end

if GP.deriv
  noise_var = [noise_var; noise_var];
end


end

