function H = poly_mat(x,order,with_deriv)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

H = ones(order+1,size(x,1));

for i=1:order
  H(i+1,:) = H(i,:) .* x';
end

if with_deriv
    HD = zeros(order+1,numel(x));
    for i=2:order+1
        HD(i,:) = x' .^ (i-2) * (i-1);
    end
    H = [H, HD];
end
