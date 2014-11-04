function [ y ] = y_vector(GP)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if ~GP.deriv
  y = GP.y;
else
  y = [GP.y; GP.dy];
end

end

