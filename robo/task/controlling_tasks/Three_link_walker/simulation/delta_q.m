function [q_plus] = delta_q (q_minus)

  % accepts q_minus as a COLUMN vector and returns
  % the post-impact configuration
  
  % delta_q() is essentially equivalent to the relative angels
  % portion of delta() function in 'full_simul.m'
  
  M = [1 1 0; 0 -1 0; 0 -1 1];
  b = [pi -2*pi -pi]';
  q_plus = M*q_minus+b;
 