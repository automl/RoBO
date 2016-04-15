function [val] = q_to_theta(q)

  % returns the theta value of any position configuration projected
  % onto the zero dynamics
  
  val = -q(1);