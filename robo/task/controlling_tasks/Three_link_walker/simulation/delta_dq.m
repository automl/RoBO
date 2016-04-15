function [dq_plus] = delta_dq(q_minus, dq_minus)

  % accepts q_minus, dq_minus as COLUMN vectors and returns
  % the post-impact velocity condition
  
  % delta_dq() is essentially equivalent to the velocity portion
  % of delta() function in 'full_simul.m'
  
  [De,Ce,Ge,Be,E] = dynamic_model_for_impacts(q_minus',dq_minus');

  dxe_minus=[dq_minus' 0 0];

  A=[De -E; E' zeros(2)];
  b=[De*dxe_minus'; 0; 0];
  c=inv(A)*b;

  % state after impact, but before leg swap (only velocities change)
  x_plus_beforeswap=[q_minus' c(1) c(2) c(3)]; 

  % perform leg swap
  tem=[1 1 0; 0 -1 0; 0 -1 1];
  S=[tem zeros(3,3); zeros(3,3) tem];
  x_plus_afterswap=x_plus_beforeswap*S' + [pi -2*pi -pi 0 0 0];
  dq_plus=[x_plus_afterswap(4:6)]';
