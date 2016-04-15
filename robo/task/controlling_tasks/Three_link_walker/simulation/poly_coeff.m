function [a0,a1,a3,a4] = poly_coeff(q_minus,dq_minus)

  % accepts q_minus and dq_minus as COLUMN vectors
  % and returns the associated elements of the a-matrix of control parameters

  % preliminary calculations
  q_plus = delta_q(q_minus);
  dq_plus = delta_dq(q_minus, dq_minus);
  theta_minus = q_to_theta(q_minus);
  theta_plus = q_to_theta(q_plus);

  % setup matrices 
  T1 = [eye(2) [0 0]'];
  T2 = [0 0 1];
  H = [0 1 0; 0 0 1; -1 0 0];
  M = 4;

  % find a0
  a0 = T1*H*q_plus;

  % find a1
  k = T2*(1/q_plus(1)*H*q_plus);
  a1 = a0 + T1*[1/(k*dq_plus(1))*H*dq_plus]*(theta_minus-theta_plus)/M;

  % find a4
  a4 = T1*H*q_minus;

  % find a3
  k = T2*(1/q_minus(1)*H*q_minus);
  a3 = a4 - T1*[1/(k*dq_minus(1))*H*dq_minus]*(theta_minus-theta_plus)/M;