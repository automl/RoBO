function [x0] = find_ic(a,theta_minus,dtheta_minus)

  % given initial conditions within zero dynamics,
  % function returns the initial conditions for the full model

  H = [0 1 0; 0 0 1; -1 0 0];  
  
  a0 = a(:,1);
  a1 = a(:,2);
  a3 = a(:,4);
  a4 = a(:,5);
  M = 4;

  q_minus = inv(H)*[a4;theta_minus];
  q_plus = delta_q(q_minus);
  theta_plus = q_to_theta(q_plus);
  dq_minus = inv(H)*[M*(a4-a3)/(theta_minus-theta_plus); 1]*dtheta_minus;
  dq_plus = delta_dq(q_minus, dq_minus);
  x0 = [q_plus' dq_plus'];