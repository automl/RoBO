function theta_plus = delta_theta(a,theta_minus)

  % returns the post-impact value of theta
  
  H = [0 1 0; 0 0 1; -1 0 0];  
  
  a0 = a(:,1);
  a1 = a(:,2);
  a3 = a(:,4);
  a4 = a(:,5);
  M = 4;

  q_minus = inv(H)*[a4;theta_minus];
  q_plus = delta_q(q_minus);
  theta_plus = q_to_theta(q_plus);