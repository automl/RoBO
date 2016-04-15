function [poly]=bezier(coeff,M,var)

  % returns a bezier polynomial in 'var' from the coefficients in 'coeff'
  % bezier() is called by 'symb_model_biped_3dof.m'

  poly=0;
  for i = 0:1:M
    poly=poly+coeff(i+1).*(factorial(M)./(factorial(i).*factorial(M-i))).*var.^i.*(1-var).^(M-i);
  end