function [Mx,Vx,Mz,Vz,Vxz] = GP_moments(GP,x,z)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~isfield(GP,'poly')
  GP.poly = -1;
end

kx = k_matrix_full(GP,GP.x,x);
if GP.poly >= 0 && ~isempty(GP.x) % special case for no data. 
  GP.H    = poly_mat(GP.x,GP.poly,GP.deriv);
  cKH     = (GP.cK' \ GP.H');
  GP.cHKH = chol(cKH' * cKH + 1.0e-4 * eye(GP.poly + 1));
  Rx      = poly_mat(x,GP.poly,GP.deriv) - GP.H * (GP.cK \ (GP.cK' \ kx'));
else
  Rx = [];
end

Mx = GP_mean(GP,x,kx,Rx);
if nargout > 1
  Vx = GP_var(GP,x,kx,Rx);
end

if nargin > 2
  kz = k_matrix_full(GP,GP.x,z);
  if GP.poly >= 0 && ~isempty(GP.x)
    Rz = poly_mat(z,GP.poly,GP.deriv) - GP.H * (GP.cK \ (GP.cK' \ kz'));
  else
    Rz = [];
  end

  Mz = GP_mean(GP,z,kz,Rz);
  Vz = GP_var(GP,z,kz,Rz);  
  Vxz = GP_var(GP,x,kx,Rx,z,kz,Rz);
end

end

function M = GP_mean(GP,x,k,R)

  if isempty(GP.x)
    if GP.deriv
      M = zeros(size(x,1) .* 2,1);
    else
      M = zeros(size(x,1),1);
    end  
  else
    Ky = GP.cK \ (GP.cK' \ y_vector(GP));
    M = k * Ky;
    if ~isempty(R)
      M = M + R' * (GP.cHKH \ (GP.cHKH' \ (GP.H * Ky)));
    end
  end
 
end

function V = GP_var(GP,x,kx,Rx,z,kz,Rz)

  if nargin < 5
    z = x;
    kz = kx;
    Rz = Rx;
  end
  kk = k_matrix_full(GP,x,z);  
  
  if isempty(GP.x)
    V = kk;
  else
    V = kk - kz * (GP.cK \ (GP.cK' \ kx'));
    if ~isempty(Rx)
      V = V + Rz' * (GP.cHKH \ (GP.cHKH' \ Rx));
    end
  end
  
end


