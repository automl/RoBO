function D = logdet(M)
% logarithm of determinant of positive definite matrix
% Philipp Hennig, 12 July 2011

D = 2 * sum(log(diag(chol(M))));