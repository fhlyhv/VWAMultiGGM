
function [J, klDiv] = gaussEM(Jinit, Pobs, dim, tol, maxIts)
%gaussEM    EM algorithm for Gaussian MRFs.
%   EM algorithm for parameter estimation of Gaussian latent variable models.
%   -- Stochastic realization: Input covariance is "truth" to be approximated
%   -- ML estimation: Input covariance is sample covariance of observed data
%   E-step implemented via brute force matrix inversion, 
%   M-step via iterative proportional fitting.
%
%     [J, klDiv] = gaussEM(adjacency, Pobs, dim, tol, maxIts)
%
% PARAMETERS:
%   Jinit = initial joint distribution over observed and latent variables
%     (inverse covariance with desired sparsity structure)
%   Pobs = desired covariance of "observed" variables (assumed last 
%     variables in Jinit)
%   dim = scalar giving dimension of hidden variable at each node (DEFAULT = 1)
%   tol = convergence tolerance (DEFAULT = 1e-8)
%   maxIts = maximum number of iterations (DEFAULT = 100)
% OUTPUT:
%   J = sparse inverse covariance approximation 
%   klDiv = KL divergence of approximation after each EM iteration

% Erik Sudderth
%  February 19, 2003 - Initial version completed


% Check parameters
if (nargin < 2)
  error('Invalid number of parameters.');
end
if (nargin < 3)
  dim = 1;
end
if (nargin < 4)
  tol = 1e-2;
end
if (nargin < 5)
  maxIts = 100;
end

% Extract adjacency matrix from starting model parameters
N = length(Jinit)/dim;
adjFull = (abs(Jinit) > 0) - eye(N*dim);
adjFull = adjFull(1:dim:end,1:dim:end);
M = length(Pobs)/dim;
indObs = [(N-M)*dim+1:N*dim];    %index of observations

% Initialize variables
Jfull = Jinit;
Pfull = inv(Jfull);
Jobs = inv(Pobs);

klQuotient = Pobs/(Pfull(indObs,indObs));
klDiv = -0.5*(logdet(Pobs)-logdet(Pfull(indObs,indObs)) + trace(eye(M*dim)-klQuotient));
fprintf(1,'klDiv = %g\n', klDiv);

% Iterate until KL divergence falls below tol
for (s = 2:maxIts)
  if (s >= 3 & abs(klDiv(s-1) - klDiv(s-2)) < tol)
    break;
  end

  % E step: create expected sufficient statistics
  Jsuff = Jfull;
  Jsuff(indObs,indObs) = Jsuff(indObs,indObs)+Jobs-inv(Pfull(indObs,indObs));
  Psuff = inv(Jsuff);

  % M step: match sufficient statistics using IPF
  Jfull = gaussIPF(adjFull, Psuff, dim);
  
  Pfull = inv(Jfull);
%  klQuotient = Pobs*inv(Pfull(indObs,indObs));
  klQuotient = Pobs/(Pfull(indObs,indObs));%Pfull(indObs, indObs)*inv(Pobs);
  klDiv(s) = -0.5*(logdet(Pobs)-logdet(Pfull(indObs,indObs)) + trace(eye(M*dim)-klQuotient));
  fprintf(1,'klDiv = %g\n', klDiv(s));
end

klQuotient = Pfull(indObs,indObs)/Pobs;
klDiv = -0.5*(logdet(Pfull(indObs,indObs))-logdet(Pobs) + trace(eye(M*dim)-klQuotient))

% Create output inverse covariance matrix
J = Jfull;

