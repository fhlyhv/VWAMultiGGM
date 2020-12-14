function y = MvPolyGamma(x,p,n)

% d: dimension
% Yu Hang, Nov. 2016, NTU

y = sum(psi(n,x+(1-(1:p))/2));
