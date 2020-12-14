function y = logMvGamma(x,p)

y = p*(p-1)/4*log(pi)+sum(gammaln(x+(1-(1:p))/2));