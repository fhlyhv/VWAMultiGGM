function [ELBO,TVinvT,ETKinvT,ETKinvT2] = ELBOTree(Vinv,nump,ET,EK,ELAMBDA,nS,idm,nscale)

% ELAMBDA needs to be updated before inputing into the function
TVinvT = cell(nscale,1);
ETKinvT = cell(nscale,1);
ETKinvT2 = cell(nscale,1);

for m = 1:nscale-1
    TVinvT{m} = ET(idm{m},idm{m+1})*Vinv{m+1}*ET(idm{m+1},idm{m});
    ETKinvT{m} = TVinvT{m}/(nump(m+1)-1);
    ETKinvT2{m} = 1/nump(m+1)/(nump(m+1)-3)*TVinvT{m}.^2 ...
        +1/nump(m+1)/(nump(m+1)-1)/(nump(m+1)-3)*diag(TVinvT{m})*diag(TVinvT{m})';
end

ELBO = 0;

for m = 2:nscale
    ELBO = ELBO -sum(sum(ET(idm{m},idm{m-1}).*nS(idm{m},idm{m-1}))) ...  %ELAMBDA{m}.*EK{m}
        -sum(sum(ETKinvT{m-1}.*nS(idm{m-1},idm{m-1})))/2;
    if m < nscale
        ELBO = ELBO -sum(sum(ELAMBDA{m}.*(2*EK{m}.*ETKinvT{m}+ETKinvT2{m})))/4;
    end
end
    
    
    






