function EKall = PostProcessing(ET, EK, EKinv, Elambda, Eomega, ELAMBDA, idm, nscale)

thr = zeros(nscale, 1);
for m = 2:nscale
    ll = Elambda{m};
    ll = ll ./ (1 + ll);
    [~, fx, x, ~] = kde(ll);
    if m == nscale
        idx = find(x > 1e-2 & x < 4e-2);  % try to find the first valley close to 0
    else
        idx = find(x > 1e-2 & x < 0.6);
    end
    fx = fx(idx);
    x = x(idx);
    fx_min = min(fx);
    q = find(fx <= fx_min);
    hold on; plot(x(q(1)), 0, 'r+');
    thr(m) = Eomega * x(q(1)) / (1 - x(q(1)));
    legend('kernel density', 'selected threshold');
    title(['kernel density of \gamma at scale ', num2str(m)]); 
end

EKall = full(ET);
for m = 2:nscale
    if m == nscale
        Ktmp = EK{m};
    else
        Ktmp = EK{m}+ET(idm{m},idm{m+1})*EKinv{m+1}*ET(idm{m+1},idm{m});%+diag(VT(idm{m},idm{m+1})*diag(EKinv{m+1}));
    end
    Ktmp(ELAMBDA{m}>thr(m)) = 0;
    EKall(idm{m},idm{m}) = Ktmp;        
end
    

