function res_test= cross_correlation_test(eps,u,N,n,m,alpha)
% eps : residual vector
% m : max value of tau - limit of the length of the sample crosscocorrelation
% alpha : significance level

var = (eps.')*eps/(N-n);
crosscorr = zeros(m,1);

for tau = n+1:n+m
    crosscorr(tau-n) = (1/(N-n))*(eps(tau+1:end).')*u(n+1:end-tau);
end

Hu = hank_mat(u,m);
Covar_u = (Hu.'*Hu/(N-n)); 

figure;
plot(n+1:n+m,crosscorr, '-o')
title('Cross Correlation $\hat{r_{\epsilon u}}(\tau)$', 'Interpreter', 'latex')
xlabel('$\tau$', 'Interpreter', 'latex')
ylabel('Cross Correlation $\hat{r_{\epsilon u}}$', 'Interpreter', 'latex')
grid on;

x = (N-n)*(crosscorr.')*(inv(Covar_u))*(crosscorr)/var^(2);
chi_alpha = chi2inv(1-alpha,m);

if x <= chi_alpha
    fprintf('Cross Correlation test: PASSED\t %3f <= %3f', x, chi_alpha);
else
    fprintf('Cross Correlation: FAILED\t %3f > %3f', x, chi_alpha);
end

res_test = 0;

end

