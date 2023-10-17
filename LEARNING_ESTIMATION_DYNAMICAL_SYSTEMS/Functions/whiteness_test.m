function res_test = whiteness_test(eps,N,n,m, alpha)

% eps : residual vector
% m : max value of tau - limit of the length of the sample autocorrelation
% alpha : significance level

var = (eps.')*eps/(N-n);
autocorr = zeros(m,1);

for tau = 1:m
    autocorr(tau) = (1/(N-n))*(eps(tau+1:end).')*eps(1:end-tau);
end

figure;
plot(1:m+1,[var; autocorr], '-o')
title('Autocorrelation $\hat{r_{\epsilon}}(\tau)$', 'Interpreter', 'latex')
xlabel('$\tau$', 'Interpreter', 'latex')
ylabel('Auto Correlation $\hat{r_{\epsilon}}$', 'Interpreter', 'latex')
grid on;

x = (N-n)*(autocorr.')*(autocorr)/var^(2);
chi_alpha = chi2inv(1-alpha,m);

if x <= chi_alpha
    fprintf('Whiteness test: PASSED\t %3f <= %3f', x, chi_alpha);
else
    fprintf('Whiteness test: FAILED\t %3f > %3f', x, chi_alpha);
end

res_test = 0;

end