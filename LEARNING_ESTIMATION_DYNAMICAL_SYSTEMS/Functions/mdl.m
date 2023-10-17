function MDL = mdl(J_theta_hat,N,n)

p = 2*n;

MDL = (N-n)*log(J_theta_hat)+2*p*log(N-n);

end