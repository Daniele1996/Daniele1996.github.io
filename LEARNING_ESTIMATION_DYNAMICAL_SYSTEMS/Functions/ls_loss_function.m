function J_theta_hat = ls_loss_function(eps,N,n)
% eps : residuals vector
J_theta_hat = (eps.')*eps/(N-n);

end