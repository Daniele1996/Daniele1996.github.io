function J_theta_hat = lr_loss_function(Y, PHI, theta_hat, lambda)

    F_zt = sigmoid(PHI*theta_hat);
    
    if lambda > 0 %regularized

        % J_theta_hat = J_theta_hat + lambda*theta_hat.'*theta_hat;
        J_theta_hat = - (Y.'*log(F_zt)+(1-Y.')*log(1-F_zt)); + lambda*theta_hat(2:end).'*theta_hat(2:end);
    
    else %not regularized

        J_theta_hat = - (Y.'*log(F_zt)+(1-Y.')*log(1-F_zt));
    end

end

