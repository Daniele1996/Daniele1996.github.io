function err = error_rate(Y, PHI, theta, N)

    F_zt = sigmoid(PHI*theta);
    Y_hat = F_zt > 0.5;
    err = (1/N)*sum(Y ~= Y_hat);

end

