function descent = newton_raphson(Y, PHI, theta, lambda)
    
    F = sigmoid(PHI*theta);
    W = diag(F.*(1-F));
    I = eye(length(theta));

    % in case we don't consider beta_0
    I(1,1) = 0;

    if lambda > 0 %regularized
        % descent = inv(PHI.'*W*PHI+ lambda*I)*(PHI.'*(F-Y)+lambda*theta);
        descent = (PHI.'*W*PHI+ lambda*I)\(PHI.'*(F-Y)+lambda*[0; theta(2:end)]);
    
    else %not regularized
        
        descent = (PHI.'*W*PHI)\(PHI.'*(F-Y));

    end
end

