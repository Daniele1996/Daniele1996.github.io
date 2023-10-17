function res = residual(y, theta, H, n)

res = (y(n+1:end)-H*theta);

end

