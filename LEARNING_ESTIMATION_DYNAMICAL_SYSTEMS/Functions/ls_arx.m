function theta = ls_arx(y,H,N,n)

theta = (H.'*H/(N-n))\(H.'*y(n+1:end))/(N-n);

end