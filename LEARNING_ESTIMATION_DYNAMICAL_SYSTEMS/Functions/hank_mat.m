function H = hank_mat(data,n)

N=length(data)-n;
H=zeros(N,n);

for i = 1:N
    for j = 1:n
        H(i,j) = data(n-j+i);
    end
end

end
