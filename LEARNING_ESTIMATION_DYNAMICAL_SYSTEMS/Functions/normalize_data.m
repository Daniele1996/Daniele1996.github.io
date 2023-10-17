function Normalized_Data = normalize_data(Data, N)

    mean_values = mean(Data);
    norms = zeros(N,1);
    
    for count = 1:N
        norms(count) = norm(Data(count,:)-mean_values);
    end
    
    max_norm = max(norms);

    Normalized_Data = Data./max_norm;
    
end

