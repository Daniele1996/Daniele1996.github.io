function Linearized_Data = linearize_data(Data)

    mean_values = mean(Data);
    
    Linearized_Data = ((Data-mean_values)).^2;

end

