function [ i, Cdist ] = getGiniIndex( data, B )

    [n, m] = size(data);
    
    Cdist = zeros(1,3);
    for i = 1:n
        x = data(i, 1:m);
        for j = 1:(m-1)
            if x(j) < B(j,1) || x(j) > B(j,2)
                Cdist(x(m)+1) = Cdist(x(m)+1) - 1;
                break;
            end
        end
        Cdist(x(m)+1) = Cdist(x(m)+1) + 1;
    end
    
    n = sum(Cdist);
    Cprob = Cdist./n;
    
    i = 1 - sumsqr(Cprob);

end

