function [ dt ] = splitLeaf( data, B, current_depth, max_depth, stride )
% This function computes a new threshold for a splitting node
    
    if current_depth == max_depth
        dt = [];
        return;
    end
    
    [n, m] = size(data);
    
    di_top = 1.0 * n;
    
    for j = 1:(m-1)
        for thresh = B(j,1):stride:B(j,2)
            b_l = B;
            b_l(j, 2) = thresh;
            b_r = B;
            b_r(j, 1) = thresh;
            [i_l, nc_l] = getGiniIndex(data, b_l);
            [i_r, nc_r] = getGiniIndex(data, b_r);
            
            di = sum(nc_l) * i_l + sum(nc_r) * i_r;
            fprintf('%f %f\n', di, j);
            
            if di < di_top
                dim_top = j;
                di_top = di;
                thresh_top = thresh;
                il_top = i_l; nl_top = nc_l;
                ir_top = i_r; nr_top = nc_r;
            end
            
        end
    end
    
    fprintf('t%i,0: Gini Index = %f\n',current_depth,il_top);
    disp(nl_top);
    fprintf('t%i,1: Gini Index = %f\n',current_depth,ir_top);
    disp(nr_top);
    
    dt = [dim_top; thresh_top];
    
    b_l = B;
    b_l(dt(1), 2) = dt(2);
    b_r = B;
    b_r(dt(1), 1) = dt(2);
    
    dt_l = splitLeaf(data, b_l, current_depth+1, max_depth, stride);
    dt_r = splitLeaf(data, b_r, current_depth+1, max_depth, stride);
    
    dt = [dt dt_l dt_r];

end

