function [all_w, J_history] = steepestGradientDescent(X, y, alpha, num_iters, num_labels, lambda)
% Function that inplement gradient desent steep by steep

% History of the cost function in each iteration
J_history = zeros(num_iters, num_labels);

% Some useful variables
n = size(X, 2);

% Variable of all optimal weight found for each class or label
all_w = zeros(num_labels, n + 1);

for c = 1:num_labels
    fprintf('\nTraining k: %f', c);
    w = all_w(c, :)';
    for iter = 1:num_iters
        [J, grad] = regCostFunction(w, X, (y == c), lambda);
        w = (w - (alpha*grad)); 
        J_history(iter, c) = J;
    end
    all_w(c, :) = w';    
end
end


%     iter = 1;
%     while (iter <= num_iters)
%         [J, grad] = regCostFunction(w, X, (y == c), lambda);
%         w = (w - (alpha*grad)); 
%         J_history(iter, c) = J;
%         if (iter > 1)
%             if (J_history(iter-1, c) > J_history(iter, c))
%                 lambda = J_history(iter-1, c) - J_history(iter, c);
%             end
%             if ((J_history(iter-1, c) - J_history(iter, c))< 0.0001)
%                 break
%             end
%         end
%         iter = iter + 1;
%     end
%     all_w(c, :) = w';