function p = predict(all_w, X)
% Using all the parameters weights all_w, evaluate and return the 
% maximun prediction found.

m = size(X, 1);
num_labels = size(all_w, 1);

% return value of class of predictions for each sample row.
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

hw = X * all_w';
[temp, p] = max(hw, [], 2);

end
