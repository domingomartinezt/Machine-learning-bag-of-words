function [J, grad] = regCostFunction(w, X, y, lambda)
%   Compute the cost of using
%   w as the parameter to learn for regularized logistic regression and the
%   gradient of the cost (partial derivate for the given weight),
%   this funtion recives a lambda parameter of regularization when 
%   is cero that means without regularization. 

% Initialize some useful values
n = length(y); % number of training examples

% variables to return cost and gradient matrix 
J = 0;
grad = zeros(size(w));

% Add ones to the X data matrix
X = [ones(n, 1) X];

% compute the cost of a given weight
h = logistic(X*w);
J = (1/n)*sum(-y.*log(h)-(1-y).*log(1-h))+ lambda/(2*n) *sum(w(2:end).^2);

%compute the partial derivate for the given weight
p = length(w)-1;
grad(1) = (1/n)*sum(h-y.*X(:,1));
grad(2:end) = (1/n)*sum(repmat(h-y,[1,p]).*X(:,2:end))'+(lambda/n)*w(2:end);

end
