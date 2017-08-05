function g = logistic(z)
% Compute logistic or sigmoid function
g = 1.0 ./ (1.0 + exp(-z));
end
