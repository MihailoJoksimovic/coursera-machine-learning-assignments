load('ex3data1.mat'); % training data stored in arrays X, y
[m n] = size(X);

lambda = 0.1;

num_labels = 10;


% You need to return the following variables correctly 
% all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
% X = [ones(m, 1) X];

% initial_theta = zeros(n + 1, 1);

% options = optimset('GradObj', 'on', 'MaxIter', 50);

% [theta, cost] = fmincg (@(t)(lrCostFunction(t, X, (y == 1), lambda)), initial_theta, options);

[all_theta] = oneVsAll(X, y, 10, lambda);

chances = zeros(num_labels, 1);

for j = 1 : num_labels
	chances(j) = sigmoid(X(i, :) * all_theta(j, :)');
endfor
