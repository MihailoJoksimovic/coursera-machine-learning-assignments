function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

[nrows, ncols] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = 1 / m;

sum = 0;

for i = 1:m
	oneProbability 	= (-y(i) * log(sigmoid(X(i, :) * theta)));
	zeroProbability = (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta));
	
	sum = sum + (oneProbability - zeroProbability);
endfor

J = (1 / m ) * sum;

regularizationPart = (lambda / (2 * m));

% calculate theta sums

regularizationSum = 0;

for j = 2:size(theta)
	regularizationSum= regularizationSum + theta(j) ^ 2;
endfor

J = J + (lambda / (2 * m)) * regularizationSum;

% calculate gradients

for j = 1 : size(theta) % calculate gradient for each theta
	grad(j) = 1 / m;
	
	% Sum up differences for each sample
	
	differenceSum = 0;
	
	for i = 1 : nrows % for each sample
		% Calculate predicted and actual value
		predicted 				= sigmoid(X(i, :) * theta);
		actual					= y(i);
		
		diffPredictedActual		= predicted - actual;
		
		differenceSum 			= differenceSum + diffPredictedActual * X(i, j);
	endfor
	
	grad(j) = (1 / m) * differenceSum;
	
	if (j > 1)
		grad(j) = grad(j) + (lambda / m) * theta(j);
	endif
endfor





% =============================================================

end
