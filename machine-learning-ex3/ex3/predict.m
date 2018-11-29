function [p, X_after_layer_1, X_after_layer_2] = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1:m % For each sample
	sample = X(i, :);
	
	X_after_layer_1	= sigmoid(X(i, :) * Theta1');
	
	X_after_layer_1 = [1 X_after_layer_1];

	% Forward propagate the X_after_layer_1 values to third layer		
	X_after_layer_2 = sigmoid(X_after_layer_1 * Theta2');
	
 	[x, ix]= max(X_after_layer_2);
	
	p(i)	= ix;
endfor







% =========================================================================


end
