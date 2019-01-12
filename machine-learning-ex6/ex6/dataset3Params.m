function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

potential_values = [0.01 ; 0.03 ; 0.1 ; 0.3 ; 1 ; 3 ; 10 ; 30 ];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Code commented out because it takes a lot of time to process every time I submit the solution ...
%
% best_prediction_error = 2;
%
% for c_index = 1:size(potential_values)
% 	C_potential = potential_values(c_index);
%
% 	for sigma_index = 1:size(potential_values)
% 		sigma_potential = potential_values(sigma_index);
%
% 		fprintf('Potential C: %f, potential sigma: %f\n', C_potential, sigma_potential);
%
% 		model= svmTrain(X, y, C_potential, @(x1, x2) gaussianKernel(x1, x2, sigma_potential));
%
% 		fprintf("Predictions: \n");
%
% 		predictions = svmPredict(model, Xval);
%
% 		prediction_error = mean(double(predictions ~= yval));
%
% 		if (prediction_error < best_prediction_error)
% 			fprintf('This one has better prediction error (%f) than the last one (%f)\n', prediction_error, best_prediction_error);
%
% 			best_prediction_error = prediction_error;
%
% 			C = C_potential;
%
% 			sigma = sigma_potential;
% 		endif
% 	endfor
% endfor

% These were found to be the best values ....
C = 1; sigma = 0.1;



% =========================================================================

end
