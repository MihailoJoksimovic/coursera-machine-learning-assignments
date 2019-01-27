function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% fprintf("Num users: %d, Num movies: %d, Num features: %d\n", num_users, num_movies, num_features);

% fprintf("Num users in theta: %d, and num users in Y: %d\n", size(Theta, 1), size(Y, 2));

% error_sum = 0;
%
% for i = 1:size(X, 1) % Iterate through movies
% 	movie_idx		= i;
%
% 	movie_features	= X(i, :);
%
% 	for j = 1:size(Theta, 1) % Iterate through users
% 		user_idx	= j;
%
% 		% Check if user has rated this movie?
% 		if (R(movie_idx, user_idx) != 1)
% 			% fprintf("User #%d has not rated movie #%d\n", user_idx, movie_idx);
% 			continue;
% 		endif
%
% 		% Ok so, it seems that user has actually rated this movie. Let's see how did he rate it?
%
% 		users_rating_of_this_movie = Y(movie_idx, user_idx);
%
% 		% What would be the predicted rating for this movie?
%
% 		predicted_rating_of_this_movie	= Theta(user_idx, :) * X(i, :)';
%
% 		% fprintf("User #%d has rated movie #%d with rating: %f; predicted rating is: %f\n", user_idx, movie_idx, users_rating_of_this_movie, predicted_rating_of_this_movie);
%
% 		error_sum += (predicted_rating_of_this_movie - users_rating_of_this_movie) ^ 2;
% 	endfor
% endfor
%
% J = (1 / 2) * error_sum;

Y_predicted	= (Theta * X')';

% Get rid of ratings for movies we never rated
Y_predicted = Y_predicted .* R;

J = (1 / 2) * sum(sum((Y_predicted - Y) .^ 2));


Y_predicted	= (Theta * X')';

K			= size(X, 2);

% Calculate gradients for each movie

for i = 1:size(X, 1) % For each movie
	
	for j = 1:size(Theta, 1) % For each user
		
		if (R(i, j) != 1) 
			continue;
		endif;
		
		% X_grad(i, :) = (Y_predicted(i, :) - Y(i, :)) .* Theta(j, :);
		
		
		% for k = 1:K % For each feature
		% 	X_grad(i, k) 		+= (Y_predicted(i, j) - Y(i, j)) * Theta(j, k);
		%
		% 	Theta_grad(j, k) 	+= (Y_predicted(i, j) - Y(i, j)) * X(i, k);
		% endfor
		
	endfor
endfor


for i = 1:size(X, 1) % For each movie
	idx = find(R(i, :) == 1);
	
	idx
	
	Theta_temp 	= Theta(idx, :);
	Y_temp		= Y(i, idx);
	
	X_grad(i, :) 		= ((X(i, :) * Theta_temp' - Y_temp) * Theta_temp)';
endfor

% TODO: Figure out how to do the same for Theta_grad


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
