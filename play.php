<?php
	
$x = [100, 200, 300, 600];

$y = [200, 400, 600, 1200];

// Hypothesis --> h(x) = theta1 * x
$h = function($x, $theta0) {
	return $theta0 * $x;
};

// Cost function
$J = function($X, $Y, $h, $theta0) {
	$m					= count($X);
	
	$totalError			= 0;

	for ($i = 0; $i < count($X); $i++) {
		$sample 	= $X[$i];

		$predicted	= $h($sample, $theta0);

		$actual		= $Y[$i];

		$totalError	+= pow($predicted - $actual, 2);
	}

	return (1 / $m) * $totalError;
	

};

// Do the gradient!

define("ITERATIONS", 10);
define("LEARNING_RATE", 0.00001);

$theta0 = 0;

for ($iteration = 0; $iteration < ITERATIONS; $iteration++) {
//	$theta0 = $theta0 - (LEARNING_RATE * );

	$totalDiff = 0;

	// Sum-up the differences
	for ($i = 0; $i < count($x); $i++) {
		$actual 	= $y[$i];

		$predicted 	= $h($x[$i], $theta0);

		$diff 		= ($predicted - $actual) * $x[$i];

		$totalDiff 	+= $diff;
	}

	$theta0 = $theta0 - (LEARNING_RATE * ((1 / count($x)) * $totalDiff));

	$cost	= $J($x, $y, $h, $theta0);

	print("$cost\t$theta0 \n");
}

//
//
//for ($i = 0; $i < count($x); $i++) {
//	$input		= $x[$i];
//	$real 		= $y[$i];
//	$predicted 	= $h($input, $theta0);
//	$error		= $J($x, $y, $h, $theta0);
//
//	print("Input: $input, Real: $real, Predicted: $predicted, Error: $error\n");
//}

//$theta0 = 0;
//$alpha	= 0.5;
//
//for ($i = 0; $i < 100; $i++) {
//	printf("Starting theta: %d \n", $theta0);
//
//	$theta0	= $theta0 - $alpha * $J($x, $y, $h, $theta0);
//
//	printf("After iteration #%d, theta is %d \n", $i, $theta0);
//}



