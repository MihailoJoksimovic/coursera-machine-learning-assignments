function [encoded] = encodeInputNumber(input_number, num_labels)
	encoded = zeros(num_labels, 1);
	
	% Put 1 on the appropriate row, depending on input number
	
	encoded(input_number) = 1;							   
end