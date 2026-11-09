function result = sigmoidFunc(input)

 result = zeros(1, length(input));

 for i = 1:length(input)

   result(i) = 1 / (1 + exp(-input(i)));

 end

 result = result';

end
