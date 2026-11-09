function result = derivativeSigmoid(input)

 result = zeros(1, length(input));

 for i = 1:length(input)

   result(i) = (1 - sigmoidFunc(input(i))) * sigmoidFunc(input(i));

 end

 result = result';

end
