function result = derivativeRelu(value)

 result = zeros(1, length(value));

 for i = 1:length(value)

   if value(i) > 0

      result(i) = 1;

   else

      result(i) = 0;

   end

 end

 result = result';

end
