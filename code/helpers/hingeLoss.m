function result = hingeLoss(input1, input2)

 result = max(1 + input1, 0) + max(1 - input2, 0);

end
