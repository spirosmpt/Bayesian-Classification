clc;

clear;

close all;

% Φόρτωση δεδομένων για τεστ

test_class_0 = (load("X0train.mat", "X0").X0)';

test_class_1 = (load("X1train.mat", "X1").X1)';

% Δημιουργία δεδομένων εκπαίδευσης

train_class_0 = randn(2, 200);

train_class_1 = zeros(2, 200);

for j = 1:2

  for k = 1:200

       if randn() >= 0

         train_class_1(j, k) = -1 + randn();

       else

         train_class_1(j, k) = 1 + randn();

       end

  end

end

% Αρχικοποίηση παραμέτρων για το νευρωνικό δίκτυο

input_size = 2;

hidden_size = 20;

output_size = 1;

learning_rate = 2 * 0.0001;

regularization_param = 0.01;

epsilon = 1e-315;

% Παράμετροι εκπαίδευσης

epochs = 5000;

best_cost = inf;

for run = 1:10

 % Τυχαία αρχικοποίηση βαρών

  weights_input_hidden = (1 / (hidden_size + input_size)) * randn(hidden_size, input_size);

 bias_input_hidden = zeros(hidden_size, 1);

  weights_hidden_output = (1 / (hidden_size + output_size)) * randn(output_size, hidden_size);

 bias_hidden_output = 0;

 iteration = 1;

 total_cost = zeros(1, epochs * 200);

 for epoch = 1:epochs

   for sample_idx = 1:200

      input_class_0 = train_class_0(:, sample_idx);

      input_class_1 = train_class_1(:, sample_idx);

      % Forward pass for class 0

     hidden_input_0 = weights_input_hidden * input_class_0 + bias_input_hidden;

     hidden_output_0 = Relu(hidden_input_0);

     final_input_0 = weights_hidden_output * hidden_output_0 + bias_hidden_output;

     output_0 = final_input_0;

     % Forward pass for class 1

     hidden_input_1 = weights_input_hidden * input_class_1 + bias_input_hidden;

     hidden_output_1 = Relu(hidden_input_1);

     final_input_1 = weights_hidden_output * hidden_output_1 + bias_hidden_output;

     output_1 = final_input_1;

     % Cost calculation

     total_cost(iteration) = Expon(output_0, output_1);

     % Derivatives

     delta_output_0 = diff_Phi_expon(output_0);

     delta_hidden_0 = delta_output_0 .* 1;

     delta_input_0 = (weights_hidden_output') * delta_hidden_0;

     delta_hidden_output_0 = delta_input_0 .* diff_Relu(hidden_input_0);

     delta_output_1 = diff_Psi_expon(output_1);

     delta_hidden_1 = delta_output_1 .* 1;

     delta_input_1 = (weights_hidden_output') * delta_hidden_1;

     delta_hidden_output_1 = delta_input_1 .* diff_Relu(hidden_input_1);

     % Update regularization parameters

     if (sample_idx == 1) && (epoch == 1)

       P2 = ((delta_hidden_0 * [hidden_output_0' 1]) + (delta_hidden_1 * [hidden_output_1' 1])) .^ 2;

     else

          P2 = (1 - regularization_param) * P2_old + regularization_param * power(((delta_hidden_0 * [hidden_output_0' 1]) + (delta_hidden_1 * [hidden_output_1'
1])), 2);

     end

     P2_old = P2;

     if (sample_idx == 1) && (epoch == 1)

        P1 = ((delta_hidden_output_0 * [input_class_0' 1]) + (delta_hidden_output_1 * [input_class_1' 1])) .^ 2;

     else

        P1 = (1 - regularization_param) * P1_old + regularization_param * power(((delta_hidden_output_0 * [input_class_0' 1]) + (delta_hidden_output_1 *
[input_class_1' 1])), 2);

     end

     P1_old = P1;

       % Weight updates with regularization

      AB2 = [weights_hidden_output bias_hidden_output] - learning_rate * ((delta_hidden_0 * [hidden_output_0' 1] + delta_hidden_1 * [hidden_output_1' 1]) ./
sqrt(epsilon + P2));

     weights_hidden_output = AB2(1:hidden_size);

     bias_hidden_output = AB2(1, hidden_size + 1);

        AB1 = [weights_input_hidden bias_input_hidden] - learning_rate * ((delta_hidden_output_0 * [input_class_0' 1] + delta_hidden_output_1 * [input_class_1'
1]) ./ sqrt(epsilon + P1));

     weights_input_hidden = AB1(1:hidden_size, 1:input_size);

     bias_input_hidden = AB1(1:hidden_size, input_size + 1);

     iteration = iteration + 1;

   end

 end

 % Mean cost calculation

  mean_cost = 0;

  for idx = 1:200

    mean_cost = mean_cost + total_cost(epochs * 200 - 200 + idx);

  end

  mean_cost = mean_cost / 200;

  % Save best model

  if best_cost > mean_cost

    best_cost = mean_cost;

    final_weights_input_hidden = weights_input_hidden;

    final_weights_hidden_output = weights_hidden_output;

    final_bias_input_hidden = bias_input_hidden;

    final_bias_hidden_output = bias_hidden_output;

    final_total_cost = total_cost;

  end

end

% Plot cost curves

figure;

plot(mean_cost / 200);

figure;

plot(total_cost);

% Testing the model

correct_class_0 = 0;

correct_class_1 = 0;

for i = 1:10^6

  test_input_class_0 = test_class_0(:, i);

  test_input_class_1 = test_class_1(:, i);

  hidden_0 = final_weights_input_hidden * test_input_class_0 + final_bias_input_hidden;

  output_0 = Relu(hidden_0);

  final_output_0 = final_weights_hidden_output * output_0 + final_bias_hidden_output;

  hidden_1 = final_weights_input_hidden * test_input_class_1 + final_bias_input_hidden;

  output_1 = Relu(hidden_1);

  final_output_1 = final_weights_hidden_output * output_1 + final_bias_hidden_output;

  if final_output_1 < 0

       correct_class_0 = correct_class_0 + 1;

  end

  if final_output_0 > 0

       correct_class_1 = correct_class_1 + 1;

  end

end

% Error rate calculation

NN_Expon_error = 0.5 * (correct_class_0 / 10^6) + 0.5 * (correct_class_1 / 10^6);

fprintf('Final Exponential Error: %.5f\n', NN_Expon_error);
