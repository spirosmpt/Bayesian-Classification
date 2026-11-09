clc;

clear;

close all;

% Φόρτωση δεδομένων για τα δεδομένα εκπαίδευσης και δοκιμών

test_data_0 = (load("X0train.mat","X0").X0)';

test_data_1 = (load("X1train.mat","X1").X1)';

% Δημιουργία τυχαίων δεδομένων για εκπαίδευση

train_data_0 = randn(2,200);

train_data_1 = zeros(2,200);

for i = 1:2

  for j = 1:200

       if randn() >= 0

         train_data_1(i,j) = -1 + randn();

       else

         train_data_1(i,j) = 1 + randn();

       end

  end

end

% Αρχικοποίηση παραμέτρων για το Νευρωνικό Δίκτυο

num_input = 2;

num_hidden = 20;

num_output = 1;

learning_rate = 2e-4;

regularization_lambda = 0.01;

epsilon = 1e-315;

num_epochs = 5000;

min_cost = inf;

% Βρόχος για εκπαίδευση

for iteration = 1:10

  % Αρχικοποίηση βαρών και παραμέτρων

  weight_ih = (1/(num_hidden + num_input)) * randn(num_hidden, num_input);

  bias_ih = zeros(num_hidden,1);

  weight_ho = (1/(num_hidden + num_output)) * randn(num_output, num_hidden);

  bias_ho = 0;

  current_epoch = 1;

  cost_history = zeros(1, num_epochs * 200);

  % Εκπαίδευση για τον αριθμό των εποχών

  for epoch = 1:num_epochs

    for sample = 1:200

      % Προώθηση δεδομένων για την κλάση 0

      input_0 = train_data_0(:,sample);

      input_1 = train_data_1(:,sample);

     % Υπολογισμός του πρώτου και δεύτερου επιπέδου του νευρωνικού δικτύου

     layer1_0 = weight_ih * input_0 + bias_ih;

     output_1_0 = relu(layer1_0);

     layer2_0 = weight_ho * output_1_0 + bias_ho;

     predicted_0 = sigmoid(layer2_0);

     layer1_1 = weight_ih * input_1 + bias_ih;

     output_1_1 = relu(layer1_1);

     layer2_1 = weight_ho * output_1_1 + bias_ho;

     predicted_1 = sigmoid(layer2_1);

     % Υπολογισμός κόστους

     cost_history(current_epoch) = cross_entropy_loss(predicted_0, predicted_1);

     % Υπολογισμός παραγώγων για το backpropagation

     delta_0 = derivative_phi_cross_entropy(predicted_0);

     grad_output_0 = delta_0 .* derivative_sigmoid(layer2_0);

     grad_hidden_0 = (weight_ho') * grad_output_0;

     grad_hidden_0 = grad_hidden_0 .* derivative_relu(layer1_0);

     delta_1 = derivative_psi_cross_entropy(predicted_1);

     grad_output_1 = delta_1 .* derivative_sigmoid(layer2_1);

     grad_hidden_1 = (weight_ho') * grad_output_1;

     grad_hidden_1 = grad_hidden_1 .* derivative_relu(layer1_1);

     % Κανονικοποίηση κόστους

     if (sample == 1) && (epoch == 1)

       regularization_2 = ((grad_output_0 * [output_1_0' 1]) + (grad_output_1 * [output_1_1' 1])) .^ 2;

     else

        regularization_2 = (1 - regularization_lambda) * prev_regularization_2 + regularization_lambda * power(((grad_output_0 * [output_1_0' 1]) + (grad_output_1 *
[output_1_1' 1])), 2);

        end

        prev_regularization_2 = regularization_2;

        if (sample == 1) && (epoch == 1)

         regularization_1 = ((grad_hidden_0 * [input_0' 1]) + (grad_hidden_1 * [input_1' 1]))
.^ 2;

        else

        regularization_1 = (1 - regularization_lambda) * prev_regularization_1 + regularization_lambda * power(((grad_hidden_0 * [input_0' 1]) + (grad_hidden_1 *
[input_1' 1])), 2);

        end

        prev_regularization_1 = regularization_1;

        % Ενημέρωση των παραμέτρων

       weight_ho = [weight_ho bias_ho] - learning_rate * (((grad_output_0 * [output_1_0'
1]) + (grad_output_1 * [output_1_1' 1])) ./ sqrt(epsilon + regularization_2));

     weight_ih = [weight_ih bias_ih] - learning_rate * (((grad_hidden_0 * [input_0' 1]) + (grad_hidden_1 * [input_1' 1])) ./ sqrt(epsilon + regularization_1));

        weight_ho = weight_ho(1:num_hidden);

        bias_ho = weight_ho(1, num_hidden+1);

        weight_ih = weight_ih(1:num_hidden, 1:num_input);

        bias_ih = weight_ih(1:num_hidden, num_hidden+1);

        current_epoch = current_epoch + 1;

    end

  end

  % Υπολογισμός του μέσου κόστους

  mean_cost = 0;

  for sample = 1:200

       mean_cost = (mean_cost + cost_history(num_epochs * 200 - 200 + sample));

  end

  mean_cost = mean_cost / 200;

  % Αποθήκευση των καλύτερων παραμέτρων

  if min_cost > mean_cost

       min_cost = mean_cost;

       best_weight_ih = weight_ih;

       best_weight_ho = weight_ho;

       best_bias_ih = bias_ih;

       best_bias_ho = bias_ho;

       final_cost_history = cost_history;

  end

  % Εκτύπωση για παρακολούθηση της εκπαίδευσης

  fprintf('Iteration %d complete with cost: %f\n', iteration, min_cost);

end

% Υπολογισμός και αποτύπωση του μέσου κόστους

final_cost_avg = zeros(1, num_epochs);

cost_index = 1;

for i = 0:200:length(final_cost_history) - 200

  for l = 0:199

       final_cost_avg(cost_index) = (final_cost_avg(cost_index) + final_cost_history(i + l + 1));

  end

  cost_index = cost_index + 1;

end

figure;

plot(final_cost_avg / 200);

figure;

plot(final_cost_history);

% Δοκιμή του Νευρωνικού Δικτύου

correct_0 = 0;

correct_1 = 0;

fprintf('Testing the Neural Network...\n');

for test_sample = 1:10^6

  test_input_0 = test_data_0(:,test_sample);

  test_input_1 = test_data_1(:,test_sample);

  hidden_layer_input_0 = best_weight_ih * test_input_0 + best_bias_ih;

  hidden_layer_output_0 = relu(hidden_layer_input_0);

  output_layer_input_0 = best_weight_ho * hidden_layer_output_0 + best_bias_ho;

  predicted_output_0 = sigmoid(output_layer_input_0);

  hidden_layer_input_1 = best_weight_ih * test_input_1 + best_bias_ih;

  hidden_layer_output_1 = relu(hidden_layer_input_1);

  output_layer_input_1 = best_weight_ho * hidden_layer_output_1 + best_bias_ho;

  predicted_output_1 = sigmoid(output_layer_input_1);

  if predicted_output_1 < 0.5

    correct_1 = correct_1 + 1;

  end

  if predicted_output_0 > 0.5

       correct_0 = correct_0 + 1;

  end

end

nn_error = 0.5 * (correct_1 / 10^6) + 0.5 * (correct_0 / 10^6);

fprintf('Final Neural Network Error: %.4f\n', nn_error);
