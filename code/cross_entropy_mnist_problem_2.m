clc;

clear;

% Φόρτωση δεδομένων εκπαίδευσης και δοκιμής

train_data_set = load('mnist_train.csv');

test_data_set = load('mnist_test.csv');

% Επιλογή μόνο των ψηφίων 0 και 8 από τα δεδομένα εκπαίδευσης (5500 δείγματα το καθένα)

train_zero_images = zeros(5500, 784);

train_eight_images = zeros(5500, 784);

zero_count = 0;

eight_count = 0;

for idx = 1:60000

 if train_data_set(idx, 1) == 0 && zero_count < 5500

   train_zero_images(zero_count + 1, :) = train_data_set(idx, 2:785);

   zero_count = zero_count + 1;

 end

 if train_data_set(idx, 1) == 8 && eight_count < 5500

   train_eight_images(eight_count + 1, :) = train_data_set(idx, 2:785);

   eight_count = eight_count + 1;

 end

 if zero_count + eight_count == 11000

   break;

 end

end

% Επιλογή μόνο των ψηφίων 0 και 8 από τα δεδομένα δοκιμής (950 δείγματα το καθένα)

test_zero_images = zeros(950, 784);

test_eight_images = zeros(950, 784);

zero_test_count = 0;

eight_test_count = 0;

for idx = 1:10000

 if test_data_set(idx, 1) == 0 && zero_test_count < 950

   test_zero_images(zero_test_count + 1, :) = test_data_set(idx, 2:785);

    zero_test_count = zero_test_count + 1;

 end

 if test_data_set(idx, 1) == 8 && eight_test_count < 950

   test_eight_images(eight_test_count + 1, :) = test_data_set(idx, 2:785);

   eight_test_count = eight_test_count + 1;

 end

 if zero_test_count + eight_test_count == 1948

   break;

 end

end

% Κανονικοποίηση των εικόνων (τιμές pixel μεταξύ 0 και 1)

train_zero_images = train_zero_images / 255;

train_eight_images = train_eight_images / 255;

test_zero_images = test_zero_images / 255;

test_eight_images = test_eight_images / 255;

% Αρχικοποίηση παραμέτρων για το νευρωνικό δίκτυο

input_size = 784; % Μέγεθος εισόδου (28x28 pixels)

hidden_size = 300; % Μέγεθος κρυφής στρώσης

output_size = 1; % Μέγεθος εξόδου (δυαδική κατηγοριοποίηση)

learning_rate = 2 * 0.0001; % Ρυθμός μάθησης

reg_param = 0.00001; % Παράμετρος κανονικοποίησης

epsilon = 1e-310; % Μικρή σταθερά για αριθμητική σταθερότητα

% Ρυθμίσεις εκπαίδευσης

num_epochs = 10;

best_loss = inf;

train_samples = 5500;

test_samples = 950;

% Αρχικοποίηση βαρών και παραμέτρων

W1 = (1 / (hidden_size + input_size)) * randn(hidden_size, input_size);

b1 = zeros(hidden_size, 1);

W2 = (1 / (hidden_size + output_size)) * randn(output_size, hidden_size);

b2 = 0;

% Επανάληψη εκπαίδευσης (στοχαστική καθοδική κλίση)

for epoch = 1:num_epochs

 epoch_costs = zeros(1, train_samples);

 for sample_idx = 1:train_samples

    % Προώθηση για τα ψηφία 0 και 8

    image_zero = train_zero_images(sample_idx, :)';

    image_eight = train_eight_images(sample_idx, :)';

    % Προώθηση για το ψηφίο '0'

    hidden_zero = W1 * image_zero + b1;

   relu_zero = Relu(hidden_zero);

   output_zero = W2 * relu_zero + b2;

   prediction_zero = Sigmoid(output_zero);

    % Προώθηση για το ψηφίο '8'

    hidden_eight = W1 * image_eight + b1;

   relu_eight = Relu(hidden_eight);

   output_eight = W2 * relu_eight + b2;

   prediction_eight = Sigmoid(output_eight);

    % Υπολογισμός κόστους χρησιμοποιώντας συνάρτηση Cross-Entropy

   epoch_costs(sample_idx) = CrossEntropy(prediction_zero, prediction_eight);

     % Υπολογισμός παραγώγων για ανατροφοδότηση (backpropagation)

     grad_output_zero = diff_Phi_CrossEntropy(prediction_zero);

     grad_relu_zero = grad_output_zero .* diff_Sigmoid(output_zero);

     grad_W2_zero = grad_relu_zero * relu_zero';

     grad_b2_zero = grad_relu_zero;

     grad_hidden_zero = (W2') * grad_relu_zero .* diff_Relu(hidden_zero);

     grad_W1_zero = grad_hidden_zero * image_zero';

     grad_output_eight = diff_Psi_CrossEntropy(prediction_eight);

     grad_relu_eight = grad_output_eight .* diff_Sigmoid(output_eight);

     grad_W2_eight = grad_relu_eight * relu_eight';

     grad_b2_eight = grad_relu_eight;

     grad_hidden_eight = (W2') * grad_relu_eight .* diff_Relu(hidden_eight);

     grad_W1_eight = grad_hidden_eight * image_eight';

     % Ενημέρωση κανονικοποίησης

     if (sample_idx == 1) && (epoch == 1)

      P2 = (grad_relu_zero + grad_relu_eight) .^ 2;

     else

      P2 = (1 - reg_param) * prev_P2 + reg_param * (grad_relu_zero + grad_relu_eight) .^
2;

     end

     prev_P2 = P2;

     if (sample_idx == 1) && (epoch == 1)

      P1 = (grad_hidden_zero + grad_hidden_eight) .^ 2;

     else

     P1 = (1 - reg_param) * prev_P1 + reg_param * (grad_hidden_zero + grad_hidden_eight) .^ 2;

    end

    prev_P1 = P1;

    % Ενημέρωση βαρών και παραμέτρων μέσω της στοχαστικής καθοδικής κλίσης

    W2 = W2 - learning_rate * (grad_W2_zero + grad_W2_eight) / sqrt(epsilon + P2);

    b2 = b2 - learning_rate * (grad_b2_zero + grad_b2_eight) / sqrt(epsilon + P2);

    W1 = W1 - learning_rate * (grad_W1_zero + grad_W1_eight) / sqrt(epsilon + P1);

    b1 = b1 - learning_rate * (grad_hidden_zero + grad_hidden_eight) / sqrt(epsilon + P1);

  end

  % Υπολογισμός μέσου κόστους για την τρέχουσα εποχή

  mean_epoch_cost = mean(epoch_costs);

  fprintf('Εποχή: %d, Μέσο Κόστος: %.4f\n', epoch, mean_epoch_cost);

  if best_loss > mean_epoch_cost

    best_loss = mean_epoch_cost;

    best_W1 = W1;

    best_W2 = W2;

    best_b1 = b1;

    best_b2 = b2;

  end

end

% Εμφάνιση του κόστους

figure;

plot(epoch_costs);

title('Κόστος ανά Επανάληψη');

% Αξιολόγηση του εκπαιδευμένου νευρωνικού δικτύου στα δεδομένα δοκιμής

correct_zeros = 0;

correct_eights = 0;

for test_idx = 1:test_samples

 test_image_zero = test_zero_images(test_idx, :)';

 test_image_eight = test_eight_images(test_idx, :)';

 % Δοκιμή για το ψηφίο '0'

 test_hidden_zero = best_W1 * test_image_zero + best_b1;

 test_relu_zero = Relu(test_hidden_zero);

 test_output_zero = best_W2 * test_relu_zero + best_b2;

 test_pred_zero = Sigmoid(test_output_zero);

 % Δοκιμή για το ψηφίο '8'

 test_hidden_eight = best_W1 * test_image_eight + best_b1;

 test_relu_eight = Relu(test_hidden_eight);

 test_output_eight = best_W2 * test_relu_eight + best_b2;

 test_pred_eight = Sigmoid(test_output_eight);

 if test_pred_eight < 0.5

   correct_zeros = correct_zeros + 1;

 end

 if test_pred_zero > 0.5

   correct_eights = correct_eights + 1;

 end

end

% Υπολογισμός του Cross-Entropy λάθους στα δεδομένα δοκιμής

final_test_error = 0.5 * (correct_zeros / test_samples) + 0.5 * (correct_eights / test_samples);

fprintf('Τελικό Σφάλμα Cross-Entropy στο Σετ Δοκιμής: %.4f\n', final_test_error);
