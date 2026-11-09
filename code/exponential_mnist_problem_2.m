clc;

clear;

% Φόρτωση δεδομένων από το MNIST

train_data = load('mnist_train.csv');

test_data = load('mnist_test.csv');

% Διαχωρισμός των δεδομένων για αριθμούς 0 και 8 (ίσες ποσότητες)

train_data_0 = zeros(5500, 784);

train_data_8 = zeros(5500, 784);

idx_0 = 0;

idx_8 = 0;

% Επεξεργασία των δεδομένων εκπαίδευσης

for i = 1:60000

  if train_data(i,1) == 0 && idx_0 < 5500

       train_data_0(idx_0 + 1,:) = train_data(i,2:785);

       idx_0 = idx_0 + 1;

  elseif train_data(i,1) == 8 && idx_8 < 5500

       train_data_8(idx_8 + 1,:) = train_data(i,2:785);

       idx_8 = idx_8 + 1;

  end

  if idx_0 + idx_8 == 11000

       break;

  end

end

% Διαχωρισμός δεδομένων δοκιμής για αριθμούς 0 και 8

test_data_0 = zeros(950, 784);

test_data_8 = zeros(950, 784);

idx_0 = 0;

idx_8 = 0;

for i = 1:10000

  if test_data(i,1) == 0 && idx_0 < 950

    test_data_0(idx_0 + 1,:) = test_data(i,2:785);

    idx_0 = idx_0 + 1;

  elseif test_data(i,1) == 8 && idx_8 < 950

    test_data_8(idx_8 + 1,:) = test_data(i,2:785);

    idx_8 = idx_8 + 1;

  end

  if idx_0 + idx_8 == 1900

    break;

  end

end

% Κανονικοποίηση των δεδομένων εικόνας

train_data_0 = train_data_0 / 255;

train_data_8 = train_data_8 / 255;

test_data_0 = test_data_0 / 255;

test_data_8 = test_data_8 / 255;

% Ρυθμίσεις και παραμέτροι Νευρωνικού Δικτύου

input_dim = 784;

hidden_units = 300;

output_dim = 1;

learning_rate = 2e-4;

lambda = 1e-5;

smoothing_factor = 1;

% Αριθμός επαναλήψεων και δεδομένων

epochs = 10;

train_samples = 5500;

test_samples = 950;

% Αρχικοποίηση βαρών και παραμέτρων

best_cost = inf;

for epoch = 1:epochs

 % Τυχαία αρχικοποίηση βαρών και παραμέτρων

 W1 = (1 / (hidden_units + input_dim)) * randn(hidden_units, input_dim);

 b1 = zeros(hidden_units, 1);

 W2 = (1 / (hidden_units + output_dim)) * randn(output_dim, hidden_units);

 b2 = 0;

 cost_history = zeros(1, epochs * train_samples);

 sample_counter = 1;

 for i = 1:train_samples

   % Επεξεργασία δεδομένων για 0 και 8

   x0 = train_data_0(i,:)';

   x8 = train_data_8(i,:)';

   % Προώθηση στο Νευρωνικό Δίκτυο (Forward Pass)

   z1_0 = W1 * x0 + b1;

   h1_0 = Relu(z1_0);

   z2_0 = W2 * h1_0 + b2;

   y0 = z2_0;

      z1_8 = W1 * x8 + b1;

      h1_8 = Relu(z1_8);

      z2_8 = W2 * h1_8 + b2;

      y8 = z2_8;

      % Υπολογισμός κόστους

      cost_history(sample_counter) = Expon(y0, y8);

      % Υπολογισμός παραγώγων για την ανατροφοδότηση (Backpropagation)

      delta2_0 = diff_Phi_expon(y0);

      delta2_8 = diff_Psi_expon(y8);

      delta1_0 = (W2') * delta2_0 .* diff_Relu(z1_0);

      delta1_8 = (W2') * delta2_8 .* diff_Relu(z1_8);

      if i == 1 && epoch == 1

        P2 = (delta2_0 * [h1_0' 1] + delta2_8 * [h1_8' 1]) .^ 2;

      else

     P2 = (1 - lambda) * P2_old + lambda * power((delta2_0 * [h1_0' 1] + delta2_8 * [h1_8' 1]), 2);

      end

      P2_old = P2;

      if i == 1 && epoch == 1

        P1 = (delta1_0 * [x0' 1] + delta1_8 * [x8' 1]) .^ 2;

      else

        P1 = (1 - lambda) * P1_old + lambda * power((delta1_0 * [x0' 1] + delta1_8 * [x8' 1]), 2);

      end

      P1_old = P1;

      % Ενημέρωση βαρών (Weight Update)

    AB2 = [W2 b2] - learning_rate * ((delta2_0 * [h1_0' 1] + delta2_8 * [h1_8' 1]) ./ sqrt(smoothing_factor + P2));

    W2 = AB2(1:hidden_units);

    b2 = AB2(end);

    AB1 = [W1 b1] - learning_rate * ((delta1_0 * [x0' 1] + delta1_8 * [x8' 1]) ./ sqrt(smoothing_factor + P1));

    W1 = AB1(1:hidden_units, 1:input_dim);

    b1 = AB1(1:hidden_units, end);

    sample_counter = sample_counter + 1;

  end

  % Υπολογισμός μέσου κόστους για την εποχή

  avg_cost = mean(cost_history(end-train_samples+1:end));

  if best_cost > avg_cost

    best_cost = avg_cost;

    best_W1 = W1;

    best_W2 = W2;

    best_b1 = b1;

    best_b2 = b2;

    final_cost_history = cost_history;

  end

end

% Υπολογισμός και εμφάνιση του μέσου κόστους για τις επαναλήψεις

mean_cost = mean(reshape(final_cost_history, train_samples, epochs), 1);

figure;

plot(mean_cost);

% Έλεγχος του μοντέλου με τα δεδομένα δοκιμής

correct_0 = 0;

correct_8 = 0;

for i = 1:test_samples

 x0 = test_data_0(i,:)';

 x8 = test_data_8(i,:)';

 % Υπολογισμός εξόδου για το 0

 y0 = W2 * Relu(W1 * x0 + b1) + b2;

 if y0 > 0

   correct_0 = correct_0 + 1;

 end

 % Υπολογισμός εξόδου για το 8

 y8 = W2 * Relu(W1 * x8 + b1) + b2;

 if y8 < 0

   correct_8 = correct_8 + 1;

 end

end

% Υπολογισμός του σφάλματος του Νευρωνικού Δικτύου

NN_error = 0.5 * (correct_8 / test_samples) + 0.5 * (correct_0 / test_samples);

disp(['NN Error: ', num2str(NN_error)]);
