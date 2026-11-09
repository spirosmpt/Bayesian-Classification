clc;

clear;

% Δημιουργία δεδομένων για X0 και X1

X0 = randn(1e6, 2);

X1 = zeros(1e6, 2);

% Δημιουργία του X1 με τροποποιημένα δεδομένα

for idx = 1:1e6

  for dim = 1:2

    if randn() >= 0

      X1(idx, dim) = -1 + randn();

    else

      X1(idx, dim) = 1 + randn();

    end

  end

end

% Αποθήκευση των δεδομένων σε αρχεία

save("X0train", "X0");

save("X1train", "X1");

% Δημιουργία ιστογραμμάτων για τα X0 και X1

figure;

histogram(X0(:), 50);

title('Histogram for X0');

figure;

histogram(X1(:), 50);

title('Histogram for X1');

% Υπολογισμός σφαλμάτων για το σύνολο X0

error_count_X0 = 0;

for i = 1:1e6

  if gaussian_pdf_X0(X0(i,1), X0(i,2)) > gaussian_pdf_X1(X0(i,1), X0(i,2))

    error_count_X0 = error_count_X0 + 1;

  elseif gaussian_pdf_X0(X0(i,1), X0(i,2)) == gaussian_pdf_X1(X0(i,1), X0(i,2))

    if rand() > 0.5

      error_count_X0 = error_count_X0 + 1;

    end

  end

end

% Υπολογισμός σφαλμάτων για το σύνολο X1

error_count_X1 = 0;

for i = 1:1e6

  if gaussian_pdf_X0(X1(i,1), X1(i,2)) < gaussian_pdf_X1(X1(i,1), X1(i,2))

    error_count_X1 = error_count_X1 + 1;

  elseif gaussian_pdf_X0(X1(i,1), X1(i,2)) == gaussian_pdf_X1(X1(i,1), X1(i,2))

    if rand() > 0.5

      error_count_X1 = error_count_X1 + 1;

    end

  end

end

% Υπολογισμός του συνολικού σφάλματος Bayes

bayesian_error = 0.5 * (error_count_X0 / 1e6) + 0.5 * (error_count_X1 / 1e6);

disp(['Bayesian Error: ', num2str(bayesian_error)]);

% Συνάρτηση PDF για το X0

function pdf_val = gaussian_pdf_X0(x, y)

  mu_X0 = [0, 0];

  sigma_X0 = eye(2);

  pdf_val = exp(-0.5 * ([x, y] - mu_X0) * inv(sigma_X0) * ([x, y] - mu_X0)') / (2 * pi * sqrt(det(sigma_X0)));

end

% Συνάρτηση PDF για το X1

function pdf_val = gaussian_pdf_X1(x, y)

  mu_X1 = [1, 1];

  sigma_X1 = 2 * eye(2);

  pdf_val = exp(-0.5 * ([x, y] - mu_X1) * inv(sigma_X1) * ([x, y] - mu_X1)') / (2 * pi * sqrt(det(sigma_X1)));

end
