# Bayesian Decision Theory & Neural Network Classification

**MATLAB** · University of Patras · Machine Learning coursework

MATLAB experiments comparing the Bayes likelihood-ratio decision rule with feed-forward neural classifiers using cross-entropy and exponential objectives, including MNIST 0-vs-8 classification.

## What this project covers

- Bayes decision rule
- probability density estimation
- 2–20–1 neural networks
- backpropagation
- cross-entropy vs exponential loss
- MNIST binary classification

## Repository layout

- Redacted course report (PDF) — included in the separate portfolio ZIP archive supplied in the accompanying ChatGPT conversation; not uploaded to this GitHub repository
- Source files:
  - `code/bayes_test_problem_1.m`
  - `code/cross_entropy_mnist_problem_2.m`
  - `code/cross_entropy_problem_1.m`
  - `code/exponential_mnist_problem_2.m`
  - `code/exponential_problem_1.m`
  - `code/helpers/crossEntropyLoss.m`
  - `code/helpers/derivativePhiCross.m`
  - `code/helpers/derivativePhiExp.m`
  - `code/helpers/derivativePsiCross.m`
  - `code/helpers/derivativePsiExp.m`
  - `code/helpers/derivativeRelu.m`
  - `code/helpers/derivativeSigmoid.m`
  - `code/helpers/expFunc.m`
  - `code/helpers/hingeLoss.m`
  - `code/helpers/phiCrossEntropy.m`
  - `code/helpers/phiExp.m`
  - `code/helpers/phiHinge.m`
  - `code/helpers/psiCrossEntropy.m`
  - `code/helpers/psiExp.m`
  - `code/helpers/psiHinge.m`
  - `code/helpers/reluFunc.m`
  - `code/helpers/sigmoidFunc.m`

## Getting started

In MATLAB, add the `code/helpers/` folder to your path, then open and run an individual `code/*.m` experiment. Some scripts generate their own simulated data; others expect the supplied course datasets.

### Data / software dependencies

- X0train.mat and X1train.mat (when referenced in scripts)
- mnist_train.csv and mnist_test.csv (for MNIST experiments)

The datasets and any large/generated artifacts have deliberately not been added to version control.

## Technical notes

The Bayes simulation provides a probabilistic reference against which neural classifiers are discussed; reported results are in the separate course-report PDF archive.

**Source provenance and execution status:** The original author confirmed that the source code in the assignment was tested. These standalone files were transcribed from the published PDF; visually wrapped lines may need reconciliation against the report before execution.

## Credits

Coursework authored by Spiros Bantis, University of Patras. The original course report (available in the separately provided project archive) remains the reference for the experimental setup and observed results. Refer to the assignment for technical details.


## Report availability

The original coursework report is provided separately as a privacy-redacted PDF in the downloadable project ZIP prepared for this portfolio. Its figures and results have **not** been uploaded to this repository. The source code is available above. The original source was tested by the author, but code transcribed from PDF into standalone MATLAB files has not been independently re-executed in this repository.
