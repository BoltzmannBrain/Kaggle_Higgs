Kaggle_Higgs
============

Higgs Boson ML Cometition from Kaggle.com.
Runs a gradient boost classifier in Python.

To train the model, run higgsml-train with input training dataset named 'training.csv'. Output is AMS scores for both the training and validation data samples (printed to the command line).

To classify the test data with the model, run higgsml-run with input testing dataset named 'test.csv'. Output is the classifications for the test data, named 'predictions.csv' and saved to the directory.

Required software and packages:
  - Python 2.7
  - Numpy
  - Scikit-learn
  - Pandas
