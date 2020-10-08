# digit-recogniser

# Description
First version of a supervised learning program to classify handwritten digits. Currently includes two models: a logistic regression and a neural network model, both of which are regularised. The program reads data from a .csv file and randomly allocates data to training, cross-validation and testing sets. The cross-validation set is used to pick the regularisation parameters of the models.

# Instructions
Set FILENAME constant to the path of the file containing the training/testing data. The file should be a .csv file where each row contains values for the input features (e.g. greyscale values) in seperate columns followed by the expected output value (i.e. a digit from 0 to 10). An example is given in data.csv in this repository. Various implementational choices can be modified by changing the values of the global constants (e.g. potential regularisation constants, number of classes (need not classify data into 10 digits) and the proportions of the data assigned to the training, cross-validation and testing sets.

# Potential Future Improvements
- Add an SVM model
- Improve clarity of docstrings
- Remove redundancy in code
- Allow user to draw digits and have models classify them
