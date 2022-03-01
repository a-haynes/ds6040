# This is code for Quadratic Discriminant Analysis
# Written by William F Basener
# University of Virginia, School of Data Science
# For use in teaching Bayesian Machine Learning
#
# The code currently computes the maximum likelihood classification
# Student is to add method to compute posterior probabilities and maximum probability classification

import pandas as pd
import numpy as np


def multivariate_gaussian_pdf(X, MU, SIGMA):
    """Code from Data Blog https://xavierbourretsicotte.github.io/MLE_Multivariate_Gaussian.html
    Maximum Likelihood Estimator: Multivariate Gaussian Distribution
        by Xavier Bourret Sicotte, Fri 22 June 2018
    Returns the pdf of a multivariate Gaussian distribution
     - X, MU are p x 1 vectors
     - SIGMA is a p x p matrix"""
    # Initialize and reshape
    X = X.reshape(-1, 1)
    MU = MU.reshape(-1, 1)
    p, _ = SIGMA.shape

    # Compute values
    SIGMA_inv = np.linalg.inv(SIGMA)
    denominator = np.sqrt((2 * np.pi) ** p * np.linalg.det(SIGMA))
    exponent = -(1 / 2) * ((X - MU).T @ SIGMA_inv @ (X - MU))

    # Return result
    return float((1. / denominator) * np.exp(exponent))


class QDA:
    """Creates a class for Quadratic Discriminant Analysis
    Input:
        fname = file name for a csv file, must have one column labeled "class" and the rest numeric data
    Methods:
        compute_probabilities = given an input observation computes the likelihood for each class and the GML class
        compute_probabilities: given an input observation and prior probabilities,
            computes the posterior probabilities for each class and most probable class"""

    def __init__(self, fname):
        # reads the data and computes the statistics needed for classification

        # read the iris data as a Pandas data frame
        df = pd.read_csv(fname)

        # separate the class labels from the rest of the data
        # we are assuming the column name with class labels is 'Class'
        # and all other columns are numeric
        self.data_labels = df.loc[:]['Class']
        self.data = np.asarray(df.drop('Class', axis=1, inplace=False))

        # get information about the dimensions the data
        self.num_rows, self.num_cols = self.data.shape

        # get the class names as an array of strings
        self.class_names = np.unique(self.data_labels)

        # determine number of observations in each class
        self.num_obs = dict()
        for name in self.class_names:
            self.num_obs[name] = sum(self.data_labels == name)

        # compute the mean of each class
        self.means = dict()
        for name in self.class_names:
            self.means[name] = np.mean(self.data[self.data_labels == name, :], 0)

        # compute the covariance matrix of each class
        self.covs = dict()
        for name in self.class_names:
            self.covs[name] = np.cov(np.transpose(self.data[self.data_labels == name, :]))

    def compute_likelihoods(self, x):
        # compute and output the likelihood of each class and the maximum likelihood class

        # check that the input data x has the correct number of rows
        if not (len(x) == self.num_cols):
            print('Data vector has wrong number of values.')
            return -1

        # reformat x as a numpy array, incase the user input a list
        x = np.asarray(x)

        # compute the likelihood of each class
        likelihoods = np.zeros(len(self.class_names))

        for idx, name in enumerate(self.class_names):
            likelihoods[idx] = multivariate_gaussian_pdf(x, self.means[name], self.covs[name])

        # return the likelihoods
        return likelihoods

    def compute_probabilities(self, x, priors):
        # compute and output the probability of each class and the maximum probability class

        # check that the input data x has the correct number of rows
        if not (len(x) == self.num_cols):
            print('Data vector has wrong number of values.')
            return -1

        if not (len(priors) == len(self.class_names)):
            print('Priors vector has wrong number of values.')
            return -1

        # reformat x as a numpy array, incase the user input a list
        x = np.asarray(x)

        # compute the likelihood of each class
        likelihoods = self.compute_likelihoods(x)
        probabilities = np.zeros(len(self.class_names))

        # process priors
        for idx, name in enumerate(self.class_names):
            probabilities[idx] = priors[name]

        denom = sum(likelihoods * probabilities)

        for idx, name in enumerate(self.class_names):
            probabilities[idx] = likelihoods[idx] * probabilities[idx] / denom

        # get the indices for sorting the likelihoods (in descending order)
        likelihood_sorted = np.argsort(likelihoods)[::-1]
        probabilities_sorted = np.argsort(probabilities)[::-1]

        # print the predicted class and all class likelihoods
        print('QDA Predicted Class: ' + self.class_names[likelihood_sorted[0]])
        print('QDA Class Likelihoods:')
        for idx in range(len(likelihood_sorted)):
            print(self.class_names[likelihood_sorted[idx]] + ': ' + str(likelihoods[likelihood_sorted[idx]]))

        # print the predicted class and all class likelihoods
        print('QDA Most Probable Class: ' + self.class_names[probabilities_sorted[0]])
        print('QDA Class Probabilities:')
        for idx in range(len(probabilities_sorted)):
            print(self.class_names[probabilities_sorted[idx]] + ': ' + str(probabilities[probabilities_sorted[idx]]))

        return probabilities


model_qda = QDA('iris_data.csv')

Iris_setosa_observation = [5.1, 3.5, 1.4, 0.2]
model_qda.compute_likelihoods(Iris_setosa_observation)

uninformative_priors = {
    "Iris-setosa": 1 / 3,
    "Iris-versicolor": 1 / 3,
    "Iris-virginica": 1 / 3
}
model_qda.compute_probabilities(Iris_setosa_observation, uninformative_priors)
ob_a = [5.5, 2.4, 3.8, 1.1]
ob_b = [5.5, 3.1, 5, 1.5]
print("\n\nQuestion 4.a")
model_qda.compute_probabilities(ob_a, uninformative_priors)
print("\n\nQuestion 4.b")
model_qda.compute_probabilities(ob_b, uninformative_priors)

updated_priors = {
    "Iris-setosa": 0.1,
    "Iris-versicolor": 0.2,
    "Iris-virginica": 0.7
}
print("\n\nQuestion 5.a")
model_qda.compute_probabilities(ob_a, updated_priors)
print("\n\nQuestion 5.b")
model_qda.compute_probabilities(ob_b, updated_priors)
