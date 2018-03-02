"""
Makes predictions for the mushroom-edibility dataset using scikit
"""

import numpy as np
#import pandas as pd
import csv
import sys
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt

def parse(order, x_dir, y_label):
    """
    Top layer wrapper for parsing txt files into scikit format

    Input
    -----
    order - List of x-data files
    x_dir - Directory of x-data files
    y_label - List of y-labels for x-data

    Output
    ------
    X - NxD matrix of x data
    Y - Nx1 vector of labels

    """
    # Training Data Order
    print("Data Order: " + order)
    ordering = [line.strip() for line in open(order)] 

    # Data and Label
    print("X data: " + x_dir)
    X = np.array([np.genfromtxt(x_dir + train) for train in ordering])
    print("Y labels: " + y_label + "\n")
    Y = np.genfromtxt(y_label)

    return X, Y


def perceptron(X, Y, output, devX = None, devY = None, testX = None):    
    """
    Running perceptron on X, Y
    Uses dev data if provided

    Input
    -----
    X - NxD matrix of training data
    Y - Dx1 matrix of training labels
    """
    
    # Train on perceptron and get the score on training data
    clf = Perceptron(penalty = 'l2', fit_intercept=False, max_iter=1000,
                        tol=None, shuffle=True).fit(X, Y)
    train_score = clf.score(X, Y)
    print("Perceptron score on training data: " + str(train_score) + "\n")

    # If dev data is provided
    if devX is not None and devY is not None:
        dev_score = clf.score(devX, devY)
        print("Perceptron score of training data on dev data: " + str(dev_score) + "\n")

    # If test data is provided
    if testX is not None:
        print("Making Predictions on Test data")
        prediction = clf.predict(testX)
        print("Writing Predictions to " + output)
        with open(output, "w") as f:
            for line in prediction:
                f.write(str(int(line)) + "\n")
        print("Write Complete")

    # Print out the weights
    #print("Coefs: " + str(clf.coef_))
    #print("Intercept: " + str(clf.intercept_))    


def main():
    print("\n" + "Parsing Training Data")
    X, Y = parse(sys.argv[1], sys.argv[2], sys.argv[3])

    # Prepare dev data if there is any
    devX = None
    devY = None
    if len(sys.argv) >= 7:
        print("Parsing Development Data")
        devX, devY = parse(sys.argv[4], sys.argv[5], sys.argv[6])

    # Prepare test data if any
    testX = None
    if len(sys.argv) == 9:
        print("Parsing Test Data")
        print("Data Order: " + sys.argv[7])
        ordering = [line.strip() for line in open(sys.argv[7])]
        print("X data: " + sys.argv[8] + "\n")
        testX = np.array([np.genfromtxt(sys.argv[8] + train) for train in ordering])
    
    perceptron(X, Y, "40.7.test-yhat", devX, devY, testX)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 40.mushroom-edibility.learn.py" +
                " <training order> <training directory> <training label> " +
                " <dev order> <dev directory> <dev label> [test x] [test dir]")
        sys.exit(1)

    main()
