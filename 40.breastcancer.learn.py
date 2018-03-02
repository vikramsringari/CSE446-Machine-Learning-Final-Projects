"""
Makes predictions for the breast cancer dataset using scikit
"""

import numpy as np
import pandas as pd
import csv
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import tree

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
    discreteX = np.zeros((len(X), 55), dtype=np.int);
    for i in range(len(X)):
        discreteX[i][0] = 0
        discreteX[i][1] = 0
        discreteX[i][int(X[i][2]) - 1 + 2] = 1

        if(X[i][3] == 9):
            discreteX[i][6 + 15] = 1
        else:
            discreteX[i][int(X[i][3]) + 15 - 1] = 1

        if(X[i][4] == 9):
            discreteX[i][2 + 22] = 1
        else:
            discreteX[i][int(X[i][4])+ 22] = 1

        if(X[i][5] == 9):
            discreteX[i][3 + 25] = 1
        else:
            discreteX[i][int(X[i][5])+ 25] = 1

        if(X[i][6] == 9):
            discreteX[i][5 + 29] = 1
        else:
            discreteX[i][int(X[i][6])+ 29] = 1

        if(X[i][7] == 9):
            discreteX[i][4 + 35] = 1
        else:
            discreteX[i][int(X[i][7]) - 1 + 35] = 1

        if(X[i][8] == 9):
            discreteX[i][2 + 40] = 1
        else:
            discreteX[i][int(X[i][8]) + 40] = 1

        if(X[i][9] == 9):
            discreteX[i][3 + 43] = 1
        else:
            discreteX[i][int(X[i][9]) - 1 + 43] = 1

        if(X[i][10] == 9):
            discreteX[i][4 + 47] = 1
        else:
            discreteX[i][int(X[i][10]) - 1 + 47] = 1

        if(X[i][11] == 9):
            discreteX[i][2 + 52] = 1
        else:
            discreteX[i][int(X[i][11]) + 52] = 1

    return discreteX, Y


def DTree(X, Y, output, testX = None):    
    """
    Running DTree on X, Y
    Uses dev data if provided

    Input
    -----
    X - NxD matrix of training data
    Y - Dx1 matrix of training labels
    output - name of y-hat file
    textX - test X data
    """

    '''params = {"hidden_layer_sizes": [i for i in range(50, 101, 10)], "alpha": [i/100000 for i in range(1, 101)], "learning_rate_init": [i/10000 for i in range(1, 101)]}
    cl = MLPClassifier(activation = "relu", solver = "adam", verbose = True, early_stopping =True)
    '''    

    trains = []
    devs = []
    tests = []
    trainX, devX, trainY, devY = train_test_split(X, Y, test_size = .2)
    for i in range(1, 21):
        print("i = " + str(i))
        clf = tree.DecisionTreeClassifier(max_depth=i).fit(trainX, trainY)

        train_score = clf.score(trainX, trainY)
        trains.append(train_score)
        print("Score on training data: " + str(train_score) + "\n")

        dev_score = clf.score(devX, devY)
        devs.append(dev_score)
        print("Score on dev data: " + str(dev_score) + "\n")
        
        if testX is not None:
            tests.append(clf.predict(testX))

    best_depth = np.argmax(devs)
    print(str(best_depth + 1))

    # If test data is provided
    if testX is not None:
        print("Writing Predictions to " + output)
        with open(output, "w") as f:
            for line in tests[best_depth]:
                f.write(str(int(line)) + "\n")
        print("Write Complete")

    df = pd.DataFrame({"Train Accuracy" : trains, "Dev. Accuracy" : devs},
                     index = range(1, 21),
                     columns = ["Train Accuracy", "Dev. Accuracy"])
    plot = df.plot()
    plot.set_xlabel("Max depth")
    plot.set_ylabel("Accuracy")
    plot.set_title("Breast Cancer Accuracy by Max Depth")
    plt.show()


def main():
    print("\n" + "Parsing Training Data")
    X, Y = parse(sys.argv[1], sys.argv[2], sys.argv[3])

    # Prepare test data if any
    testX = None
    if len(sys.argv) == 6:
        print("Parsing Test Data")
        print("Data Order: " + sys.argv[4])
        ordering = [line.strip() for line in open(sys.argv[4])]
        print("X data: " + sys.argv[5] + "\n")
        testX = np.array([np.genfromtxt(sys.argv[5] + train) for train in ordering])

        discreteX = np.zeros((len(testX), 55), dtype=np.int);
        for i in range(len(testX)):
            discreteX[i][0] = 0
            discreteX[i][1] = 0
            discreteX[i][int(testX[i][2]) - 1 + 2] = 1

            if(testX[i][3] == 9):
                discreteX[i][6 + 15] = 1
            else:
                discreteX[i][int(testX[i][3]) + 15 - 1] = 1

            if(testX[i][4] == 9):
                discreteX[i][2 + 22] = 1
            else:
                discreteX[i][int(testX[i][4])+ 22] = 1

            if(testX[i][5] == 9):
                discreteX[i][3 + 25] = 1
            else:
                discreteX[i][int(testX[i][5])+ 25] = 1

            if(testX[i][6] == 9):
                discreteX[i][5 + 29] = 1
            else:
                discreteX[i][int(testX[i][6])+ 29] = 1

            if(testX[i][7] == 9):
                discreteX[i][4 + 35] = 1
            else:
                discreteX[i][int(testX[i][7]) - 1 + 35] = 1

            if(testX[i][8] == 9):
                discreteX[i][2 + 40] = 1
            else:
                discreteX[i][int(testX[i][8]) + 40] = 1

            if(testX[i][9] == 9):
                discreteX[i][3 + 43] = 1
            else:
                discreteX[i][int(testX[i][9]) - 1 + 43] = 1

            if(testX[i][10] == 9):
                discreteX[i][4 + 47] = 1
            else:
                discreteX[i][int(testX[i][10]) - 1 + 47] = 1

            if(testX[i][11] == 9):
                discreteX[i][2 + 52] = 1
            else:
                discreteX[i][int(testX[i][11]) + 52] = 1
    
    DTree(X, Y, "40.42.test-yhat", discreteX)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 40.breastcancer.learn.py" +
                " <training order> <training directory> <training label> " +
                " [test x] [test dir]")
        sys.exit(1)

    main()
