import ast
import os
import sys
from collections import Counter
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split


def transformX(Y):
    '''
    Naive Bayes on the review for a value of how real
    '''
    X = []
    ordering = [line.strip() for line in open(sys.argv[1])]
    for train in ordering:
        with open(sys.argv[2] + train) as f:
            X.append(ast.literal_eval(f.read().replace("\n", "")))

    testX = None

    if len(sys.argv) == 6:
        testX = []
        testOrdering = [line.strip() for line in open(sys.argv[4])]
        for train in testOrdering:
            with open(sys.argv[5] + train) as f:
                temp = ast.literal_eval(f.read().replace("\n", ""))
                temp[0] = int(temp[0])
                testX.append(temp)

    fake = []
    real = []

    for i in range(len(Y)):
        if Y[i] == 0:
            real.append(X[i][1])
        else:
            fake.append(X[i][1])

    fakeCounter = Counter()
    realCounter = Counter()

    for sentence in fake:
        sen = sentence.replace("\n", "")
        tokens = sen.split(" ")
        for word in tokens:
            fakeCounter[word] += 1 
        
    for sentence in real:
        sen = sentence.replace("\n", "")
        tokens = sen.split(" ")
        for word in tokens:
            realCounter[word] += 1 

    pFake = len(fake) / (len(fake) + len(real))
    pReal = len(real) / (len(fake) + len(real))

    for data in X:
        # Probability of Rating of Product
        data.append(data[2][data[0] - 1] / sum(data[2]))
        data.append(data[3][data[0] - 1] / sum(data[3]))

        # Discretize the probability
        for dist in data[2]:
            data.append(dist)
        for dist in data[2]:
            data.append(dist)
        del data[3]
        del data[2]

        # Naive Bayes for Realness of a Message
        sen = data[1].replace("\n", "").split(" ")
        pRealGivenWords = 0
        pFakeGivenWords = 0
        for word in sen:
            pRealGivenWords += math.log10((float(realCounter[word]) + 1.0)/(len(real) + 2.0))
            pFakeGivenWords += math.log10((float(fakeCounter[word]) + 1.0)/(len(fake) + 2.0))
        
        data.append(math.log10(pReal) + pRealGivenWords - math.log10(pFake) - pFakeGivenWords)
        del data[1] 

    if testX is not None:
        for data in testX:
            # Probability of Rating of Product
            data.append(data[2][data[0] - 1] / sum(data[2]))
            data.append(data[3][data[0] - 1] / sum(data[3]))

            # Discretize the probability
            for dist in data[2]:
                data.append(dist)
            for dist in data[2]:
                data.append(dist)
            del data[3]
            del data[2]

            # Naive Bayes for Realness of a Message
            sen = data[1].replace("\n", "").split(" ")
            pRealGivenWords = 0
            pFakeGivenWords = 0
            for word in sen:
                pRealGivenWords += math.log10((float(realCounter[word]) + 1.0)/(len(real) + 2.0))
                pFakeGivenWords += math.log10((float(fakeCounter[word]) + 1.0)/(len(fake) + 2.0))
        
            data.append(math.log10(pReal) + pRealGivenWords - math.log10(pFake) - pFakeGivenWords)
            del data[1] 
        return np.array(X), np.array(testX)

    return np.array(X), None


def MLP(X, Y, output, testX = None):
    """
    Running MLP on X, Y
    Uses test data if provided

    Input
    -----
    X - NxD matrix of training data
    Y - Dx1 matrix of training labels
    output - name of yhat file
    testX - testX files
    """
    
    '''params = {"hidden_layer_sizes" : [(i, j, k, l, m, n, o, p, q, r) for i in range(10, 101, 10) 
                                                                     for j in range(2, 53, 5) 
                                                                     for k in range(2, 28, 5)
                                                                     for l in range(2, 18, 3)
                                                                     for m in range(2, 11, 2)
                                                                     for n in range(2, 11, 2)
                                                                     for o in range(2, 11, 2)
                                                                     for p in range(2, 11, 2)
                                                                     for q in range(2, 11, 2)
                                                                     for r in range(2, 11, 2)]}

    cl = MLPClassifier(alpha=.00001, random_state=1, max_iter=300)
    clf = RandomizedSearchCV(cl, params, n_iter=50).fit(X, Y)
    train_score = clf.score(X, Y)
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
    plot.set_title("FraudBGone Accuracy by Max Depth")
    plt.show()


def main(): 
    Y = np.genfromtxt(sys.argv[3])
    X, testX = transformX(Y)
    
    MLP(X, Y, "40.22.test-yhat", testX)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 40.FraudBGone.learn.py" +
                " <training order> <training directory> <training label> <test order> <test directory>")
        sys.exit(1)

    main()
