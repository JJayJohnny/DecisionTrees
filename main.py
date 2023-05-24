import pandas as pd
import c45
import id3New
from sklearn.model_selection import train_test_split
from sklearn import tree
import time
import matplotlib.pyplot as plt

dataLoc='data/waterQuality1.csv'

def GetData(location):
    data = pd.read_csv(location)
    return data

if __name__ == '__main__':
    print("Main")
    fullData = GetData(dataLoc)

    xAxis = []

    id3Times = []
    c45Times = []
    libTreeTimes = []

    id3TestAccuracy = []
    id3TrainAccuracy = []

    c45TestAccuracy = []
    c45TrainAccuracy = []

    libTreeTestAccuracy = []
    libTreeTrainAccuracy = []

    for n in [100, 500, 1000, 2000, 3000, 5000, 7999]:
        data = fullData.head(n)
        xTrain, xTest, yTrain, yTest = train_test_split(data.drop('is_safe', axis=1), data['is_safe'], test_size=0.2)
        xAxis.append(xTrain.shape[0])

        print("Number of training samples: "+str(xTrain.shape[0]))
        print("Number of testing samples: "+str(xTest.shape[0]))

        id3 = id3New.id3Node()
        start = time.time()
        id3.recursiveGenerateTree(xTrain, yTrain, 0)
        stop = time.time()
        trainTime = stop - start
        trainAccuracy = id3.evaluate(xTrain, yTrain)
        testAccuracy = id3.evaluate(xTest, yTest)
        print("ID3:")
        print("Accuracy on train data: " + str(trainAccuracy))
        print("Accuracy on test data: " + str(testAccuracy))
        print("Training time: " + str(trainTime))
        id3Times.append(trainTime)
        id3TrainAccuracy.append(trainAccuracy)
        id3TestAccuracy.append(testAccuracy)

        print("C4.5:")
        c45Tree = c45.c45Node()
        start = time.time()
        c45Tree.recursiveGenerateTree(xTrain, yTrain, 0)
        stop = time.time()
        trainTime = stop - start
        trainAccuracy = c45Tree.evaluate(xTrain, yTrain)
        testAccuracy = c45Tree.evaluate(xTest, yTest)
        print("Accuracy on train data: " + str(trainAccuracy))
        print("Accuracy on test data: " + str(testAccuracy))
        print("Training time: " + str(trainTime))
        c45Times.append(trainTime)
        c45TrainAccuracy.append(trainAccuracy)
        c45TestAccuracy.append(testAccuracy)

        libTree = tree.DecisionTreeClassifier(criterion='entropy')
        start = time.time()
        libTree = libTree.fit(xTrain, yTrain)
        stop = time.time()
        trainTime = stop - start
        trainAccuracy = libTree.score(xTrain, yTrain)
        testAccuracy = libTree.score(xTest, yTest)
        print("Scikit learn DecisionTreeClassifier:")
        print("Accuracy on train data: " + str(trainAccuracy))
        print("Accuracy on test data: " + str(testAccuracy))
        print("Training time: " + str(trainTime))
        libTreeTimes.append(trainTime)
        libTreeTrainAccuracy.append(trainAccuracy)
        libTreeTestAccuracy.append(testAccuracy)
        tree.plot_tree(libTree)
        plt.savefig(str(n)+"decisionTree.png")

    plt.figure()
    plt.plot(xAxis, id3Times, label="ID3")
    plt.plot(xAxis, libTreeTimes, label="ScikitLearn")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Czas treningu[s]")
    plt.title("Czas treningu drzew w zależności od liczby próbek treningowych")
    plt.legend()
    plt.savefig("times.png")

    plt.figure()
    plt.plot(xAxis, id3Times, label="ID3")
    plt.plot(xAxis, c45Times, label="C4.5")
    plt.plot(xAxis, libTreeTimes, label="ScikitLearn")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Czas treningu[s]")
    plt.title("Czas treningu drzew w zależności od liczby próbek treningowych")
    plt.legend()
    plt.savefig("timesWithC45.png")

    plt.figure()
    plt.plot(xAxis, id3TrainAccuracy, '--b', label="ID3 Train")
    plt.plot(xAxis, id3TestAccuracy, label="ID3 Test")
    plt.plot(xAxis, c45TrainAccuracy, '-.r', label="C4.5 Train")
    plt.plot(xAxis, c45TestAccuracy, label="C4.5 Test")
    plt.plot(xAxis, libTreeTrainAccuracy, ':y', label="ScikitLearn Train")
    plt.plot(xAxis, libTreeTestAccuracy, label="ScikitLearn Test")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Trafność (0-1)")
    plt.title("Trafność przewidywania w zależnośći od liczby próbek treningowych")
    plt.legend()
    plt.savefig("accuracy.png")









