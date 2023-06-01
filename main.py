import pandas as pd
import c45
import id3
from sklearn.model_selection import train_test_split
from sklearn import tree
import time
import matplotlib.pyplot as plt
import cart

dataLoc='data/waterQuality1.csv'
imgLoc='img/'

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
    cartTimes = []

    id3TestAccuracy = []
    id3TrainAccuracy = []

    c45TestAccuracy = []
    c45TrainAccuracy = []

    libTreeTestAccuracy = []
    libTreeTrainAccuracy = []

    cartTreeTestAccuracy = []
    cartTreeTrainAccuracy = []

    for n in [100, 500, 1000, 2000, 3000, 5000, 7999]:
        data = fullData.head(n)
        xTrain, xTest, yTrain, yTest = train_test_split(data.drop('is_safe', axis=1), data['is_safe'], test_size=0.2)
        xTrain = xTrain.to_numpy()
        xTest = xTest.to_numpy()
        yTrain = yTrain.to_numpy()
        yTest = yTest.to_numpy()
        xAxis.append(xTrain.shape[0])

        print("\nNumber of training samples: "+str(xTrain.shape[0]))
        print("Number of testing samples: "+str(xTest.shape[0]))

        id3Tree = id3.id3Node()
        start = time.time()
        id3Tree.recursiveGenerateTree(xTrain, yTrain, 0)
        stop = time.time()
        trainTime = stop - start
        trainAccuracy = id3Tree.evaluate(xTrain, yTrain)
        testAccuracy = id3Tree.evaluate(xTest, yTest)
        print("ID3:")
        print("Accuracy on train data: " + str(trainAccuracy))
        print("Accuracy on test data: " + str(testAccuracy))
        print("Training time: " + str(trainTime))
        print("Tree width: "+str(id3Tree.getMaxWidth(0)))
        print("Tree depth: "+str(id3Tree.getMaxDepth(0)))
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
        print("Tree width: " + str(c45Tree.getMaxWidth(0)))
        print("Tree depth: " + str(c45Tree.getMaxDepth(0)))
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
        plt.savefig(imgLoc+str(n)+"decisionTree.png")

        cartTree = cart.cartNode()
        start = time.time()
        cartTree.recursiveGenerateTree(xTrain, yTrain, 0)
        stop = time.time()
        trainTime = stop - start
        trainAccuracy = cartTree.evaluate(xTrain, yTrain)
        testAccuracy = cartTree.evaluate(xTest, yTest)
        print("CART:")
        print("Accuracy on train data: "+str(trainAccuracy))
        print("Accuracy on test data: "+str(testAccuracy))
        print("Training time: "+str(trainTime))
        print("Tree width: " + str(cartTree.getMaxWidth(0)))
        print("Tree depth: " + str(cartTree.getMaxDepth(0)))
        cartTimes.append(trainTime)
        cartTreeTrainAccuracy.append(trainAccuracy)
        cartTreeTestAccuracy.append(testAccuracy)

    plt.figure()
    plt.plot(xAxis, id3Times, label="ID3")
    plt.plot(xAxis, libTreeTimes, label="ScikitLearn")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Czas treningu[s]")
    plt.title("Czas treningu drzew w zależności od liczby próbek treningowych")
    plt.legend()
    plt.savefig(imgLoc+"times.png")

    # plt.figure()
    # plt.plot(xAxis, id3Times, label="ID3")
    plt.plot(xAxis, c45Times, label="C4.5")
    # plt.plot(xAxis, libTreeTimes, label="ScikitLearn")
    # plt.xlabel("Liczba próbek treningowych")
    # plt.ylabel("Czas treningu[s]")
    # plt.title("Czas treningu drzew w zależności od liczby próbek treningowych")
    plt.legend()
    plt.savefig(imgLoc+"timesWithC45.png")

    plt.plot(xAxis, cartTimes, label='CART')
    plt.legend()
    plt.savefig(imgLoc+"timesWithC45andCART.png")

    plt.figure()
    plt.plot(xAxis, id3TrainAccuracy, '--b', label="ID3 Train")
    plt.plot(xAxis, id3TestAccuracy, label="ID3 Test")
    plt.plot(xAxis, libTreeTrainAccuracy, '-.y', label="ScikitLearn Train")
    plt.plot(xAxis, libTreeTestAccuracy, label="ScikitLearn Test")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Trafność (0-1)")
    plt.title("Trafność przewidywania w zależnośći od liczby próbek treningowych")
    plt.legend()
    plt.savefig(imgLoc+"accuracy.png")

    plt.plot(xAxis, c45TrainAccuracy, ':r', label="C4.5 Train")
    plt.plot(xAxis, c45TestAccuracy, label="C4.5 Test")
    plt.legend()
    plt.savefig(imgLoc+"accuracyWithC45.png")

    plt.plot(xAxis, cartTreeTrainAccuracy, ':g', label="CART Train")
    plt.plot(xAxis, cartTreeTestAccuracy, label="CART Test")
    plt.legend()
    plt.savefig(imgLoc+"accuracyWitchC45andCART.png")











