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
    data = GetData(dataLoc)
    data = data.head(500)
    xTrain, xTest, yTrain, yTest = train_test_split(data.drop('is_safe', axis=1), data['is_safe'], test_size=0.2)

    print("Number of training samples: "+str(xTrain.shape[0]))
    print("Number of testing samples: "+str(xTest.shape[0]))

    print("C4.5:")
    c45 = c45.c45Node()
    start = time.time()
    c45.recursiveGenerateTree(xTrain, yTrain, 0)
    stop = time.time()
    print("Accuracy on train data: " + str(c45.evaluate(xTrain, yTrain)))
    print("Accuracy on test data: " + str(c45.evaluate(xTest, yTest)))
    print("Training time: " + str(stop - start))

    id3 = id3New.id3Node()
    start = time.time()
    id3.recursiveGenerateTree(xTrain, yTrain, 0)
    stop = time.time()
    print("ID3:")
    print("Accuracy on train data: " + str(id3.evaluate(xTrain, yTrain)))
    print("Accuracy on test data: " + str(id3.evaluate(xTest, yTest)))
    print("Training time: " + str(stop - start))

    libTree = tree.DecisionTreeClassifier()
    start = time.time()
    libTree = libTree.fit(xTrain, yTrain)
    stop = time.time()
    print("Scikit learn DecisionTreeClassifier:")
    print("Accuracy on train data: " + str(libTree.score(xTrain, yTrain)))
    print("Accuracy on test data: " + str(libTree.score(xTest, yTest)))
    print("Training time: " + str(stop - start))
    tree.plot_tree(libTree)
    plt.show()







