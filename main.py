import pandas as pd
import id3
import c45
import id3New
from sklearn.model_selection import train_test_split
import time

dataLoc='data/waterQuality1.csv'

def GetData(location):
    data = pd.read_csv(location)
    return data

if __name__ == '__main__':
    print("Main")
    data = GetData(dataLoc)
    data = data.head(1000)
    xTrain, xTest, yTrain, yTest = train_test_split(data.drop('is_safe', axis=1), data['is_safe'], test_size=0.2)
    start = time.time()
    tree = id3.id3(xTrain, yTrain)
    stop = time.time()
    print(tree)
    print("ID3:")
    print("Accuracy on train data: "+str(id3.evaluate(tree, xTrain, yTrain)))
    print("Accuracy on test data: "+str(id3.evaluate(tree, xTest, yTest)))
    print("Training time: "+str(stop-start))

    print("\nC4.5:")
    c45 = c45.c45Node(max_depth=3)
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





