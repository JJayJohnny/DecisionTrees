import pandas as pd
import id3
from sklearn.model_selection import train_test_split

dataLoc='data/waterQuality1.csv'

def GetData(location):
    data = pd.read_csv(location)
    return data

if __name__ == '__main__':
    print("Main")
    data = GetData(dataLoc)
    xTrain, xTest, yTrain, yTest = train_test_split(data.drop('is_safe', axis=1), data['is_safe'], test_size=0.2)
    tree = id3.id3(xTrain, yTrain)
    print(tree)
    print("Accuracy on train data: "+str(id3.evaluate(tree, xTrain, yTrain)))
    print("Accuracy on test data: "+str(id3.evaluate(tree, xTest, yTest)))




