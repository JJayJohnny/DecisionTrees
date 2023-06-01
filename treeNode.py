import numpy as np
from numbers import Number
import random

class TreeNode:
    def __init__(self, max_depth=None, seed=2):
        self.children = {}
        self.decision = None
        self.split_feat_name = None  # Splitting feature
        self.threshold = None  # Where to split the feature

        self.max_depth = max_depth
        self.seed = seed  # Seed for random numbers

    def getMajClass(self, yTrain):

            values, counts = np.unique(yTrain, return_counts=True)

            # Select the name of the class (classes) that has the max number of records
            majClass = values[counts == counts.max()]
            # If there are two classes with equal number of records, select one randomly
            if len(majClass) > 1:
                decision = majClass[random.Random(self.seed).randint(0, len(majClass) - 1)]
            # If there is only onle select that
            else:
                decision = majClass[0]
            return decision


    def evaluate(self, xTest, yTest):
        correct_preditct = 0
        wrong_preditct = 0
        for index in range(xTest.shape[0]):
            result = self.predict(xTest[index])
            if result == yTest[index]:
                correct_preditct += 1
            else:
                wrong_preditct += 1
        accuracy = correct_preditct / (correct_preditct + wrong_preditct)
        return accuracy

    def getMaxWidth(self, maxWidth):
        width = len(self.children.items())
        if width > maxWidth:
            maxWidth = width
        for key, value in self.children.items():
            width = value.getMaxWidth(maxWidth)
            if width > maxWidth:
                maxWidth = width
        return maxWidth

    def getMaxDepth(self, depth):
        depth += 1
        d = []
        for key, value in self.children.items():
            d.append(value.getMaxDepth(depth))
        if len(d) == 0:
            return depth
        else:
            return max(d)


