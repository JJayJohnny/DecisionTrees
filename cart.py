from numbers import Number
import numpy as np
import pandas as pd
import random

from pandas.core.dtypes.common import is_string_dtype


class cartNode:
    def __init__(self, min_samples_split=2, max_depth=None, seed=2, verbose=False):
        # Sub nodes -- recursive, those elements of the same type (TreeNode)
        self.children = {}
        self.decision = None
        self.split_feat_name = None  # Splitting feature
        self.threshold = None  # Where to split the feature

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.seed = seed  # Seed for random numbers
        self.verbose = verbose  # True to print the splits

    def recursiveGenerateTree(self, xTrain, yTrain, current_depth):
        # print(xTrain.shape[0])
        if len(yTrain.unique()) == 1:
            self.decision = yTrain.unique()[0]
            return
        elif current_depth == self.max_depth:
            self.decision = self.getMajClass(yTrain)
            return
        else:
            bestAttribute, bestValue= self.find_best_split(xTrain, yTrain)
            self.children = {}
            self.split_feat_name = bestAttribute
            self.threshold = bestValue
            current_depth += 1

            if bestValue is None or bestAttribute is None:
                self.decision = self.getMajClass(yTrain)
                return

            (X_left, y_left), (X_right, y_right) = self.split_data(xTrain, yTrain, self.split_feat_name, self.threshold)

            if X_left is None or X_right is None:
                self.decision = self.getMajClass(yTrain)
                return

            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                self.decision = self.getMajClass(yTrain)
                return

            # if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            #     self.children[v] = cartNode(min_samples_split=self.min_samples_split, max_depth=1, seed=self.seed,verbose=self.verbose)
            #     self.children[v].recursiveGenerateTree(xTrain, yTrain, current_depth=1)
            # v is 'greater' or 'lesser'
            # index = splitter == v
            self.children['greater'] = cartNode(min_samples_split=self.min_samples_split, max_depth=self.max_depth, seed=self.seed, verbose=self.verbose)
            self.children['greater'].recursiveGenerateTree(X_right, y_right, current_depth)
            self.children['lesser'] = cartNode(min_samples_split=self.min_samples_split, max_depth=self.max_depth, seed=self.seed, verbose=self.verbose)
            self.children['lesser'].recursiveGenerateTree(X_left, y_left, current_depth)




    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        for possible_split in possible_splits:
            left_positive = 0
            left_negative = 0
            right_positive = 0
            right_negative = 0
            for i in range(0, possible_split):
                if y.iloc[i] == 1:
                    left_positive = left_positive + 1
                else:
                    left_negative = left_negative + 1
            for i in range(possible_split, len(y)):
                if y.iloc[i] == 1:
                    right_positive = right_positive + 1
                else:
                    right_negative = right_negative + 1

            if left_negative + left_positive > 0 and right_negative + right_positive > 0:
                gini_left = 1 - pow(left_positive / (left_positive + left_negative), 2) - pow(left_negative / (left_positive + left_negative), 2)
                gini_right = 1 - pow(right_positive / (right_positive + right_negative), 2) - pow(right_negative / (right_positive + right_negative), 2)
                gini_gain = 1 - (left_positive + left_negative) / (left_positive + left_negative + right_negative + right_positive) * gini_left - (right_positive + right_negative) / (left_positive + left_negative + right_negative + right_positive) * gini_right
                if gini_gain > best_gain:
                    best_gain = gini_gain
                    best_idx = possible_split

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        if idx is not None:
            left_mask = X[idx] < val
            return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])
        else:
            return (None, None), (None, None)

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data.iloc[idx] != data.iloc[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y):
        best_gain = -np.inf
        bestSplitIndex = None
        bestSplitArgument = None

        selected_features = X.keys()

        for d in selected_features:
            order = X.sort_values(by=d).index
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[d][order])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                bestSplitArgument = d
                bestSplitIndex = idx

        if bestSplitArgument is None:
            return None, None

        bestSplitValue = (X[bestSplitArgument].iloc[bestSplitIndex] + X[bestSplitArgument].iloc[bestSplitIndex+1])/2

        return bestSplitArgument, bestSplitValue

    def getMajClass(self, yTrain):

        freq = yTrain.value_counts().sort_values(ascending=False)

        # Select the name of the class (classes) that has the max number of records
        MajClass = freq.keys()[freq == freq.max()]
        # If there are two classes with equal number of records, select one randomly
        if len(MajClass) > 1:
            decision = MajClass[random.Random(self.seed).randint(0, len(MajClass) - 1)]
        # If there is only onle select that
        else:
            decision = MajClass[0]
        return decision

    def predict(self, sample):

        # If there is a decision in the node, return it
        if self.decision is not None:
            return self.decision

        # If not, it means that it is an internal node
        else:
            attr_val = sample[self.split_feat_name]

            if not isinstance(attr_val, Number):
                child = self.children[attr_val]
            else:
                #only for numeric values
                if attr_val > self.threshold:
                    child = self.children['greater']
                else:
                    child = self.children['lesser']
            return child.predict(sample)

    def evaluate(self, xTest, yTest):
        correct_preditct = 0
        wrong_preditct = 0
        for index in range(xTest.shape[0]):
            result = self.predict(xTest.iloc[index])
            if result == yTest.iloc[index]:
                correct_preditct += 1
            else:
                wrong_preditct += 1
        accuracy = correct_preditct / (correct_preditct + wrong_preditct)
        return accuracy
