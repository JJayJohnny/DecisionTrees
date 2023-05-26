from numbers import Number
import numpy as np
import random
from treeNode import TreeNode

class cartNode(TreeNode):

    def recursiveGenerateTree(self, xTrain, yTrain, current_depth):
        remainingClasses = np.unique(yTrain)
        if len(remainingClasses) == 1:
            self.decision = remainingClasses[0]
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

            # v is 'greater' or 'lesser'
            self.children['greater'] = cartNode(max_depth=self.max_depth, seed=self.seed)
            self.children['greater'].recursiveGenerateTree(X_right, y_right, current_depth)
            self.children['lesser'] = cartNode(max_depth=self.max_depth, seed=self.seed)
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
                if y[i] == 1:
                    left_positive = left_positive + 1
                else:
                    left_negative = left_negative + 1
            for i in range(possible_split, len(y)):
                if y[i] == 1:
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
            left_mask = X[:, idx] < val
            return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])
        else:
            return (None, None), (None, None)

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y):
        best_gain = -np.inf
        bestSplitIndex = None
        bestSplitArgument = None

        selected_features = range(X.shape[1])

        for d in selected_features:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                bestSplitArgument = d
                bestSplitIndex = idx

        if bestSplitArgument is None:
            return None, None

        bestSplitValue = (X[bestSplitIndex, bestSplitArgument] + X[bestSplitIndex+1, bestSplitArgument])/2

        return bestSplitArgument, bestSplitValue

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

