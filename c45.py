from numbers import Number
import numpy as np
import pandas as pd
import random

from pandas.core.dtypes.common import is_string_dtype


class c45Node:
    def __init__(self, min_samples_split=2, max_depth=None, seed=2,
                 verbose=False):
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
        if len(yTrain.unique()) == 1:
            self.decision = yTrain.unique()[0]
        elif len(yTrain) < self.min_samples_split:
            self.decision = self.getMajClass(yTrain)
        elif current_depth == self.max_depth:
            self.decision = self.getMajClass(yTrain)
        else:
            best_attribute, best_threshold, splitter = self.splitAttribute(xTrain, yTrain)
            self.children = {}
            self.split_feat_name = best_attribute
            self.threshold = best_threshold
            current_depth += 1

            for v in splitter.unique():
                # v is 'greater' or 'lesser'
                index = splitter == v
                if len(xTrain[index]) > 0:
                    self.children[v] = c45Node(min_samples_split=self.min_samples_split, max_depth=self.max_depth, seed=self.seed, verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(xTrain[index], yTrain[index], current_depth)
                else:
                    #
                    self.children[v] = c45Node(min_samples_split=self.min_samples_split, max_depth=1, seed=self.seed, verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(xTrain, yTrain, current_depth=1)

    def splitAttribute(self, xTrain, yTrain):
        info_gain_max = -1 * float("inf")  # Info gain set to a minimun

        splitter = pd.Series(dtype='str')
        best_attribute = None
        best_threshold = None

        for attribute in xTrain.keys():
            if is_string_dtype(xTrain[attribute]):

                aig = self.compute_info_gain(xTrain[attribute], yTrain)

                if aig > info_gain_max:
                    splitter = xTrain[attribute]
                    info_gain_max = aig
                    best_attribute = attribute
                    best_threshold = None
            else:
                # only for continuous attributes
                sorted_index = xTrain[attribute].sort_values(ascending=True).index
                sorted_sample_data = xTrain[attribute][sorted_index]
                sorted_sample_target = yTrain[sorted_index]

                for j in range(0, len(sorted_sample_data) - 1):

                    classification = pd.Series(dtype='str')

                    if sorted_sample_data.iloc[j] != sorted_sample_data.iloc[j + 1]:
                        threshold = (sorted_sample_data.iloc[j] + sorted_sample_data.iloc[j + 1]) / 2
                        # assign all the values of the same attribute as either grater or lesser than threshold
                        classification = xTrain[attribute] > threshold
                        classification[classification] = 'greater'
                        classification[classification == False] = 'lesser'

                        aig = self.compute_info_gain(classification, yTrain)
                        # save the best threshold split
                        if aig >= info_gain_max:
                            splitter = classification
                            info_gain_max = aig
                            best_attribute = attribute
                            best_threshold = threshold
        return best_attribute, best_threshold, splitter

    def compute_entropy(self, sampleSplit):
        #If there is only only one class, the entropy is 0
        if len(sampleSplit) < 2:
            return 0
        else:
            freq = np.array(sampleSplit.value_counts(normalize=True))
            return -(freq * np.log2(freq + 1e-6)).sum()

    def compute_info_gain(self, sampleAttribute, sample_target):

        values = sampleAttribute.value_counts(normalize=True)
        split_ent = 0

        # Iterate for each class of the sample attribute
        for v, fr in values.items():

            index = sampleAttribute == v
            sub_ent = self.compute_entropy(sample_target[index])

            # Weighted sum of the entropies
            split_ent += fr * sub_ent

        # Compute the entropy without any split
        ent = self.compute_entropy(sample_target)
        return ent - split_ent

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
                if self.verbose:
                    print("Testing ", self.split_feat_name, "->", attr_val)
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
