from numbers import Number
import numpy as np
import random


class c45Node:
    def __init__(self, min_samples_split=2, max_depth=None, seed=2):
        self.children = {}
        self.decision = None
        self.split_feat_name = None  # Splitting feature
        self.threshold = None  # Where to split the feature

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.seed = seed  # Seed for random numbers

    def recursiveGenerateTree(self, xTrain, yTrain, current_depth):
        remainingClasses = np.unique(yTrain)
        if len(remainingClasses) == 1:
            self.decision = remainingClasses[0]
            return
        elif current_depth == self.max_depth:
            self.decision = self.getMajClass(yTrain)
            return
        else:
            best_attribute, best_threshold, splitter = self.splitAttribute(xTrain, yTrain)
            self.children = {}
            self.split_feat_name = best_attribute
            self.threshold = best_threshold
            current_depth += 1

            for v in np.unique(splitter):
                # v is 'greater' or 'lesser'
                index = splitter == v
                if len(xTrain[index, best_attribute]) > 0:
                    self.children[v] = c45Node(min_samples_split=self.min_samples_split, max_depth=self.max_depth, seed=self.seed)
                    self.children[v].recursiveGenerateTree(xTrain[index, :], yTrain[index], current_depth)
                else:
                    #
                    self.children[v] = c45Node(min_samples_split=self.min_samples_split, max_depth=1, seed=self.seed)
                    self.children[v].recursiveGenerateTree(xTrain, yTrain, current_depth=1)

    def splitAttribute(self, xTrain, yTrain):
        info_gain_max = -np.inf  # Info gain set to a minimun

        splitter = []
        best_attribute = None
        best_threshold = None

        for attribute in range(xTrain.shape[1]):
            if isinstance(xTrain[0, attribute], str):

                aig = self.compute_info_gain(xTrain[:, attribute], yTrain)

                if aig > info_gain_max:
                    splitter = xTrain[:, attribute]
                    info_gain_max = aig
                    best_attribute = attribute
                    best_threshold = None
            else:
                # only for continuous attributes
                sorted_index = np.argsort(xTrain[:, attribute])
                sorted_sample_data = xTrain[sorted_index, attribute]

                for j in range(0, len(sorted_sample_data) - 1):

                    if sorted_sample_data[j] != sorted_sample_data[j + 1]:
                        threshold = (sorted_sample_data[j] + sorted_sample_data[j + 1]) / 2
                        # assign all the values of the same attribute as either grater or lesser than threshold
                        classification = np.where(xTrain[:, attribute] > threshold, 'greater', 'lesser')

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
            values, freq = np.unique(sampleSplit, return_counts=True)
            freq = freq / len(sampleSplit)
            return -(freq * np.log2(freq + 1e-6)).sum()

    def compute_info_gain(self, sampleAttribute, sample_target):

        values, counts = np.unique(sampleAttribute, return_counts=True)
        counts = counts / len(sampleAttribute)
        split_ent = 0

        # Iterate for each class of the sample attribute
        for i in range(len(values)):

            index = sampleAttribute == values[i]
            sub_ent = self.compute_entropy(sample_target[index])

            # Weighted sum of the entropies
            split_ent += counts[i] * sub_ent
        # Compute the entropy without any split
        ent = self.compute_entropy(sample_target)
        return ent - split_ent

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
            result = self.predict(xTest[index])
            if result == yTest[index]:
                correct_preditct += 1
            else:
                wrong_preditct += 1
        accuracy = correct_preditct / (correct_preditct + wrong_preditct)
        return accuracy
