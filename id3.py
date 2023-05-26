import numpy as np
import random
from treeNode import TreeNode

class id3Node(TreeNode):

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
                    self.children[v] = id3Node(max_depth=self.max_depth, seed=self.seed)
                    self.children[v].recursiveGenerateTree(xTrain[index, :], yTrain[index], current_depth)
                else:
                    #
                    self.children[v] = id3Node(max_depth=1, seed=self.seed)
                    self.children[v].recursiveGenerateTree(xTrain, yTrain, current_depth=1)

    def splitAttribute(self, xTrain, yTrain):
        info_gain_max = -np.inf  # Info gain set to a minimun

        splitter = []
        best_attribute = None
        best_threshold = None

        for attribute in range(xTrain.shape[1]):

            aig = self.compute_info_gain(xTrain[:, attribute], yTrain)

            if aig > info_gain_max:
                splitter = xTrain[:, attribute]
                info_gain_max = aig
                best_attribute = attribute
                best_threshold = None
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

    def predict(self, sample):

        # If there is a decision in the node, return it
        if self.decision is not None:
            return self.decision

        # If not, it means that it is an internal node
        else:
            attr_val = sample[self.split_feat_name]

            # only for discrete values
            if self.children.__contains__(attr_val):
                child = self.children[attr_val]
            else:
                return None

            return child.predict(sample)

