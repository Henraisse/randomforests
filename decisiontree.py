import math
from math import log
from math import inf
import numpy
from numpy import random




#Returns a subset (with no replacement) of the features
def getSubFeatures(features):
    size = int(math.sqrt(len(features)))
    sub = random.choice(features, size, False)
    return sub

class Dtree:
    def __init__(self, maxdepth=1000):
        self.root = None
        self.maxdepth = maxdepth

    def train(self, dataset, features):
        self.root = self.buildTree(dataset, features, 1)

    def classify(self, sample):
        node = self.root
        while node.label == None:
            feature = node.feature
            value = node.value
            if sample[feature] > value:
                node = node.rightChild
            else:
                node = node.leftChild

        return node.label

    def buildTree(self, dataset, features, depth):
        if isHomogeneous(dataset):
            return Node(label=dataset[0][-1])
        if depth >= self.maxdepth:
            majorityLabel = getMajorityLabel(dataset)
            return Node(label=majorityLabel)

        left, right, bestFeature, bestValue = bestSplit(dataset, features)

        features.remove(bestFeature)

        node = Node(feature=bestFeature, value=bestValue)

        if len(left) == 0 or len(right) == 0:
            majorityLabel = getMajorityLabel(left + right)
            return Node(label=majorityLabel)

        node.leftChild = self.buildTree(left, features, depth + 1)
        node.rightChild = self.buildTree(right, features, depth + 1)

        return node


class Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature
        self.value = value
        self.label = label
        self.leftChild = None
        self.rightChild = None


def extractUniqueLabels(dataset):
    "Returns a list of all unique labels in the dataset"
    labels = set()
    for data in dataset:
        labels.add(data[-1])
    labels = list(labels)
    return labels




def bestSplit(dataset, features):
    """Calculates the best split to perform on the dataset.
       Returns two datasets,the feature and the feature's value that was used to perfrom the split."""
    n = len(dataset)
    bestFeature = None
    bestValue = inf
    bestGini = inf
    bestGroup = None
    labels = extractUniqueLabels(dataset)

    for feature in getSubFeatures(features):
        for data in dataset:
            groups = split(dataset, feature, data[feature])
            giniValue = giniIndex(groups, labels)
            if giniValue < bestGini:
                bestFeature = feature
                bestValue = data[feature]
                bestGini = giniValue
                bestGroup = groups

    return bestGroup[0], bestGroup[1], bestFeature, bestValue


def split(dataset, feature, value):
    "Split the dataset into two different subsets on a given feature and a value"
    left = list()
    right = list()
    for data in dataset:
        if data[feature] > value:
            right.append(data)
        else:
            left.append(data)
    return left, right


def isHomogeneous(dataset):
    "Checks to see if all the data in the dataset belongs to the same class"
    for i in range(len(dataset) - 1):
        if dataset[i][-1] != dataset[i + 1][-1]:
            return False

    return True


def getMajorityLabel(dataset):
    "Returns the label with the greatest occurrence in the dataset"
    lables = dict()
    for data in dataset:
        label = data[-1]
        if label in lables:
            lables[label] += 1
        else:
            lables[label] = 1
    return max(lables, key=lambda label: lables[label])


def giniIndex(datasets, labels):
    "Calculate the gini index for a split"
    gini = 0.0
    for label in labels:
        for dataset in datasets:
            n = len(dataset)
            if n != 0:
                count = 0
                for data in dataset:
                    if data[-1] == label:
                        count += 1
                gini += float(count) / float(n) * (1.0 - float(count) / float(n))
    return gini