import numpy as np
from random import randrange
from sklearn.tree import DecisionTreeClassifier

#Returns a subset (with replacement) of the data samples
def getSubSamples(data, ratio=1):
    sample = list()
    n_sample = round(len(data) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index])
    return sample

#Returns the majority vote for a specific sample
def getForestMajorityVote(forest, sample, numFeatures):
    results = list()
    features = np.array(sample[:numFeatures]).reshape(1, -1)
    res = sample[-1]
    for tree in forest:
        results.append(tree.predict(features))

    results = [list(x)[0] for x in results]
    return max(set(results), key=results.count)

#Returns the accuracy of a single Decision Tree
def getForestAccuracy(forest, testSet, numFeatures):
    correct = 0
    for testSample in testSet:
        result = getForestMajorityVote(forest, testSample, numFeatures)
        if result == testSample[-1]:
            correct += 1
    return (correct / len(testSet) * 100);



def trainRandomForest(numTrees, dataset, trainingSet, testSet, numFeatures):
    Trees = []
    subSets = []
    accuracyAverage = 0;
    totFeatures = len(trainingSet[0])-1

    #Grow all the trees in the forest
    for k in range(0, numTrees):
        tree = DecisionTreeClassifier(max_features=numFeatures)
#         features = [x for x in range(len(dataset[0]) - 1)]
        subSet = getSubSamples(trainingSet, 1)
        features = [x[:totFeatures] for x in subSet]
        targets = [x[totFeatures:] for x in subSet]
#         subSets.append(subSet)
        tree.fit(features, targets)
        Trees.append(tree)

#     return outOfBagEstimate(dataset, subSets, Trees)
    return getForestAccuracy(Trees, testSet, totFeatures)

def outOfBagEstimate(dataset, subSets, trees):
    numSamples = len(dataset)
    numTrees = len(trees)
    correct = 0

    for sample in dataset:
        for i in range(numTrees):
            predictions = []
            if sample not in subSets[i]:
                predictions.append(trees[i].classify(sample))

        if len(predictions) == 0:
            numSamples -= 1
        else:
            result = max(set(predictions), key=predictions.count)
            if result == sample[-1]:
                correct += 1

    return (correct / numSamples * 100);
