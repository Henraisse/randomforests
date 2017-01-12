from randomforest import trainRandomForest
from random import shuffle
from csv import reader
from math import log
import re

#The number of iterations over a specific test to average on
avgIters = 100
forestSizes = [1, 10, 25, 50, 100];

# Load a CSV file
def loadFile(filename):
    dataset = list()
    f = open(filename, 'r')
    lines = f.readlines()

    pattern = '[,\s]\s*'
    for line in lines:
    	row = re.split(pattern,line.strip())
    	dataset.append(row)
    return dataset

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def partition(dataset, fraction):
    breakPoint = int(len(dataset) * fraction)
    shuffle(dataset)
    return dataset[:breakPoint], dataset[breakPoint:]

def getDatasets():
    datasets = []

    datasets.append(["Sonar", loadFile("datasets/formatted/sonar.csv")])
    datasets.append(["vowel", loadFile("datasets/formatted/vowel.csv")])
    datasets.append(["Glass", loadFile("datasets/formatted/glass.csv")])
    datasets.append(["Ionosphere", loadFile("datasets/formatted/ionosphere.csv")])
    datasets.append(["Sat Images", loadFile("datasets/formatted/sat-images.csv")])

    #datasets.append(["Breast Cancer", loadFile("datasets/formatted/breast-cancer.csv")])# - KRASHAR
    #datasets.append(["Votes", loadFile("datasets/formatted/votes.csv")])# - KRASHAR
    #datasets.append(["vowel", loadFile("datasets/formatted/german-credit.csv")])# - KRASHAR
    #datasets.append(["vowel", loadFile("datasets/formatted/ecoli.csv")])# - KRASHAR
    #datasets.append(["vowel", loadFile("datasets/formatted/letters.csv")])# - KRASHAR

    return datasets



#Iterates over every dataset, and tries every possible combination of datasets, variables, and such
def runTests():
    counter = 0
    datasets = getDatasets()
    output = []
    print("Running total of (", (len(datasets)), "×", len(forestSizes),"×",3, ") times until finished.")

    # Run once for every dataset
    for dataset in datasets:
        print("")

        dataName = dataset[0]
        dataSamples = dataset[1]

        totalNumFeatures = len(dataSamples[0]) - 1
        features = [x[:totalNumFeatures] for x in dataSamples]
        targets = [x[totalNumFeatures:] for x in dataSamples]

        numFeatures1 = 1;                                       numFeaturesTag1 = "1"
        numFeatureslog = round(log(totalNumFeatures + 1, 2));   numFeaturesTaglog = "log2(M+1)"
        numFeaturesAll = totalNumFeatures;                      numFeaturesTagAll = "ALL"
        numTrees = 100
        for k in forestSizes:
            runTest(output, dataSamples, dataName, numFeatures1, numFeaturesTag1, k)
            print(counter, end=",", flush=True); counter += 1
            runTest(output, dataSamples, dataName, numFeatureslog, numFeaturesTaglog, k)
            print(counter, end=",", flush=True);counter += 1
            runTest(output, dataSamples, dataName, numFeaturesAll, numFeaturesTagAll, k)
            print(counter, end=",", flush=True);counter += 1

    printResults(output)



#Fills output list with test result data
def runTest(output, dataSamples, dataName, numFeatures, numFeaturesTag, numTrees):
    trainingSet, testSet = partition(dataSamples, 0.9)
    mean_accuracy = 0
    for k in range(0,avgIters):
        accuracy = trainRandomForest(numTrees, dataSamples, trainingSet, testSet, numFeatures)
        mean_accuracy += accuracy/avgIters
    output.append([dataName, numFeaturesTag, numTrees, mean_accuracy])


#Prints the output list in a nice manner
def printResults(output):
    print("------------------------------------------------------------------------------")
    print("All tests is an average on: ", avgIters, " iterations.")
    print("------------------------------------------------------------------------------")
    print('%-20s%-20s%-20s%-20s' %("DATASET", "FEATURE COUNT", "TREES", "MEAN ACCURACY"))
    print("------------------------------------------------------------------------------")
    for out in output:
        print('%-20s%-20s%-20i%-20f' % (out[0], out[1], out[2], out[3]))









def main():
    runTests()






if __name__ == '__main__':
        main()
