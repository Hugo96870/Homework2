# Grupo 117 Aprendizagem HomeWork 2
# Bernardo Castico ist196845
# Hugo Rita ist196870

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

#Res = the 10-fold cross validation with our group number (117)
Res = KFold(n_splits=10, random_state=117, shuffle=True)

def getDataToMatrix(lines):
    realLines = []
    data = []
    toDelete = []
    for i in range(len(lines)):
        if i > 11:
            realLines += [lines[i]]
    for i in range(len(realLines)):
        for j in range(len(realLines[i])):
            if realLines[i][j] == "benign\n":
                realLines[i][j] = 1
            elif realLines[i][j] == "malignant\n":
                realLines[i][j] = 0
            elif realLines[i][j] == '?':
                toDelete += [i]
            else:
                realLines[i][j] = int(realLines[i][j])
    for i in range(len(realLines)):
        if i not in toDelete:
            data += [realLines[i]]
    return data

def splitData(list):
    a = []
    b = []
    for i in list:
        a.append(i[:-1])
        b.append(i[-1])
    return [a,b]


def main():
    res = []
    with open("HW2.txt") as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split(',')
        res.append(tmp)
    data = getDataToMatrix(res)

    for i in [1,3,5,9]:
        testData = []
        trainData = []
        accuracies1 = []
        accuracies2 = []

        for train, test in Res.split(data):
            for j in test:
                testData += [data[j]]
            for j in train:
                trainData += [data[j]]
            trainDataSplit = splitData(trainData)
            testDataSplit = splitData(testData)

            decision = SelectKBest(mutual_info_classif, k=i).fit(trainDataSplit[0], trainDataSplit[1])
            decisionTrainData = decision.transform(trainDataSplit[0])
            decisionTestData = decision.transform(testDataSplit[0])

            result1 = DecisonTreeClassifier(max_depth=None, criterion="entropy")
            result2 = DecisonTreeClassifier(max_depth=i, criterion="entropy")

            result1.fit(decisionTrainData, trainDataSplit[1])
            result2.fit(decisionTrainData, trainDataSplit[1])

            result1.Predict(testData)
            result2.Predict(testData)

            result1.Predict(trainData)
            result2.Predict(trainData)

main()