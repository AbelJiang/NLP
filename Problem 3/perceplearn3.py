import json
import operator
import sys
import random
from string import punctuation
from random import shuffle

# corpFile = open(sys.argv[1], 'r')
corpFile = open('/Users/abel/Desktop/NLP/Problem 3/corpus/train-labeled.txt', 'r')
corp = [i.strip().split() for i in corpFile.readlines()]
corpFile.close()

negSuffix = ('n\'t', 'not', 'no', 'never', "n't")
vocabStat = {}
featWeightV = {}
featWeightA = {}
vocabFeat = set()

for review in corp:
    for i in range(3, len(review)):
        review[i] = review[i].lower().strip().strip(punctuation)
        if review[i].endswith(tuple(negSuffix)) and i + 1 < len(review):
            review[i + 1] = 'n_' + review[i + 1]
        if review[i].startswith('$'):
            review[i] = '$'

        vocabStat[review[i]] = vocabStat.get(review[i], 0) + 1

# vocabFeat = {k for (k, v) in sorted(vocabStat.items(), key=operator.itemgetter(1), reverse=True)[5:]}
vocabFeat = {i for i in vocabStat}

vocabFeat.add('*BIAS')
vocabFeat.add('*RLEN')

reviewStat = {}
for review in corp:
    review[-1] = len(review)
    reviewStat[review[0]] = {}
    for i in range(3, len(review) - 1):
        if review[i] in reviewStat[review[0]]:
            reviewStat[review[0]][review[i]] = reviewStat[review[0]][review[i]] + 1
        else:
            reviewStat[review[0]][review[i]] = 1

for word in vocabFeat:
    featWeightV[word] = {}
    featWeightA[word] = {}
    featWeightV[word]['TF'] = 0
    featWeightV[word]['PN'] = 0
    featWeightA[word]['TF'] = 0
    featWeightA[word]['PN'] = 0
MaxIterTF = 30
MaxIterPN = 25
thresh = 1.0

# vanilla perceptron for T/F
b = 0
for iter in range(0, MaxIterTF):
    random.Random(1).shuffle(corp)
    for review in corp:
        rid = review[0]
        a = 0
        if review[1] == 'True':
            y = thresh
        else:
            y = -thresh
        for word in reviewStat[rid]:
            if word in vocabFeat:
                a = a + featWeightV[word]['TF'] * reviewStat[rid][word]
        a = a + b
        if y * a <= 0:
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    featWeightV[word]['TF'] = featWeightV[word]['TF'] + y * reviewStat[rid][word]
            b = b + y
featWeightV['*BIAS']['TF'] = b

# vanilla perceptron for P/N
b = 0
for iter in range(0, MaxIterPN):
    random.Random(3).shuffle(corp)
    for review in corp:
        rid = review[0]
        a = 0
        if review[2] == 'Pos':
            y = thresh
        else:
            y = -thresh
        for word in reviewStat[rid]:
            if word in vocabFeat:
                a = a + featWeightV[word]['PN'] * reviewStat[rid][word]
        a = a + b
        if y * a <= 0:
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    featWeightV[word]['PN'] = featWeightV[word]['PN'] + y * reviewStat[rid][word]
            b = b + y
featWeightV['*BIAS']['PN'] = b

# averaged perceptron for T/F
b = 0
beta = 0
c = 1
u = {}
for word in vocabFeat:
    u[word] = 0

for iter in range(0, MaxIterTF):
    random.Random(5).shuffle(corp)
    for review in corp:
        rid = review[0]
        a = 0
        if review[1] == 'True':
            y = thresh
        else:
            y = -thresh
        for word in reviewStat[rid]:
            if word in vocabFeat:
                a = a + featWeightA[word]['TF'] * reviewStat[rid][word]
        a = a + b
        if y * a <= 0:
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    featWeightA[word]['TF'] = featWeightA[word]['TF'] + y * reviewStat[rid][word]
            b = b + y
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    u[word] = u[word] + y * reviewStat[rid][word] * c
            beta = beta + y * c
        c = c + 1

for word in vocabFeat:
    featWeightA[word]['TF'] = featWeightA[word]['TF'] - u[word] / c
featWeightA['*BIAS']['TF'] = b - beta / c

# averaged perceptron for P/N
b = 0
beta = 0
c = 1
u = {}
for word in vocabFeat:
    u[word] = 0

for iter in range(0, MaxIterPN):
    random.Random(7).shuffle(corp)
    for review in corp:
        rid = review[0]
        a = 0
        if review[2] == 'Pos':
            y = thresh
        else:
            y = -thresh
        for word in reviewStat[rid]:
            if word in vocabFeat:
                a = a + featWeightA[word]['PN'] * reviewStat[rid][word]
        a = a + b
        if y * a <= 0:
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    featWeightA[word]['PN'] = featWeightA[word]['PN'] + y * reviewStat[rid][word]
            b = b + y
            for word in reviewStat[rid]:
                if word in vocabFeat:
                    u[word] = u[word] + y * reviewStat[rid][word] * c
            beta = beta + y * c
        c = c + 1

for word in vocabFeat:
    featWeightA[word]['PN'] = featWeightA[word]['PN'] - u[word] / c
featWeightA['*BIAS']['PN'] = b - beta / c

ModelFile = open('/Users/abel/Desktop/NLP/Problem 3/vanillamodel.txt', 'w')
ModelFile.write(json.dumps(featWeightV))
ModelFile.close()

ModelFile = open('/Users/abel/Desktop/NLP/Problem 3/averagedmodel.txt', 'w')
ModelFile.write(json.dumps(featWeightA))
ModelFile.close()

