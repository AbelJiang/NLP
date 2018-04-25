import json
import operator
import sys
from string import punctuation
from random import shuffle

corpFile = open(sys.argv[1], 'r')
#corpFile = open('/Users/abel/Desktop/NLP/Problem 3/corpus/train-labeled.txt', 'r')
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
        vocabStat[review[i]] = vocabStat.get(review[i], 0) + 1

#vocabFeat = {k for (k, v) in sorted(vocabStat.items(), key=operator.itemgetter(1), reverse=True)}
vocabFeat={i for i in vocabStat}
vocabFeat.add('*BIAS')
vocabFeat.add('*RLEN')

for review in corp:
    vocab=set()
    rmv=[]
    for i in range(0,len(review)-1):
        if review[i] in vocab:
            rmv.append(review[i])
        else:
            vocab.add(review[i])
    for item in rmv:
        review.remove(item)


for word in vocabFeat:
    featWeightV[word] = {}
    featWeightA[word] = {}
    featWeightV[word]['TF'] = 0
    featWeightV[word]['PN'] = 0
    featWeightA[word]['TF'] = 0
    featWeightA[word]['PN'] = 0
MaxIterTF=35
MaxIterPN=30
thresh=10

# vanilla perceptron for T/F
b = 0
for iter in range(0, MaxIterTF):
    shuffle(corp)
    for review in corp:
        a = 0
        if review[1] == 'True':
            y=thresh
        else:
            y=-thresh
        for i in range(3, len(review)):
            if review[i] in vocabFeat:
                a = a + featWeightV[review[i]]['TF']
        a = a + b
        if y * a <= 0:
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    featWeightV[review[i]]['TF'] = featWeightV[review[i]]['TF'] + y * 1
            b = b + y
featWeightV['*BIAS']['TF'] = b

# vanilla perceptron for P/N
b = 0
for iter in range(0, MaxIterPN):
    shuffle(corp)
    for review in corp:
        a = 0
        if review[2] == 'Pos':
            y=thresh
        else:
            y=-thresh
        for i in range(3, len(review)):
            if review[i] in vocabFeat:
                a = a + featWeightV[review[i]]['PN']
        a = a + b
        if y * a <= 0:
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    featWeightV[review[i]]['PN'] = featWeightV[review[i]]['PN'] + y * 1
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
    shuffle(corp)
    for review in corp:
        a = 0
        if review[1] == 'True':
            y=thresh
        else:
            y=-thresh
        for i in range(3, len(review)):
            if review[i] in vocabFeat:
                a = a + featWeightA[review[i]]['TF']
        a = a + b
        if y * a <= 0:
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    featWeightA[review[i]]['TF'] = featWeightA[review[i]]['TF'] + y * 1
            b = b + y
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    u[review[i]] = u[review[i]] + y * 1 * c
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
    shuffle(corp)
    for review in corp:
        a = 0
        if review[2] == 'Pos':
            y=thresh
        else:
            y=-thresh
        for i in range(3, len(review)):
            if review[i] in vocabFeat:
                a = a + featWeightA[review[i]]['PN']
        a = a + b
        if y * a <= 0:
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    featWeightA[review[i]]['PN'] = featWeightA[review[i]]['PN'] + y * 1
            b = b + y
            for i in range(3, len(review)):
                if review[i] in vocabFeat:
                    u[review[i]] = u[review[i]] + y * 1 * c
            beta = beta + y * c
        c = c + 1

for word in vocabFeat:
    featWeightA[word]['PN'] = featWeightA[word]['PN'] - u[word] / c
featWeightA['*BIAS']['PN'] = b - beta / c

ModelFile=open('./vanillamodel.txt','w')
ModelFile.write(json.dumps(featWeightV))
ModelFile.close()

ModelFile=open('./averagedmodel.txt','w')
ModelFile.write(json.dumps(featWeightA))
ModelFile.close()


