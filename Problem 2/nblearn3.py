import json
import json
import operator
import sys
import math
from string import punctuation
from collections import defaultdict

#corpFile = open(sys.argv[1], 'r')
corpFile = open('./corpus/train-labeled.txt', 'r')
corp = [i.strip().split(' ') for i in corpFile.readlines()]
corpFile.close()

classSet = {'True', 'Fake', 'Neg', 'Pos'}
revCount = len(corp)
revCountByClass = {'True': 0, 'Fake': 0, 'Neg': 0, 'Pos': 0}
revPrior = {'True': 0, 'Fake': 0, 'Neg': 0, 'Pos': 0}
vocabCountByClass = {'True': 0, 'Fake': 0, 'Neg': 0, 'Pos': 0}
VocabMat = defaultdict(dict)

negSuffix = ('n\'t', 'not', 'no', 'never', "n't")

vocabStat={}
stopWords=set()

for review in corp:
    for i in range(3, len(review)):
        vocabStat[review[i]]=vocabStat.get(review[i],0)+1
stopWords={k for (k,v) in sorted(vocabStat.items(), key=operator.itemgetter(1),reverse=True)[:100]}
for word in vocabStat:
    if vocabStat[word]<=4:
        stopWords.add(word)
for review in corp:
    classA = review[1]
    revCountByClass[classA] = revCountByClass[classA] + 1
    vocab = set()
    for i in range(3, len(review)):
        review[i] = review[i].lower()
        vocab.add(review[i].strip(punctuation))
    for word in vocab:
        if word not in stopWords:
            VocabMat[word][classA] = VocabMat[word].get(classA, 0) + 1

for review in corp:
    negation = 0
    classB = review[2]
    revCountByClass[classB] = revCountByClass[classB] + 1
    vocab = set()
    for i in range(3, len(review)):
        if (negation == 1):
            if (review[i].endswith(tuple(punctuation))):
                negation = 0
                review[i] = review[i].strip(punctuation)
            review[i] = review[i] + '_N'
        if (review[i].endswith(negSuffix)):
            negation = 1
        review[i] = review[i].strip(punctuation)
        vocab.add(review[i])
    for word in vocab:
        if word not in stopWords:
            VocabMat[word][classB] = VocabMat[word].get(classB, 0) + 1

for word in VocabMat:
    for c in classSet:
        VocabMat[word][c]=VocabMat[word].get(c,0)+0.65
        vocabCountByClass[c]=vocabCountByClass[c]+VocabMat[word][c]

for word in VocabMat:
    for c in classSet:
        VocabMat[word][c]=math.log(VocabMat[word][c]/vocabCountByClass[c])

for c in revCountByClass:
    revPrior[c]=math.log(revCountByClass[c]/revCount)

Model=[revPrior,VocabMat]
ModelFile=open('./nbmodel.txt','w')
ModelFile.write(json.dumps(Model))
ModelFile.close()


