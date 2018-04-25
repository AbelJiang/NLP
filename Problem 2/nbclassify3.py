import json
import sys
import random
from string import punctuation
from collections import defaultdict

#inputFile = open(sys.argv[1], 'r')
inputFile = open('./corpus/dev-text.txt', 'r')
input = [i.strip().split(' ') for i in inputFile.readlines()]

ModelFile = open('./nbmodel.txt')
Model = json.loads(ModelFile.read())
ModelFile.close()

revPrior = Model[0]
VocabMat = Model[1]
negSuffix = ('n\'t', 'not', 'no', 'never', "n't")
output = ''
out = defaultdict(list)

for review in input:
    vocab = set()
    for i in range(1, len(review)):
        review[i] = review[i].lower()
        vocab.add(review[i].strip(punctuation))
    revPosterior = {'True': revPrior['True'], 'Fake': revPrior['Fake']}
    for word in vocab:
        if word in VocabMat:
            for c in revPosterior:
                revPosterior[c] = revPosterior[c] + VocabMat[word][c]
    if revPosterior['True'] > revPosterior['Fake']:
        out[review[0]].append('True')
    elif revPosterior['True'] < revPosterior['Fake']:
        out[review[0]].append('Fake')
    else:
        out[review[0]].append(random.choice(['True','Fake']))

for review in input:
    negation = 0
    vocab = set()
    for i in range(1, len(review)):
        if (negation == 1):
            if (review[i].endswith(tuple(punctuation))):
                negation = 0
                review[i] = review[i].strip(punctuation)
            review[i] = review[i] + '_N'
        if (review[i].endswith(negSuffix)):
            negation = 1
        review[i] = review[i].strip(punctuation)
        vocab.add(review[i])
    revPosterior = {'Pos': revPrior['Pos'], 'Neg': revPrior['Neg']}
    for word in vocab:
        if word in VocabMat:
            for c in revPosterior:
                revPosterior[c] = revPosterior[c] + VocabMat[word][c]
    if revPosterior['Pos'] > revPosterior['Neg']:
        out[review[0]].append('Pos')
    elif revPosterior['Pos'] < revPosterior['Neg']:
        out[review[0]].append('Neg')
    else:
        out[review[0]].append(random.choice(['Pos', 'Neg']))

for hypo in out:
    output = output + hypo + ' ' + out[hypo][0] + ' ' + out[hypo][1] + '\n'
OutputFile = open('./nboutput.txt', 'w')
OutputFile.write(output)
OutputFile.close()
