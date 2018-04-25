import json
import sys
import random
from string import punctuation

inputFile = open(sys.argv[2], 'r')
#inputFile = open('./corpus/dev-text.txt', 'r')
input = [i.strip().split() for i in inputFile.readlines()]

ModelFile= open(sys.argv[1], 'r')
#ModelFile = open('./vanillamodel.txt')
featWeight = json.loads(ModelFile.read())
ModelFile.close()

negSuffix = ('n\'t', 'not', 'no', 'never', "n't")

for review in input:
    for i in range(3, len(review)):
        review[i] = review[i].lower().strip().strip(punctuation)
        if review[i].endswith(tuple(negSuffix)) and i + 1 < len(review):
            review[i + 1] = 'n_' + review[i + 1]

for review in input:
    vocab=set()
    rmv=[]
    for i in review:
        if i in vocab:
            rmv.append(i)
        else:
            vocab.add(i)
    for item in rmv:
        review.remove(item)

output=''
for review in input:
    output=output+review[0]+' '
    tf=0
    pn=0
    for i in range(3, len(review)):
        if review[i] in featWeight:
            tf = tf + featWeight[review[i]]['TF']
            pn = pn + featWeight[review[i]]['PN']
    tf=tf+featWeight['*BIAS']['TF']
    pn=pn+featWeight['*BIAS']['PN']
    if tf>=0:
        output=output+'True '
    else:
        output=output+'Fake '
    if pn>=0:
        output=output+'Pos\n'
    else:
        output=output+'Neg\n'
OutputFile = open('./percepoutput.txt', 'w')
OutputFile.write(output)
OutputFile.close()

