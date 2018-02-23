import json
import sys

CorpusFile=open('./corpus/en_train_tagged.txt','r')
Corpus=[i.strip().split(' ') for i in CorpusFile.readlines()]
CorpusFile.close()

A={}
B={}

A['start']={}
A['end']={}

for line in Corpus:
    start=line[0].rsplit('/',1)
    end=line[-1].rsplit('/',1)
    A['start'][start[1]]=A['start'].get(start[1],0)+1
    A['end'][end[1]]=A['end'].get(end[1],0)+1
    B[end[1]]=B.get(end[1],{})
    B[end[1]][end[0]]=B[end[1]].get(end[0],0)+1
    next_word=start
    for i in range(len(line)-1):
        cur_word=next_word
        next_word=line[i+1].rsplit('/',1)
        A[cur_word[1]]=A.get(cur_word[1],{})
        A[cur_word[1]][next_word[1]]=A[cur_word[1]].get(next_word[1],0)+1
        B[cur_word[1]]=B.get(cur_word[1],{})
        B[cur_word[1]][cur_word[0]]=B[cur_word[1]].get(cur_word[0],0)+1  
totalTrans=len(B)
totalWords=0

for k in B:
    totalWords=totalWords+len(B.get(k))
        
for k in A:
    items=A.get(k)
    total=sum(items.values())
    pTrans=1/(total+len(items))
    for trans in items:
        A[k][trans]=A[k][trans]/total-pTrans
    A[k]['other']=len(items)*pTrans/(totalTrans-len(items))

    
        
for k in B:
    if not k in A:
        A[k]={'other':0}
    items=B.get(k)
    total=sum(items.values())
    pEm=1/(total+len(items))
    for em in items:
        B[k][em]=B[k][em]/total-pEm
    B[k]['other']=len(items)*pEm/(totalWords-len(items))
        
Model=[A,B]
ModelFile=open('./hmmmodel.txt','w')
ModelFile.write(json.dumps(Model))
ModelFile.close()