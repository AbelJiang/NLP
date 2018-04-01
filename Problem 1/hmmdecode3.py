import json
import sys

InputFile=open(sys.argv[1],'r')
Input=[i.strip().split(' ') for i in InputFile.readlines()]

ModelFile=open('./hmmmodel.txt')
Model=json.loads(ModelFile.read())
ModelFile.close()

A=Model[0]
B=Model[1]

output=''
for sentence in Input:
    viterbi={}
    backpointer={}
    viterbi[0]={}
    backpointer[0]={}
    #Initialization
    for state in B.keys():
        viterbi[0][state]=A['start'].get(state,A['start']['other'])*B[state].get(sentence[0],B[state]['other'])
        backpointer[0][state]='start'
    for t in range(len(sentence)-1):
        viterbi[t+1]={}
        backpointer[t+1]={}
        for state in B.keys():
            candi={s:viterbi[t][s]*A[s].get(state,A[s]['other'])*B[state].get(sentence[t+1],B[state]['other']) for s in B.keys()}
            result=max(candi.items(),key=lambda k: k[1])
            viterbi[t+1][state]=result[1]
            backpointer[t+1][state]=result[0]
    candiEnd={s:viterbi[len(sentence)-1][s]*A[s].get('end',A[s]['other']) for s in B.keys()}
    result=max(candiEnd.items(),key=lambda k: k[1])
    viterbi['end']=result[1]
    backpointer['end']=result[0]
    tag=backpointer['end']
    for i in range(len(sentence)):
        sentence[-i-1]=sentence[-i-1]+'/'+tag+' '
        tag=backpointer[len(sentence)-i-1][tag]
    output=output+''.join(sentence).strip()+'\n'
OutputFile=open('./hmmoutput.txt','w')
OutputFile.write(output)
OutputFile.close()
