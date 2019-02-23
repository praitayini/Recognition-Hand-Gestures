 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:09:09 2016

@author: praitayinikanakaraj
"""
import mymyo
import numpy as np
import matplotlib.pyplot as plt
import mlutils as ml
import neuralnetworksbylayer as nn
import qdalda as ql
import time

plt.ion()


def sampleDataAndTrain(segmentSize, X, T):
    nSegments = int(X.shape[0] / segmentSize)

    new_X = []  
    new_T = []  

    nRowsInX = int(X.shape[0] / segmentSize) * segmentSize
    for firstRow in range(0,nRowsInX,segmentSize):
        oneSegmentX = X[firstRow:firstRow+segmentSize, :]
        oneSegmentT = T[firstRow:firstRow+segmentSize, :]
        if np.all(oneSegmentT[0] == oneSegmentT):
            oneSegmentX = oneSegmentX.reshape((-1))
            # nCols = len(oneSegment)
            new_X.append( oneSegmentX.tolist() ) # [row,:nCols] = oneSegment[:]
            new_T.append( [oneSegmentT[0]] )
    new_X = np.array(new_X)
    new_T = np.array(new_T).reshape((-1,1))
    #new_T, new_X
    if 'LDA' in input('Which classifier would you like to train your data on? (LDA or NN)'):
        lda = ql.LDA()
        lda.train(new_X,new_T)
        resultsLDA = ml.trainValidateTestKFoldsClassification( trainLDA,useLDA, new_X,new_T, [None],
                                                       nFolds=5, shuffle=False,verbose=False)
        printResults('LDA:',resultsLDA)
        return lda
    else:
        Val=np.unique(new_T)
        numberOfHiddenUnits = 10
        nnetwk = nn.NeuralNetworkClassifier([new_X.shape[1], numberOfHiddenUnits, len(Val)])
        nnetwk.train(new_X,new_T,100,verbose=True)
        resultsNN = ml.trainValidateTestKFoldsClassification( trainNN,evaluateNN, new_X, new_T, 
                                                     [ [ [0], 10], [[10], 100] ],
                                                     nFolds=6, shuffle=False,verbose=False)
        printResults('NN:',resultsNN)
        return nnetwk
    

                
def trainLDA(new_X,new_T,parameters=None):
    lda_val=ql.LDA()
    lda_val.train(new_X,new_T)
    return lda_val
    
def useLDA(model,new_X,new_T):
    col,p,dev = model.use(new_X)
    return np.sum(col==new_T)/new_X.shape[0] * 100
    
def trainNN(new_X,new_T,parameters):
    Val=np.unique(new_T)
    nnetwk = nn.NeuralNetworkClassifier(new_X.shape[1],parameters[0],len(Val)) 
    nnetwk.train(new_X,new_T,parameters[1],errorPrecision=1.e-8, verbose=True)
    return nnetwk

def evaluateNN(model,new_X,new_T):
    p_test,pro_test,_ = model.use(new_X,allOutputs=True)
    return 100*np.sum(p_test==new_T)/len(new_T)
    
def printResults(label,results):
            print('{:4s} {:>20s}{:>8s}{:>8s}{:>8s}'.format('Algo','Parameters','TrnAcc','ValAcc','TesAcc'))
            print('-------------------------------------------------')
            for row in results:
                print('{:>4s} {:>20s} {:7.2f} {:7.2f} {:7.2f}'.format(label,str(row[0]),*row[1:]))
    
def segmentX(segmentSize, X):
    new_X = []
    nRowsInX = int(X.shape[0] / segmentSize) * segmentSize
    for firstRow in range(0,nRowsInX,segmentSize):
        oneSegmentX = X[firstRow:firstRow+segmentSize, :]
        oneSegmentX = oneSegmentX.reshape((-1))
        new_X.append( oneSegmentX.tolist() ) # [row,:nCols] = oneSegment[:]
    new_X = np.array(new_X)
    return new_X   


emgs = []
def handleEMG(emg, moving):
    global emgs
    emgs.append(emg)

def connectWithEMGHandler(handler):
    print('Connecting to myo ...',end="")
    myo = mymyo.MyoRaw(None)
    myo.add_emg_handler(handler)
    myo.connect()
    print('Connected')
    return myo


if 'y' in input('Re-train? (y or n) '):
    print('Enter task names you want to train, one per line. End with empty line.')
    done = False
    tasks =[]
    while not done:
        s = input(': ')
        if s == '':
            done = True
        else:
            tasks.append(s)
    print('Tasks are',tasks)

    nSamplesEach = 500



    taskemgs = []
    for taski,task in enumerate(tasks):
        myo = connectWithEMGHandler(handleEMG)
        print('------------------------------DO TASK',task)
        print()
        # time.sleep(1)
        try:
            emgs = []
            while len(emgs) < nSamplesEach:
                myo.run(1)
                # print('len(emgs)',len(emgs))
        finally:
            myo.disconnect()
            print('finally reached')

        taskemgs += emgs


    X = np.array(taskemgs)
    T = np.tile(range(1,len(tasks)+1), (nSamplesEach,1)).T.reshape((1,-1)).T
    print('len(X)',len(X),'len(T)',len(T))


    if False:
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(X)
        plt.subplot(2,1,2)
        plt.plot(T)

if 'y' in input('Try new segment size? (y or n) '):
    print ('Enter number of samples.')
    segmentSize = int(input(': '))

    model = sampleDataAndTrain(segmentSize, X, T)
    print('done training')

def classifyEMG(emg, moving):
    global emgs
    emgs.append(emg)
    if len(emgs) > segmentSize:
        Xtest = np.array(emgs)
        XtestSeg = segmentX(segmentSize,Xtest)
        if isinstance(model, nn.NeuralNetwork):
            predict = model.use(XtestSeg)
        else:
            predict,_,_ = model.use(XtestSeg)
        print('Predicted task', tasks[predict-1])
        emgs = []

emgs = []
myo = connectWithEMGHandler(classifyEMG)


#myo.connect()
time.sleep(1)
print('start')

try:
    while True:
        myo.run(None)
finally:
    myo.disconnect()
    print()


