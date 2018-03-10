import numpy as np
import Data
import math

'''
# of hidden layer:2 
hidden layer 1: 3 elements 1 bias unit
hidden layer 2: 2 elements 1 bias unit
output layer: 10 units
learning rate : alpha=0.1 sigmod function:1/(1+np.exp(-x))
Theta1: 3*1025 Theta2: 2*4 Theta3: 10*3
'''

def sigmod(mat):
    return 1/(1+np.exp(-mat))

Theta1=np.ones([3,1025]) 
Theta2=np.ones([2,4]) 
Theta3=np.ones([10,3])

def Initialization(Theta):
    Int=1/np.sqrt(Theta.shape[1])
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            Theta[i,j]=np.random.uniform(-Int,Int,1)
    return Theta

Theta1=Initialization(Theta1)
Theta2=Initialization(Theta2)
Theta3=Initialization(Theta3)
Theta=[Theta1,Theta2,Theta3]

def Calculation(Theta):
    [trainingSet, origLabels, numOftrain] = Data.TrainingdataTransfer()

    Traininglabels = np.zeros([numOftrain,10])
    for i in range(numOftrain):
        Traininglabels[i, origLabels[i]] = 1

    trainingSet=np.hstack((np.ones([numOftrain,1]),trainingSet))

    T=0
    Delta1=np.zeros([3,1025])
    Delta2=np.zeros([2,4])
    Delta3=np.zeros([10,3])

    for i in range(numOftrain):
        z1=np.dot(Theta[0],trainingSet[[i],:].T)
        act1=np.vstack((1,sigmod(z1)))
        z2=np.dot(Theta[1],act1)
        act2=np.vstack((1,sigmod(z2)))
        z3=np.dot(Theta[2],act2)
        act3=sigmod(z3)
        error=0.5*(act3.T-Traininglabels[[i],:])*(act3.T-Traininglabels[[i],:])
        delta3=(act3-Traininglabels[[i],:].T)*(act3*(1-act3))
        Delta3=Delta3+np.dot(delta3,act2.T)
        delta2=np.dot(np.delete(Theta[2].T,0,0),delta3)*(sigmod(z2)*(1-sigmod(z2)))
        Delta2=Delta2+np.dot(delta2,act1.T)
        delta1=np.dot(np.delete(Theta[1].T,0,0),delta2)*(sigmod(z1)*(1-sigmod(z1)))
        Delta1=Delta1+np.dot(delta1,trainingSet[[i],:])
        Detla=[Delta1,Delta2,Delta3]
        T = error.sum(1) + T
    return Detla,T


def check(Theta):
    [trainingSet, origLabels, numOftrain] = Data.TrainingdataTransfer()

    Traininglabels = np.zeros([numOftrain, 10])
    for i in range(numOftrain):
        Traininglabels[i, origLabels[i]] = 1

    trainingSet = np.hstack((np.ones([numOftrain, 1]), trainingSet))

    T = 0

    for i in range(numOftrain):
        z1 = np.dot(Theta[0], trainingSet[[i], :].T)
        act1 = np.vstack((1, sigmod(z1)))
        z2 = np.dot(Theta[1], act1)
        act2 = np.vstack((1, sigmod(z2)))
        z3 = np.dot(Theta[2], act2)
        act3 = sigmod(z3)
        error = 0.5 * (act3.T - Traininglabels[[i], :]) * (act3.T - Traininglabels[[i], :])
        T = error.sum(1) + T
    return T

for i in range(3):
    print(Theta[0][0, 0])
    [Delta,Cost]=Calculation(Theta)
    print(Delta[0][0, 0])

    Theta_0=Theta[0]

    Theta[0][0, 0] = Theta[0][0, 0] + 0.0001
    Theta = [Theta1, Theta2, Theta3]
    T1 = check(Theta)
    print(T1)
    Theta[0][0, 0] = Theta[0][0, 0] - 0.0002
    Theta = [Theta1, Theta2, Theta3]
    T2 = check(Theta)
    print(T2)
    print((T1-T2)/(0.0002))
    Theta1 = Theta_0- 0.01* Delta[0]
    Theta2 = Theta[1] - 0.01* Delta[1]
    Theta3 = Theta[2] - 0.01* Delta[2]
    Theta=[Theta1,Theta2,Theta3]

'''
def exe(Theta):
    for i in range(10):
        [Delta,Cost]=Calculation(Theta)
        print (Cost)
        for j in range(3):
            Theta[j]=Theta[j]-0.01*Delta[j]
        Theta=[Theta[0],Theta[1],Theta[2]]
    return Theta

Theta_test=exe(Theta)

def Validation(Theta):
    [testSet, origLabels, numOftest] = Data.TestdataTransfer()
    Testlabels = np.zeros([numOftest, 10])
    for i in range(numOftest):
        Testlabels[i, origLabels[i]] = 1

    testSet = np.hstack((np.ones([numOftest, 1]), testSet))

    count=0

    for i in range(numOftest):
        vect=testSet[[i],:].T
        z1 = np.dot(Theta[0], vect)
        act1 = np.vstack((1, sigmod(z1)))
        z2 = np.dot(Theta[1], act1)
        act2 = np.vstack((1, sigmod(z2)))
        z3 = np.dot(Theta[2], act2)
        act3 = sigmod(z3)
        error = 0.5 * (act3.T - Testlabels[[i], :]) * (act3.T - Testlabels[[i],:])
        print (error)
        testresu=np.argmax(error)
        if testresu == origLabels[i]:
            count=count+1

    rate=count/numOftest
    return rate

error_rate=Validation(Theta_test)
print (error_rate)
'''



