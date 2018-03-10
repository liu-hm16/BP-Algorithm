import numpy as np
from os import listdir

def img2vector(filename):
    returnVect = np.zeros([1,1024])
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def TrainingdataTransfer():
    hwLabels =[]
    trainingFileList = listdir('/Users/Huiming/Desktop/Directory/Python/knn/trainingDigits')
    m=len(trainingFileList)
    trainMat=np.zeros([m,1024])
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i,:]=img2vector('/Users/Huiming/Desktop/Directory/Python/knn/trainingDigits/%s' % fileNameStr)
    hwLabels=np.array(hwLabels)
    return trainMat,hwLabels,m

def TestdataTransfer():
    hwLabels =[]
    testFileList = listdir('/Users/Huiming/Desktop/Directory/Python/knn/testDigits')
    m=len(testFileList)
    testMat=np.zeros([m,1024])
    for i in range(m):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        testMat[i,:]=img2vector('/Users/Huiming/Desktop/Directory/Python/knn/testDigits/%s' % fileNameStr)
    hwLabels=np.array(hwLabels)
    return testMat,hwLabels,m