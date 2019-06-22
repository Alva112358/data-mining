
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import threading
import random
import math
import time
from sklearn.model_selection import cross_val_score


# In[2]:


class readThread(threading.Thread):
    def __init__(self, filename):
        threading.Thread.__init__(self)
        self.filename = filename
        
    def run(self):
        self.df = pd.read_csv(self.filename, encoding='utf-8', header=None)
        print("Finish read {}".format(self.filename))
        
    def get_result(self):
        try:
            return self.df
        except Exception:
            return None


# In[3]:


# Load read train data set with multiprocess.
def mutil_read_process(file_dir):
    dataset = pd.DataFrame()
    threadList = []

    if (file_dir == './train'):
        thread1 = readThread('./train/train1.csv')
        thread2 = readThread('./train/train2.csv')
        thread3 = readThread('./train/train3.csv')
        thread4 = readThread('./train/train4.csv')
        thread5 = readThread('./train/train5.csv')
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread5.start()
        threadList.append(thread1)
        threadList.append(thread2)
        threadList.append(thread3)
        threadList.append(thread4)
        threadList.append(thread5)
    elif (file_dir == './label'):
        thread1 = readThread('./label/label1.csv')
        thread2 = readThread('./label/label2.csv')
        thread3 = readThread('./label/label3.csv')
        thread4 = readThread('./label/label4.csv')
        thread5 = readThread('./label/label5.csv')
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread5.start()
        threadList.append(thread1)
        threadList.append(thread2)
        threadList.append(thread3)
        threadList.append(thread4)
        threadList.append(thread5)
    elif (file_dir == './test'):
        thread1 = readThread('./test/test1.csv')
        thread2 = readThread('./test/test2.csv')
        thread3 = readThread('./test/test3.csv')
        thread4 = readThread('./test/test4.csv')
        thread5 = readThread('./test/test5.csv')
        thread6 = readThread('./test/test6.csv')
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread5.start()
        thread6.start()
        threadList.append(thread1)
        threadList.append(thread2)
        threadList.append(thread3)
        threadList.append(thread4)
        threadList.append(thread5)
        threadList.append(thread6)
            
    for thread in threadList:
        thread.join()
        dataset = dataset.append(thread.get_result(), ignore_index=True)  
        
    return dataset


# In[15]:


class read_once(threading.Thread):
    def __init__(self, filedir):
        threading.Thread.__init__(self)
        self.filedir = filedir
        
    def run(self):
        self.dataset = mutil_read_process(self.filedir) 
        
    def get_result(self):
        try:
            return self.dataset
        except Exception:
            return None
        
def read_data():
    dir_list = ['./train', './label', './test']
    thread_list = []
    result_list = []
    
    for _dir in dir_list:
        thread = read_once(_dir)
        thread.start()
        thread_list.append(thread)
        
    for thread in thread_list:
        thread.join()
        result_list.append(thread.get_result())
        
    return result_list


# In[14]:


start = time.time()
traindataset, labeldataset, testdataset = read_data()
end = time.time()
print("Total time parallel: {}s".format(end-start))


# In[6]:


print("Train dataset's length: {}".format(len(traindataset)))
print("Label dataset's length: {}".format(len(labeldataset)))
print("Test  dataset's length: {}".format(len(testdataset)))

print("Train data sample:")
print(traindataset.head())

print("Label data sample:")
print(labeldataset.head())


# In[7]:


# create a regression tree.
def createRegressionTree(dataset, features, max_depth=20):

    # get the best split feature and value.
    tree = {}
    bestFeature, bestValue = splitDataSet(dataset, features)
    if (bestFeature == -1 or max_depth == 0):
        tree["result"] = np.mean(dataset[13])
        #print(tree["result"])
        return tree

    # split dataset into two separate part according to the best split feature and value.
    leftData, rightData = [], []
    for i in range(len(dataset)):
        if (dataset[bestFeature][i] <= bestValue):
            leftData.append(dataset.iloc[i,:])
        else:
            rightData.append(dataset.iloc[i,:])
    leftData = pd.DataFrame(np.array(leftData)).reset_index(drop=True)
    rightData = pd.DataFrame(np.array(rightData)).reset_index(drop=True)

    
    if (leftData.size == 0):
        tree["result"] = np.mean(rightData[13])
    elif (rightData.size == 0):
        tree["result"] = np.mean(leftData[13])
    else:
        tree["result"] = None
        tree["Feature"] = bestFeature
        tree["Value"] = bestValue
        features.remove(bestFeature)
        tree["left"] = createRegressionTree(leftData, features, max_depth-1)
        tree["right"] = createRegressionTree(rightData, features, max_depth-1)

    return tree


# find the best feature and value to split the data.
# dataset : the data to be trained, type is dataframe, size >= 2.
# features: the remained features, type is array, not null.
def splitDataSet(dataset, features):
    bestFeature = -1
    bestValue = -1
    minLoss = float('inf')

    for featureIndex in features:
        for i in range(len(dataset)-1):
            splitValue = (dataset[featureIndex][i] + dataset[featureIndex][i+1]) / 2
            loss = leastSquareLoss(dataset, featureIndex, splitValue)

            if (loss < minLoss):
                bestFeature, bestValue, minLoss = featureIndex, splitValue, loss

    #print("Best Feature: {}, Best Value: {}".format(bestFeature, bestValue))
    return bestFeature, bestValue


# calculate the least square loss.
# dataset: type is dataframe.
# featureIndex: the feature to be classified.
# splitValue: the criterion to split right and left.
# format: Loss(Left_data) + Loss(Right_data).
def leastSquareLoss(dataset, featureIndex, splitValue):
    leftData, rightData = [], []
    for i in range(len(dataset)):
        if (dataset[featureIndex][i] <= splitValue):
            leftData.append(dataset[featureIndex][i])
        else:
            rightData.append(dataset[featureIndex][i])
    leftData  = np.array(leftData)
    rightData = np.array(rightData)

    # calculate variance in case of empty array.
    leftVar  = leftData.var() if (leftData.size != 0) else 0
    rigthVar = rightData.var() if (rightData.size != 0) else 0

    return leftVar + rigthVar


def treePredict(tree, testdata):
    while (tree["result"] == None):
        if (testdata[tree["Feature"]] <= tree["Value"]):
            tree = tree["left"]
        elif (testdata[tree["Feature"]] > tree["Value"]):
            tree = tree["right"]

    return tree["result"]


# In[20]:


class createTree(threading.Thread):
    def __init__(self, traindataset, max_train=1.0, max_feature=1.0):
        threading.Thread.__init__(self)
        self.traindataset = traindataset
        self.max_train = max_train
        self.max_feature = max_feature
        self.tree_num = int(max_train*len(traindataset))
        self.tree_fes = int((max_feature*traindataset.shape[1]-1))
        
    def run(self):
        data = self.traindataset.sample(n=self.tree_num)
        data = data.reset_index(drop=True)
        features = [i for i in range(data.shape[1]-1)]
        features = random.sample(features, self.tree_fes)
        self.tree = createRegressionTree(data, features)
        #print("Finish")
        
    def get_result(self):
        try:
            return self.tree
        except Exception:
            return None


# In[21]:


class RandomForest():
    def __init__(self, n_estimator=10, max_train=1.0, max_feature=1.0, max_depth=15):
        self.n_estimator = n_estimator
        self.max_train = max_train
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.treeList = []
        
    def fit(self, traindataset):
        threadList = []
        for i in range(self.n_estimator):
            thread = createTree(traindataset, max_train=self.max_train, max_feature=self.max_feature)
            thread.start()
            threadList.append(thread)
            
        for i in range(self.n_estimator):
            threadList[i].join()
            self.treeList.append(threadList[i].get_result())
        
    def predict(self, data):
        result = []
        treeValues = []
        for tree in self.treeList:
            value = treePredict(tree, data)
            treeValues.append(value)
        result.append(np.mean(np.array(treeValues)))
        return np.mean(np.array(result))


# In[ ]:


n_estimator = 10
max_train = 0.00001
max_feature = 1.0
test_num = len(testdataset)
result = []

trainset = np.concatenate((traindataset, labeldataset), axis=1)
df = pd.DataFrame(trainset)
df = df.reset_index(drop=True)

print("Train dataset length: {}".format(int(len(trainset)*max_train)))
print("Test  dataset length: {}".format(test_num))

start = time.time()
clf = RandomForest(n_estimator=n_estimator, max_train=max_train, max_feature=max_feature)
clf.fit(df)
end = time.time()
print("Random Forest Train Time: {:.2f}s".format(end-start))


for i in range(test_num):
    result.append(clf.predict(testdataset.iloc[i,:]))
result = np.array(result)
print(result)


# In[27]:


#Identity = np.array(np.array(range(1,testdataset.shape[0]+1)))
Identity = np.array(np.array(range(1,len(result)+1)))
Prediction = result
Result = pd.DataFrame({'id':Identity, 'Predicted':Prediction})
print(Result)
Result.to_csv("Forest_t.csv", encoding='utf-8_sig', index=False)

