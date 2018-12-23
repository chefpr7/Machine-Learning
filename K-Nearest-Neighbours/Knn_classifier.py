#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class KNN:
    
    def fit(self,Xin,Yin,k=0):
        self.X = Xin
        self.Y = Yin
        self.k = k
        if(k==0):
            self.k=np.unique(self.Y).shape[0]
        
    
    def KNN(self):
        knn=[0]*self.m
        for i in range(self.m): 
            knn[i] = np.sqrt(np.sum(np.square(self.X - self.X_test[i]),axis=1))
            neighbours = self.Y[np.argsort(knn[i])]
            c = np.bincount(neighbours[0:self.k].astype(int))
            knn[i]=np.argmax(c)
        return knn
   


    def pred(self,X_test):
        self.m=X_test.shape[0]
        self.X_test=X_test
        return self.KNN()
        
        
        
    def Accuracy(self,Y_test,Y_pred):
        return np.mean(Y_test==Y_pred)*100
    
    
    
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 
    
    

