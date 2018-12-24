#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class KNN:
    
    def fit(self,Xin,Yin,k=0):
    # This function initializes the training set and k
        self.X = Xin
        self.Y = Yin
        self.k = k
        if(k==0): 
            self.k=np.unique(self.Y).shape[0]      # Initializes k if not given by the user
        
    
    def KNN(self):
        #  
        knn=[0]*self.m
        for i in range(self.m): 
            knn[i] = np.sqrt(np.sum(np.square(self.X - self.X_test[i]),axis=1)) # calculates and stores the distance between each test data                                                                                   example and every training example
            neighbours = self.Y[np.argsort(knn[i])]                             # sorts the values of Y accordingly as knn[i] is sorted
            c = np.bincount(neighbours[0:self.k].astype(int))
            knn[i]=np.argmax(c)                                                 # y with largest frequency amongst the k nearest neighbours 
        return knn
   


    def pred(self,X_test):
        #predicts the values on the testing data provided by user by calling KNN function
        self.m=X_test.shape[0]
        self.X_test=X_test
        return self.KNN()
        
        
        
    def Accuracy(self,Y_test,Y_pred):
        # Calculates the accuracy of the prediction , always satisfying when it matches sklearn
        return np.mean(Y_test==Y_pred)*100
    
    
    
    def normalize(self,X):
        # Normalizes the features
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 
    
    

