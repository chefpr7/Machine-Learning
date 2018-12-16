#!/usr/bin/env python
# coding: utf-8

# In[5]:


#module for linear regression
import numpy as np
import random

class LinR:
    
   # def __init__(self):    
    def fitt(self,Xin,Yin,alpha,i,length):
        self.X=Xin  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.M=np.random.rand(2,1)
        self.C=np.ones((Xin.shape[0],1))*random.random()
        self.gradientdes(alpha,i)
        #self.mini_gradient_descent(length,alpha,i)
    
    #Cost function
    def cost(self):
        J = sum((self.X.dot(self.M)+C-self.Y)**2)/(2*self.X.shape[0])
        return J
    
 
    #gradient descent
    def gradientdes(self,alpha=0.1,i=1000):
        m=self.X.shape[0]
        for j in range(i):
            DM=((((self.X.dot(self.M)+self.C - self.Y).T).dot(self.X)).T)/m
            DC=(self.X.dot(self.M)+self.C - self.Y)/m
            self.M = self.M - DM*alpha
            self.C = self.C - DC*alpha
        return self.M,self.C
    
    #mini gradient descent
    def mini_gradient_descent(self,length,alpha=0.1,i=10000):
        m=self.X.shape[0]
        v=length
        for j in range(i):
            for k in range(m//length):
                u=k*length
                if(u+v>m):
                    v=m-u
                X1=self.X[u:u+v,:]
                Y1=self.Y[u:u+v,:]
                C1=self.C[u:u+v,:]
                
                DM=((((X1.dot(self.M)+C1 - Y1).T).dot(X1)).T)/v
                DC=(X1.dot(self.M)+C1 - Y1)/v
                self.M = self.M - DM*alpha
                self.C[u:u+v,:] = self.C[u:u+v,:] - DC*alpha
        return self.M,self.C
    
    #normalization for x
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X    
    
    def accuracy(self,y_test,y_pred):
        err=(y_pred-y_test)*100/y_test
        return 100-np.mean(err) 

    #predictions on test data
    def predict(self,X):
        C1=self.C[0:X.shape[0],:]
        Y_pred=X.dot(self.M)+C1
        return Y_pred
        
        
    
    
    
 # In[ ]:




