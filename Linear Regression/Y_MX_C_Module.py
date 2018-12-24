#!/usr/bin/env python
# coding: utf-8

# In[5]:


#module for linear regression
import numpy as np
import random
import math

class LinR:
    
     
    def fitt(self,Xin,Yin,alpha=0.1,i=10000,length=0):
        self.X=Xin  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.M=np.random.rand(2,1)
        self.C=np.ones((Xin.shape[0],1))*random.random()
        
        if(length==0):
            length=self.X.shape[0]//10   # Initializes the size of the mini batches if not provided by user
        
        self.alpha=alpha
        self.length=length
        self.i=i
        
        #self.gradientdes(alpha,i)
        self.mini_gradient_descent()
    
    
     #Hypothesis function
    def hypothesis(self,X,C):
        H=X.dot(self.M)+C
        return H
    
    
    #Cost function
    def cost(self):
        H=self.hypothesis(self.X,self.C)
        J = sum((H-self.Y)**2)/(2*self.X.shape[0])
        return J
                                                                                                                   
                                                                                                                                    
    
    def gradientdes(self):
    # Derivatives being calculated,this function directs the weights into a journey towards the minima 
        m=self.X.shape[0]
        for j in range(self.i):
            H=self.hypothesis(self.X,self.C)
            DM=((H - self.Y).T.dot(self.X).T)/m
            DC= (H - self.Y)/m
            self.M = self.M - DM*self.alpha
            self.C = self.C - DC*self.alpha
        
    
    # Again,functions like gradient descent BUT does so in small batches, hence named mini batch gradient descent
    def mini_gradient_descent(self):
        
        m=self.X.shape[0]
        r=math.ceil(m/self.length)               # determines the no of mini batches
        
        for j in range(self.i):
            v=self.length
            for k in range(r):
                u=k*self.length
                if(u+v>m):
                    v=m-u                         # manages overflowing
                    
                X1=self.X[u:u+v,:]
                Y1=self.Y[u:u+v,:]
                C1=self.C[u:u+v,:]
               
                H=self.hypothesis(X1,C1)
                DM=((H - Y1).T.dot(X1).T)/v
                DC=(H - Y1)/v
                
                self.M = self.M - DM*self.alpha
                C1 = C1 - DC*self.alpha
                self.C[u:u+v,:]=C1
                
   
    
    #normalization for x
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X    
    
    
    def accuracy(self,y_test,y_pred):
     # Calculates the accuracy of the model,always satisfying if it matches sklearn
        err=(y_pred-y_test)*100/y_test
        return 100-np.mean(err) 

    
    #predictions on test data
    def predict(self,X):
        C1=self.C[0:X.shape[0],:]
        Y_pred=self.hypothesis(X,C1)
        return Y_pred
        
        
    
    
    
 # In[ ]:




