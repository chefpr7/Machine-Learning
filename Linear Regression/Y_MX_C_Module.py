#!/usr/bin/env python
# coding: utf-8

# In[5]:


#module for linear regression
import numpy as np
import random
import math

class LinR:
    
     
    def fitt(self,Xin,Yin,alpha,i,length):
        self.X=Xin  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.M=np.random.rand(2,1)
        self.C=np.ones((Xin.shape[0],1))*random.random()
        #self.gradientdes(alpha,i)
        self.mini_gradient_descent(length,alpha,i)
    
    
     #Hypothesis function
    def hypothesis(self,X,C):
        H=X.dot(self.M)+C
        return H
    
    
    #Cost function
    def cost(self):
        H=self.hypothesis(self.X,self.C)
        J = sum((H-self.Y)**2)/(2*self.X.shape[0])
        return J
                                                                                                                   
                                                                                                                                    
    #gradient descent
    def gradientdes(self,alpha=0.1,i=1000):
        m=self.X.shape[0]
        for j in range(i):
            H=self.hypothesis(self.X,self.C)
            DM=((H - self.Y).T.dot(self.X).T)/m
            DC= (H - self.Y)/m
            self.M = self.M - DM*alpha
            self.C = self.C - DC*alpha
        
    
    #mini gradient descent
    def mini_gradient_descent(self,length,alpha=0.1,i=10000):
        m=self.X.shape[0]
        v=length
        r=math.ceil(m/length)
        for j in range(i):
            for k in range(r):
                u=k*length
                if(u+v>m):
                    v=m-u
                    
                X1=self.X[u:u+v,:]
                Y1=self.Y[u:u+v,:]
                C1=self.C[u:u+v,:]
               
                H=self.hypothesis(X1,C1)
                DM=((H - Y1).T.dot(X1).T)/v
                DC=(H - Y1)/v
                
                self.M = self.M - DM*alpha
                C1 = C1 - DC*alpha
                self.C[u:u+v,:]=C1
                
                if(j==0):
                    print(C1.shape)
    
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
        Y_pred=self.hypothesis(X,C1)
        return Y_pred
        
        
    
    
    
 # In[ ]:




