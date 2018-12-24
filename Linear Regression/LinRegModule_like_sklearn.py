#!/usr/bin/env python
# coding: utf-8

# In[5]:


#module for linear regression
import numpy as np
import math

class LinR:
        
    def fitt(self,Xin,Yin,alpha=0.1,i=10000,length=0):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)
        
        if(length==0):
            length=self.X.shape[0]//10   # Initializes the size of the mini batches if not provided by user
        
        self.alpha=alpha
        self.length=length
        self.i=i
        
        self.mini_gradient_descent()

    
    #Hypothesis function
    def hypothesis(self,X):
        H=X.dot(self.theta)
        return H
    
    
    #Cost function
    def cost(self):
        H=self.hypothesis(self.X)
        J = sum((H-self.Y)**2)/(2*self.X.shape[0])
        return J
    

    #gradient descent
    def gradientdes(self):
        m=self.X.shape[0]
        D=np.zeros((3,1))
        for j in range(self.i):
            H=self.hypothesis(self.X)
            D=((H - self.Y).T.dot(self.X).T)/m
            self.theta = self.theta - D*self.alpha
        return self.theta
    
    
    def mini_gradient_descent(self):
   # Again,functions like gradient descent BUT does so in small batches, hence named mini batch gradient descent
        m=self.X.shape[0]
        r=math.ceil(m/self.length)               # decides the no of mini batches
        
        for j in range(self.i):
            v=self.length                        # manages overflowing
            for k in range(r):
                u=k*self.length
                if(u+v>m):
                    v=m-u
                X1=self.X[u:u+v,:]
                Y1=self.Y[u:u+v,:]
                
                H=self.hypothesis(X1)
                D=((H - Y1).T.dot(X1).T)/v
                self.theta = self.theta - D*self.alpha
        return self.theta
    
    
    def normalize(self,X):
    #normalization for x
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X    
    
    def accuracy(self,y_test,y_pred):
    # Calculates the accuracy of the prediction , always satisfying when it matches sklearn
        err=(y_pred-y_test)*100/y_test
        return 100-np.mean(err) 

    #predictions on test data
    def predict(self,X):
        X=np.insert(X,0,1,axis=1)
        return self.hypothesis(X)
        
    
    
    
 # In[ ]:




