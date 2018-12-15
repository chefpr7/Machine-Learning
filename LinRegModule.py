#!/usr/bin/env python
# coding: utf-8

# In[5]:


#module for linear regression
import numpy as np

class LinR:
    
    def __init__(self,Xin,Yin):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)

    
    #Hypothesis function
    def hypothesis(self,X,theta):
        H=np.zeros((X[:,0].size,1))
        H=X.dot(theta)
        return H
    
    
    #Cost function
    def cost(self):
        J = sum((self.X.dot(self.theta)-self.Y)**2)/(2*self.X.shape[0])
        return J
    

    #gradient descent
    def gradientdes(self,alpha=0.1,i=1000):
        m=self.X.shape[0]
        D=np.zeros((3,1))
        for j in range(i):
            D=((((self.X.dot(self.theta) - self.Y).T).dot(self.X)).T)/m
            self.theta = self.theta - D*alpha
        return self.theta
    
   
    
    
    
       
#feature scaling for x
def normalize(X):
    R=np.std(X,axis=0)
    M=np.mean(X,axis=0)
    X=(X-M)/R
    return X    
    
def accuracy(y_test,y_pred):
    err=(y_pred-y_test)*100/y_test
    return 100-np.mean(err) 

#predictions on test data
def predict(X,theta):
    X=np.insert(X,0,1,axis=1)
    return X.dot(theta)
 

    
    
        
        
    
        


# In[ ]:




