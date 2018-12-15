#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class LogR:
    def __init__(self,Xin,Yin):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)
        
    def sigmoid_func(self):
        Z=1/(1+np.exp(self.X.dot(self.theta)))
        return Z
    
    
    def cost_func(self,Z):
        cost = self.Y*np.log(Z)+(1-self.Y)*np.log(1-Z)
        return cost


    def gradient_desc(self,Z,alpha=0.1,i=10000):
        m=Z.shape[0]
        D=np.zeros((3,1))
        for j in range(i):
            D=((((Z - self.Y).T).dot(self.X)).T)/m
            self.theta = self.theta - D*alpha
        return self.theta
   
   


        
    

    


# In[ ]:


#normalization for x
def normalize(X):
    R=np.std(X,axis=0)
    M=np.mean(X,axis=0)
    X=(X-M)/R
    return X 


def accuracy(Y_pred,Y):
    acc=np.mean((Y_pred==Y))*100
    return acc


def sigmoid(X,theta):
    Z=1/(1+np.exp(X.dot(theta)))
    return Z
    

def predict(X,theta):
    X=np.insert(X,0,1,axis=1)
    Y_pred=sigmoid(X,theta)
    Y_pred=1-(Y_pred//0.500000000001)
    return Y_pred

