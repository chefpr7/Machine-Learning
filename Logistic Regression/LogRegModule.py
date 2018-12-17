#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


class LogR:
    def __init__(self,Xin,Yin):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)
        
    def sigmoid_func(self):
        Z=self.X.dot(self.theta)
        return 1/(1+np.exp(-Z))
    
    
    def cost_func(self):
        Z=self.sigmoid_func()
        cost = -1*(self.Y.T.dot(np.log(Z))+(1-self.Y).T.dot(np.log(1-Z)))
        return cost


    def gradient_desc(self,Z,alpha=0.1,i=10000):
        m=self.X.shape[0]
        D=np.zeros((3,1))
        for j in range(i):
            Z=self.sigmoid_func()
            D=((Z - self.Y).T.dot(self.X)).T/m
            self.theta = self.theta - D*alpha
        return self.theta
    
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
                Z=1/(1+np.exp(X1.dot(self.theta)))
                D=((((Z - Y1).T).dot(X1)).T)/v
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


def sigmoidf(X,theta):
    Z=1/(1+np.exp(-X.dot(theta)))
    return Z
    

def predict(X,theta):
    X=np.insert(X,0,1,axis=1)
    Z=sigmoidf(X,theta)
    #print(Z)
    Y_pred=(Z>=0.5)
    return Y_pred


def plott(X,Y,theta):
    plt.figure()
    for i in range(Y.shape[0]):
        if(Y[i][0]==1):
            plt.plot(X[i][0],X[i][1],"bx")
        if(Y[i][0]==0):
            plt.plot(X[i][0],X[i][1],"rx")
    R=np.array([np.arange(-2,2,0.02)]).T
    for j in range(R.shape[0]):
        plt.plot(R[j][0],(theta[0][0]+theta[1][0]*R[j][0])/(theta[2][0]*-1),"gx")
