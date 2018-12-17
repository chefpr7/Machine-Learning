#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math


# In[ ]:


class LogR:
    
    
    def fitt(self,Xin,Yin,alpha,i,length):
        self.X=Xin  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.M=np.random.rand(2,1)
        self.C=np.ones((Xin.shape[0],1))*random.random()
        #self.gradient_desc(alpha,i)
        self.mini_gradient_descent(length,alpha,i)
        
    def sigmoidf(self,X,C):
        Z=1/(1+np.exp(-(X.dot(self.M)+C)))
        return Z
   
    def cost_func(self):
        Z=self.sigmoidf(self.X,self.C)
        cost = -1*(self.Y.T.dot(np.log(Z))+(1-self.Y).T.dot(np.log(1-Z)))
        return cost

    def gradient_desc(self,alpha=0.1,i=10000):
        m=self.X.shape[0]
        for j in range(i):
            Z=self.sigmoidf(self.X,self.C)
            DM=((Z - self.Y).T.dot(self.X).T)/m
            DC=(Z - self.Y)/m
            self.M = self.M - DM*alpha
            self.C = self.C - DC*alpha
            
    
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
                Z=self.sigmoidf(X1,C1)
                
                DM=((Z - Y1).T.dot(X1).T)/v
                DC=(Z - Y1)/v
                
                self.M = self.M - DM*alpha
                C1 = C1 - DC*alpha
                self.C[u:u+v,:]=C1
                
                if(j==0):
                    print(C1.shape)
    
    
    
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 


    def accuracy(self,Y_pred,Y):
        acc=np.mean(Y_pred==Y)*100
        return acc
    

    def predict(self,X):
        C1=self.C[0:X.shape[0],:]
        Z=self.sigmoidf(X,C1)
        Y_pred=(Z>=0.5)
        return Y_pred


    def plott(self,X,Y):
        plt.figure()
        for i in range(Y.shape[0]):
            if(Y[i][0]==1):
                plt.plot(X[i][0],X[i][1],"bx")
            if(Y[i][0]==0):
                plt.plot(X[i][0],X[i][1],"rx")
        R=np.array([np.arange(-2,2,0.02)]).T
        for j in range(R.shape[0]):
            plt.plot(R[j][0],(self.C[0][0]+self.M[0][0]*R[j][0])/(self.M[1][0]*-1),"gx")
        plt.xlabel("X1-->")
        plt.ylabel("X2-->")

        

#
