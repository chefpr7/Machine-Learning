#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


class LogR:
    
    
    def fitt(self,Xin,Yin,alpha,i,length):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)
        self.mini_gradient_descent(length,alpha,i)
        
    def sigmoid_func(self):
        Z=1/(1+np.exp(self.X.dot(self.theta)))
        return Z
    
    def sigmoidf(self,X):
        Z=1/(1+np.exp(X.dot(self.theta)))
        return Z
    
    def cost_func(self):
        Z=self.sigmoid_func()
        cost = -1*(self.Y.T.dot(np.log(Z))+(1-self.Y).T.dot(np.log(1-Z)))
        return cost


    def gradient_desc(self,alpha=0.1,i=10000):
        m=self.X.shape[0]
        D=np.zeros((3,1))
        for j in range(i):
            Z=self.sigmoid_func()
            D=((Z - self.Y).T.dot(self.X)).T/m
            self.theta = self.theta + D*alpha
        return self.theta
    
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
                Z=self.sigmoidf(X1)
                D=((((Z - Y1).T).dot(X1)).T)/v
                self.theta = self.theta + D*alpha
        return self.theta
    
    
    
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 


    def accuracy(self,Y_pred,Y):
        acc=np.mean((Y_pred==Y))*100
        return acc
    

    def predict(self,X):
        X=np.insert(X,0,1,axis=1)
        Z=self.sigmoidf(X)
        Y_pred=(Z//0.500000000001)
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
            plt.plot(R[j][0],(self.theta[0][0]+self.theta[1][0]*R[j][0])/(self.theta[2][0]*-1),"gx")
        plt.xlabel("X1-->")
        plt.ylabel("X2-->")

        

# In[ ]:



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #normalization for x
