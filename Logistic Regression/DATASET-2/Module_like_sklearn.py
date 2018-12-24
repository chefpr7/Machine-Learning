#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


class LogR:
    
    
    def fit(self,Xin,Yin,alpha=0.1,i=10000,length=0):
        self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
        self.Y=Yin
        self.theta=np.random.rand(3,1)
        
        if(length==0):
            length=self.X.shape[0]//10   # Initializes the size of the mini batches if not provided by user
        
        self.alpha=alpha
        self.length=length
        self.i=i
        
        #self.gradient_desc()
        self.mini_gradient_descent()
    
    def sigmoidf(self,X):
    #Calculates the value of the sigmoid function for a matrix
        Z=X.dot(self.theta)
        return 1/(1+np.exp(-Z))
        
    
    def cost_func(self):
    #Calculates the value of Cost function
        Z=self.sigmoidf(self.X)
        cost = -1*(self.Y.T.dot(np.log(Z))+(1-self.Y).T.dot(np.log(1-Z)))
        return cost

    def gradient_desc(self):
    # Carries gradient descent 
        m=self.X.shape[0]
        D=np.zeros((3,1))
        for j in range(self.i):
            Z=self.sigmoidf(self.X)
            D=((Z - self.Y).T.dot(self.X)).T/m        
            self.theta = self.theta - D*self.alpha
        
    
    def mini_gradient_descent(self):
    # Carries mini batch gradient descent
        m=self.X.shape[0]
        r=math.ceil(m/self.length)
        
        for j in range(self.i):
            v=self.length
            for k in range(r):
                u=k*self.length
                if(u+v>m):
                    v=m-u           # to avoid problem of overflowing
                X1=self.X[u:u+v,:]
                Y1=self.Y[u:u+v,:]
                Z=self.sigmoidf(X1)
                D=((((Z - Y1).T).dot(X1)).T)/v       #derivative of cost function wrt theta
                self.theta = self.theta - D*self.alpha
        
    
    def normalize(self,X):
    # Normalizes the features of X
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 


    def accuracy(self,Y_pred,Y):
    # Calculates the accuracy of the model,always satisfying if it matches sklearn
        acc=np.mean((Y_pred==Y))*100
        return acc
    

    def predict(self,X):
    # Predicts the value of Y, based on test data
        X=np.insert(X,0,1,axis=1)
        Z=self.sigmoidf(X)
        Y_pred=(Z>=0.5)
        return Y_pred


    def plott(self,X,Y):
    # Used to plot the decision boundaries
        plt.figure()
        for i in range(Y.shape[0]):
            if(Y[i][0]==1):
                plt.plot(X[i][0],X[i][1],"bx")              #plots the points with Y=1
            if(Y[i][0]==0):
                plt.plot(X[i][0],X[i][1],"rx")              #plots the points with Y=0
                
        R=np.arange(-2,2,0.02)
        m=R.shape[0]
        X_plot=[None]*m                                     #to store the points X of the decision boundary
        Y_plot=[None]*m                                     #to store the points Y of the decision boundary
        
        #Inside this values, X_plot & Y_plot are given values
        for j in range(m):
            X_plot[j]=R[j]                                  
            Y_plot[j]=-(self.theta[0][0]+self.theta[1][0]*R[j])/self.theta[2][0]
        
        plt.plot(X_plot,Y_plot,"g")
        plt.xlabel("X1-->")
        plt.ylabel("X2-->")
        plt.title("Decision Boundary")
        
