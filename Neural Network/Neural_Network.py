#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import math
 
class NN :
    
    def fitt(self,Xin,Yin,sol,L=4,alpha=1,itr=500):
        self.X=Xin
        self.Y=Yin
        self.L=L                           # No of layers
        self.A = [0] * L                   #Activation function
        self.A[0]=self.X
        
        self.w = [0] * (L-1)               # Weights
                    
        self.b = [0] * (L-1)               # bias
        self.m = self.X.shape[0]
        self.d = [0] * L
       
        self.der = [0]*L
        self.Lambda=0.1
        self.c=np.unique(self.Y).shape[0]
        self.y = np.zeros((self.m,self.c))
        
        self.alpha=alpha
        self.length=Xin.shape[0]//15
        self.itr=itr
        
        epsilon_init=0.2
        
        for i in range(self.m):
            self.y[i][int(self.Y[i][0])]=1                 # y as a vector of 0 and 1 i.e. y = 3 means [0 0 0 1] if y = {0,1,2,3}

        for i in range(L-1):                 
            self.w[i] = np.random.rand(sol[i+1],sol[i])* 2 * epsilon_init - epsilon_init   #initializing weights and bias
            self.b[i] = np.random.rand(1,sol[i+1])* 2 * epsilon_init - epsilon_init
        
        #self.gradient_descent()
        self.mini_gradient_descent()
    
    
    def sigmoidf(self,z):
        return 1/(1+np.exp(-z))
    
    
    
    
    def fwdx(self):
        for i in range(self.L-1):
            self.A[i+1] = self.sigmoidf(np.dot(self.A[i],self.w[i].T)+self.b[i])

     
    
    
    def backpropg(self):
        self.d[self.L-1] = self.A[self.L-1] - self.y
        for i in range(self.L-2,0,-1):
            self.d[i]=np.dot(self.d[i+1],self.w[i])*self.A[i]*(1-self.A[i])
        for i in range(self.L-1):
            self.der[i] += np.dot(self.d[i+1].T,self.A[i])/self.m + self.Lambda/self.m* self.w[i]
            
            self.w[i] = self.w[i] - (self.alpha)*self.der[i]
            self.b[i] = self.b[i] - (self.alpha/self.m)*np.sum(self.d[i+1],axis=0)
        
    
    
    
    def mini_back_propg(self,u,v):
        self.d[self.L-1] = self.A[self.L-1][u:u+v,:] - self.y[u:u+v,:]
        for i in range(self.L-2,0,-1):
            self.d[i]=np.dot(self.d[i+1],self.w[i])*self.A[i][u:u+v,:]*(1-self.A[i][u:u+v,:])
        for i in range(self.L-1):
            self.der[i] += np.dot(self.d[i+1].T,self.A[i][u:u+v,:])/self.m + self.Lambda/self.m* self.w[i]
            self.w[i] = self.w[i] - (self.alpha)*self.der[i]
            self.b[i] = self.b[i] - (self.alpha/self.m)*np.sum(self.d[i+1],axis=0)
        self.fwdx()
       
    
    
    def gradient_descent(self):
        for j in range(self.itr):
            self.fwdx()
            self.backpropg()
               
    
    def mini_gradient_descent(self):
  
        r=math.ceil(self.m/self.length)
        
        for j in range(self.itr):
            v=self.length
            self.fwdx()
            for k in range(r):
                u=k*self.length
                if(u+v>self.m):
                    v=self.m-u    
                self.mini_back_propg(u,v)
            
    
    def pred(self,X):
        self.A[0]=X
        self.fwdx()
        y_=self.A[self.L-1]>=0.5
        for i in range(X.shape[0]):
            y_pred = np.argmax(y_,1) 
        return y_pred
    
     
    def accuracy(self,y_test,y_pred):
        return np.mean((y_pred==y_test.flatten())*100)
    
    
    
    def normalize(self,X):
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 
    
    
    
        
        

