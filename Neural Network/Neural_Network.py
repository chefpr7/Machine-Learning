#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import math
 
class NN :
    
    def fit(self,Xin,Yin,size=[25],L=2,alpha=1,itr=500,mini_batch_size=0):
        # This function initializes all the required hyperparameters and parameters & other variables required
        self.X=Xin                         
        self.Y=Yin
        self.L=L+2                          # No of Layers                        
        self.A = [0] * self.L                   # Activation function
        self.A[0]=self.X
        
        self.w = [0] * (self.L-1)               # Weights
        self.b = [0] * (self.L-1)               # bias
        self.m = self.X.shape[0]           # No of training examples
        
        self.d = [0] * self.L                   # For calculating partial derivative
        self.der = [0]*self.L                   # For calculating derivative
        self.Lambda=0.3                    # For regularization 
          
        self.c=np.unique(self.Y).shape[0]  # No of unique values of Y training set 
        self.y = np.zeros((self.m,self.c)) # For coverting y into a column matrix of shape c, e.g. y=3 is taken as [0 0 1 0] 
        
      # Size of layers
        sol=np.zeros((1,L))
        sol+=size
        sol=sol.astype(int)
        sol=np.insert(sol,0,self.X.shape[1])
        sol=np.insert(sol,self.L-1,self.c)
        print(sol)
        
        self.alpha=alpha # Learning rate
        
        if(mini_batch_size==0):
            self.length=Xin.shape[0]//15       # Length of mini batch
            
        self.itr=itr                       # No of iterations
        
        epsilon_init=0.2                 
        
        for i in range(self.m):
            self.y[i][int(self.Y[i][0])]=1                 # y as a vector of 0 and 1 i.e. y = 3 means [0 0 0 1] if y = {0,1,2,3}

        for i in range(self.L-1):                 
            self.w[i] = np.random.rand(sol[i+1],sol[i])* 2 * epsilon_init - epsilon_init   #initializing weights and bias
            self.b[i] = np.random.rand(1,sol[i+1])* 2 * epsilon_init - epsilon_init
        
       # self.gradient_descent()
        self.mini_gradient_descent()
    
    
    def sigmoidf(self,z):
        # Not better than any other sigmoid function out there
        return 1/(1+np.exp(-z))
    
    
    
    
    def fwdx(self):
        #This function optimizes the values of the activation variable through forward propogation
        for i in range(self.L-1):
            self.A[i+1] = self.sigmoidf(np.dot(self.A[i],self.w[i].T)+self.b[i])
        
     
    
    
    def backpropg(self):
        # NO sooner has forward propogation taken place,than this function is called, it calculates derivatives
        # the cost function w.r.t parameters 
        self.d[self.L-1] = self.A[self.L-1] - self.y
        
        for i in range(self.L-2,0,-1):
            self.d[i]=np.dot(self.d[i+1],self.w[i])*self.A[i]*(1-self.A[i])
            
        for i in range(self.L-1):
            
            self.der[i] += np.dot(self.d[i+1].T,self.A[i])/self.m + self.Lambda*self.w[i]/self.m
            
            self.w[i] = self.w[i] - (self.alpha)*self.der[i]
            self.b[i] = self.b[i] - (self.alpha/self.m)*np.sum(self.d[i+1],axis=0)
        
    
    
    
    def mini_back_propg(self,u,v):
        # Works in the same manner as back propogation BUT does so in small batches, when called
        self.d[self.L-1] = self.A[self.L-1][u:u+v,:] - self.y[u:u+v,:]
        
        #calculates partial derivative 
        for i in range(self.L-2,0,-1):
            self.d[i]=np.dot(self.d[i+1],self.w[i])*self.A[i][u:u+v,:]*(1-self.A[i][u:u+v,:])
        
        #calculates actual derivative & optimizes weights and bias
        for i in range(self.L-1):
            self.der[i] += np.dot(self.d[i+1].T,self.A[i][u:u+v,:])/self.m + self.Lambda/self.m* self.w[i]
            self.w[i] = self.w[i] - (self.alpha)*self.der[i]
            self.b[i] = self.b[i] - (self.alpha/self.m)*np.sum(self.d[i+1],axis=0)
        self.fwdx()
       
    
    
    def gradient_descent(self):
        # Derivatives being calculated,this function directs the weights into a journey towards the minima 
        for j in range(self.itr):
            self.fwdx()
            self.backpropg()
            #print(self.A)
               
    
    def mini_gradient_descent(self):
        # Again,functions like gradient descent BUT does so in small batches, hence named mini batch gradient descent
        
        r=math.ceil(self.m/self.length)    # Well,no of times the loop should run to cover all mini batches           
        
        for j in range(self.itr):
            v=self.length
            self.fwdx()
            for k in range(r):
                u=k*self.length
                if(u+v>self.m):
                    v=self.m-u    
                self.mini_back_propg(u,v)
            
    
    def pred(self,X):
        # Once weights are optimized, pred() predicts the values on the testing data provided by user
        self.A[0]=X
        self.fwdx()
        #print(self.A)
        y_=self.A[self.L-1]>=0.5
        for i in range(X.shape[0]):
            y_pred = np.argmax(y_,1) 
        return y_pred
    
     
        
    def accuracy(self,y_test,y_pred):
        # Calculates the accuracy of the prediction , always satisfying when it matches sklearn
        return np.mean((y_pred==y_test.flatten())*100)
    
    
    
    def normalize(self,X):
        # Normalizes the features 
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X 
    
    
    
        
        

