#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class K_Means :
    
    def __init__(self,K=2):
        # Takes the value of number of cluster centres K
        self.K = K
        
        
    def fit(self,X):
        # fits the untrained data & calls the kmeans function to analyze the data
        
        self.X=X                                        # dataset provided by user
        self.Cluster_centre = [0]*self.K                # Coordinates of Cluster Centres will be kept in here
        self.distance=np.zeros((X.shape[0],self.K))     # Distance of every point from centres will be stored
        self.m=X.shape[0]                               # no of examples 
        
        self.random_initialization()                    # random initialization 
        self.KMeans()                                   # K means clustering function is called
        
        
    def normalize(self,X):
        # This function normalizes the features
        
        R=np.std(X,axis=0)
        M=np.mean(X,axis=0)
        X=(X-M)/R
        return X
        
        
    def dist(self,C):
        # Calculates the square of distance between all examples of X and the point C
        return np.sum(np.square(self.X - C),axis=1)
    
    
    
    def cost(self,D):
        #Calculate & returns the value of cost function
        return np.sum(D,axis=0)/self.m 
        
        
        
    def random_initialization(self):
        """ This function helps to assign
            ideal value to the cluster centers to
            begin with """
        itr=100                                 #No of iterations
        centers=[None]*itr                      #randomn cluster centres
        J=[None]*itr                            #calculates the value of the cost function
        
        for i in range(itr):
            centers[i]=self.X[np.random.randint(self.m),:]         #Random points from the dataset itself are tested as centres
            D = self.dist(centers[i])                              # Distance of points from the random centers
            J[i] = self.cost(D)                                    #Cost function 
        
        """ Amongst the random cluster centres
            the ones for which the cost function
            gives minimum value are selected """
        for i in range(self.K):
            self.Cluster_centre[i]=centers[np.argsort(J)[i]]       
            
            
            
    def KMeans(self):
        """ Kmeans clustering takes place
            the points are assigned labels wrt their nearness to 
            a cluster centre & the cluster centre is then shifted
            to their means. This is repeated..."""
        
        for i in range(100):   
            for j in range(self.K):
                self.distance[:,j]=self.dist(self.Cluster_centre[j])   # Distance of points from cluster centers
        
            self.labels = np.argmin(self.distance,axis=1)              # Label of nearest cluster center assigned
    
            """ The following loop shifts the cluster 
                centers to the mean of respective 
                labelled data """ 
            for k in range(self.K):                                    
                sum_x1=0                                        
                sum_x2=0
                n=1
                for j in range(self.m):
                    if(self.labels[j]==k):
                        sum_x1=sum_x1+self.X[j][0]
                        sum_x2=sum_x2+self.X[j][1]
                        n=n+1
                self.Cluster_centre[k]=np.array([sum_x1/n,sum_x2/n])
            
        return self.labels

        

