#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class D_TREE :
    
    def fit(self,Xin,Yin):
        #fitting the values
        self.X=Xin
        self.Y=Yin
        self.my_tree=self.tree(Xin)
    
    
    
    def label_count(self,t):
        #count the unique labels
        count = {}
        for i in range(len(t)):
            lbl = t[i][-1]
            if lbl not in count:
                count[lbl] = 0
            count[lbl]+=1
        return count

    
    
    
    class Question :
        #stores the question and matches the question 
        def __init__(self,col,value):
            self.col = col
            self.question = value
        
        
        def is_digit_or_char(self,n):
            return isinstance(n,int) or isinstance(n,float)
    
        def check(self,row):
            value=row[self.col]
            if(self.is_digit_or_char(self.question)):
                return value>=self.question
            else :
                return value==self.question
        
        
   

    def gini(self,t):
        #Calculates the gini score
        label = np.unique(t)
        impurity = 1
    
        for i in range(len(label)):
            impurity -= (np.sum(t[:,-1]==label[i])/t.shape[0])**2
    
        return impurity

    
    

    def information_gain(self,l,r,current_uncertainity):
        #Information gain is calculated
        p = len (l) / float ( len(l) + len(r) )
        return current_uncertainity - p*self.gini(l) - (1-p)*self.gini(r)
    
    
    
    
    def best_split(self,t):
        #Selects the best question and split based on the gini score
        maxm=0
        best_question = None
        tr_row=[]
        fl_row=[]
            
        for i in range(t.shape[1]-1):
            y=np.unique(t[:,i])
            m=y.shape[0]
            for j in range(m):
                question = self.Question(i,y[j])
                tr_row , fl_row = self.split(t,question)
                if(len(fl_row)==0 or len(tr_row)==0):
                    continue
                
                info_gain= self.information_gain(tr_row,fl_row,self.gini(t))
                
                if(info_gain>maxm):
                    maxm = info_gain
                    best_question = question
                
        return maxm,best_question

    
    
   
    def split(self,t,question)
    #Splits the dataset based on the best question
        tr_row=[]
        fl_row=[]
        for k in range(t.shape[0]):
            if question.check(t[k]):
                tr_row=np.append(tr_row,t[k])
            else:
                fl_row=np.append(fl_row,t[k])
                    
        tr_row = np.reshape(tr_row,(len(tr_row)//t.shape[1],t.shape[1]))
        fl_row = np.reshape(fl_row,(len(fl_row)//t.shape[1],t.shape[1]))
    
        return tr_row,fl_row
    
    
    
    
    
    class Decision_Node:
        #Stores the different question,true branch and false branch for all parts of the tree
        def __init__(self,question,true_branch,false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch


            
            
    class Leaf:
        #the terminal of a tree is the leaf
        def __init__(self,t):
            self.predictions = D_TREE().label_count(t)    

            
            
            

    def tree(self,t):
        #This function constructs the tree
        gain,question = self.best_split(t)
        if(gain==0):
            return self.Leaf(t)
        true_rows , false_rows = self.split(t,question)
        true_node = self.tree(true_rows)
        false_node= self.tree(false_rows)
        return self.Decision_Node(question,true_node,false_node)
    
    
    
    
    
    def check_testing_data(self,test,node):
        #checks the testing data by recursively calling itself
        if isinstance(node,self.Leaf):
            return node.predictions
        if(node.question.check(test)):
            return self.check_testing_data(test,node.true_branch)
        else:
            return self.check_testing_data(test,node.false_branch)
        
    
    
    
    
    def print_leaf(self,LEAF):
        #prints a leaf
        p={}
        for i in LEAF.keys():
            p[i] = str(100*LEAF[i]/float(sum(LEAF.values()))) + "%"
        
        print(p)
        
    
    
    
    
    def pred(self,X_test):
        #predicts values for test data
        y_pred=[0]*X_test.shape[0]
        for i in range(X_test.shape[0]):
            r= self.check_testing_data(X_test[i],self.my_tree)
            y_pred[i] = max(r.keys(), key=(lambda k: r[k]))
        return y_pred
    
    
    
    
    
    def accuracy(self,y_test,y_pred):
        #Calculate the accuracy of the model
        return np.mean(y_test==y_pred)*100
    
    
    

