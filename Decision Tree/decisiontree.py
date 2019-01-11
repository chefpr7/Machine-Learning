#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class D_TREE :
    
    def fit(self,Xin):
        #fitting the values
        self.X=Xin                            #training_dataset_
        self.my_tree=self.tree(Xin)           #calls tree() function to create a tree based on the dataset provided
    
    
    
    def label_count(self,t):
        #count the unique labels
        count = {}                           #a dictionary that will store the no of times every label has occurred
        for i in range(len(t)):
            lbl = t[i][-1]                   #The last field or column in t actually contains the labels 
            if lbl not in count:
                count[lbl] = 0               #If the label is not present previously,initialize it with zero
            count[lbl]+=1                    #Everytime a particular label is encountered its count is increased by 1           
        return count

    
    
    
    class Question :
        #stores the question and matches the question 
        def __init__(self,col,value):
            self.col = col                  #The column to which the question belongs to
            self.question = value           #the particualr cell in the column which is treated as question
        
        
        def is_digit_or_char(self,n):
            #checks whether a particular value is a number or not
            return isinstance(n,int) or isinstance(n,float)
    
        def check(self,row):
            value=row[self.col]              #the value to be tested with the question
            if(self.is_digit_or_char(self.question)):
                return value>=self.question  #if the value is numeric in nature check whether it is greater or equal to question
            else :
                return value==self.question  #if the value is a character or string check whether it is equal to the question or not
         
        
   

    def gini(self,t):
        #Calculates the gini score
        label = np.unique(t)                #No of unique labels
        impurity = 1
    
        for i in range(len(label)):
            impurity -= (np.sum(t[:,-1]==label[i])/t.shape[0])**2    #formula for calculating impurity based on probability
    
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
            y=np.unique(t[:,i])                         #no of unique labels in a particular column
            m=y.shape[0]                                #no of examples
            for j in range(m):
                question = self.Question(i,y[j])        #each unique label is considered a question one at a time
                tr_row , fl_row = self.split(t,question)#splits the rows based on the question
                if(len(fl_row)==0 or len(tr_row)==0):
                    continue                            #if any of the branch has zero rows,the question is skipped
                
                info_gain= self.information_gain(tr_row,fl_row,self.gini(t))  #information gain is calculated
                
                if(info_gain>maxm):
                    """best question
                       with maximum informaion
                       gain is selected"""
                    maxm = info_gain                 
                    best_question = question
                
        return maxm,best_question

    
    
   
    def split(self,t,question)
    #Splits the dataset based on the best question
        tr_row=[]       
        fl_row=[]
        for k in range(t.shape[0]):
            """checks every row of the dataset 
               with the queston & if it matches,
               it is appended to the true rows
               else to the false rows"""
            if question.check(t[k]):
                tr_row=np.append(tr_row,t[k])   
            else:
                fl_row=np.append(fl_row,t[k])
                    
        tr_row = np.reshape(tr_row,(len(tr_row)//t.shape[1],t.shape[1]))   #just reshapes the one-d matrix into a readable 2d matrix
        fl_row = np.reshape(fl_row,(len(fl_row)//t.shape[1],t.shape[1]))   #just reshapes the one-d matrix into a readable 2d matrix
    
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
        """the most important part of the entire algorithm
        this is where the tree is constructed from the root 
        to the leaves"""
        gain,question = self.best_split(t)                #best question with maximum gain is selected
        if(gain==0):
            return self.Leaf(t)                           #no gain indicates that leaf is reached
        
        """if the control has reached this far,it means
        there is useful gain and teh datset can be subdivided
        or branched into true rows and false rows"""
        true_rows , false_rows = self.split(t,question)    
        true_node = self.tree(true_rows)                  #A recursion is carried out till all the true rows are found out
        false_node= self.tree(false_rows)                 #after true rows,the false rows are assigned to the node in a reverse fashion
                                                            
        return self.Decision_Node(question,true_node,false_node)  
    
    
    
    
    
    def check_testing_data(self,test,node):
        #checks the testing data by recursively calling itself
        if isinstance(node,self.Leaf):
            return node.predictions        #when the leaf is reached prediction is made
        
        """a row is made to travel in the tree,till it reaches a leaf,
           it is checked with all decision nodes, and accordingly
           it travels along true branch or false branch,till
           it reaches a leaf"""
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
            """when a row reaches a particular leaf
               it is assigned the label which
               appears maximum in the leaf"""
            r= self.check_testing_data(X_test[i],self.my_tree)      #deals with one row at a time
            y_pred[i] = max(r.keys(), key=(lambda k: r[k]))         
        return y_pred
    
    
    
    
    
    def accuracy(self,y_test,y_pred):
        #Calculate the accuracy of the model
        return np.mean(y_test==y_pred)*100
    
    
    

