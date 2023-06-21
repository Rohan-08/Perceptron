# -*- coding: utf-8 -*-
"""
@author: Rohan Chhabra
"""

"""
Question 2:- Implement a binary perceptron. The implementation should be consistent with the pseudo code in the answer to Question 1.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

train_df = pd.read_csv('train.data', names=['f1','f2', 'f3', 'f4', 'tgt'])
test_df = pd.read_csv('test.data', names=['f1','f2', 'f3', 'f4', 'tgt'])


train_df.head()
test_df.head()

X_train = train_df.drop(['tgt'], axis=1)
y_train = train_df['tgt']

MaxIter=20

def train_perceptron(X_train, y_train, MaxIter):
    
    # Creating a list for weights. number of columns is the number of weights required
    w= [0.0 for x in range(len(X_train.columns))]
    # Initialising bias
    b=0
    
    # Running the code for total epochs
    for _ in range(MaxIter):     
        # Iterating through X to get the acitvation scores
        for i in range(len(X_train)):
            a=0
            for j in range(len(X_train.columns)):
                a+= w[j]* X_train.iloc[i,j] + b
            
            # If the product of y and actiavtion is negative, then we follow the steps below
            if y_train.iloc[i]*a <=0:
                for j in range(len(X_train.columns)):
                    w[j]= w[j] + y_train.iloc[i]*X_train.iloc[i,j]
                    b+=y_train.iloc[i]
    return w,b


def test_perceptron(X_test, w, b):
    # Creating a list to save the predictions
    pred_y=[0.0 for x in range(len(X_test))]
    
    # Iterating through the X set and calculating the activation score for each row
    for i in range(len(X_test)):
        a=0
        for j in range(len(X_test.columns)):
            a+= w[j]*X_test.iloc[i,j] + b
        
        # Checking the sign of actiavtion score and setting the value. if a<=0, then we take as -1, else 1
        if a>0:
            pred_y[i]=1
        else:
            pred_y[i]=-1
        
    return pd.Series(pred_y)


def accuracy(y_pred, y_test):
    # This function takes in the actual y and the predicted y and gives the accuracy or the number of matched rows
    acc=0.0
    for i in range(len(y_pred)):
        if y_pred.iloc[i]==y_test.iloc[i]:
            acc+=1
    
    accuracy_= acc/len(y_pred)
    return f"{accuracy_*100:.3f}%"
        
"""
Use the binary perceptron to train classifiers to discriminate between
• class 1 and class 2,
• class 2 and class 3, and
• class 1 and class 3.
Report the train and test classification accuracies for each of the three classifiers after
training for 20 iterations. Which pair of classes is most difficult to separate?
"""        



#%%
"""Class-1 vs Class-2"""

#Class 1 and Class 2: Filtering out Class 3 variables from the data
train_df_1 = train_df[train_df['tgt']!='class-3']

#Replacing the target variable as -1 and +1
train_df_1.loc[train_df_1['tgt'] == 'class-2', 'tgt_enc'] = -1
train_df_1.loc[train_df_1['tgt'] != 'class-2', 'tgt_enc'] = 1

# Creating training dataset
X_train_1 = train_df_1.drop(['tgt','tgt_enc'], axis=1)
y_train_1 = train_df_1['tgt_enc']

# Fitting the model on training data
w,b = train_perceptron(X_train_1, y_train_1, MaxIter)

# Running the model on training data
pred_train_y_1 = test_perceptron(X_train_1,w,b)

# Checking accuracy on the training model
print("\nThe accuracy of the Class 1 vs Class 2 training model is:",accuracy(pred_train_y_1, y_train_1))

# Running the model on test data
test_df_1 = test_df[test_df['tgt']!='class-3']

#Replacing the target variable as -1 and +1
test_df_1.loc[test_df_1['tgt'] == 'class-2', 'tgt_enc'] = -1
test_df_1.loc[test_df_1['tgt'] != 'class-2', 'tgt_enc'] = 1

X_test_1 = test_df_1.drop(['tgt','tgt_enc'], axis=1)
y_test_1 = test_df_1['tgt_enc']

# Running the model on test data
pred_y_1 = test_perceptron(X_test_1,w,b)

# Calculating Accuracy
print("The accuracy of the Class 1 vs Class 2 test model is:",accuracy(pred_y_1, y_test_1))


#%%
"""#Class 2 and Class 3: Filtering out Class 1 variables from the data"""

train_df_2 = train_df[train_df['tgt']!='class-1']

#Replacing the target variable as -1 and +1
train_df_2.loc[train_df_2['tgt'] == 'class-2', 'tgt_enc'] = -1
train_df_2.loc[train_df_2['tgt'] != 'class-2', 'tgt_enc'] = 1

X_train_2 = train_df_2.drop(['tgt','tgt_enc'], axis=1)
y_train_2 = train_df_2['tgt_enc']

# Fitting the model on training data
w,b = train_perceptron(X_train_2, y_train_2, MaxIter)

# Running the model on training data
pred_train_y_2 = test_perceptron(X_train_2,w,b)

# Checking accuracy on the training model
print("\nThe accuracy of the Class 2 vs Class 3 training model is:",accuracy(pred_train_y_2, y_train_2))

# Running the model on test data

test_df_2 = test_df[test_df['tgt']!='class-1']  

#Replacing the target variable as -1 and +1

test_df_2.loc[test_df_2['tgt'] == 'class-2', 'tgt_enc'] = -1
test_df_2.loc[test_df_2['tgt'] != 'class-2', 'tgt_enc'] = 1

X_test_2 = test_df_2.drop(['tgt','tgt_enc'], axis=1)
y_test_2 = test_df_2['tgt_enc']

# Running the model on test data
pred_y_2 = test_perceptron(X_test_2,w,b)

# Calculating Accuracy
print("The accuracy of the Class 2 vs Class 3 testing model is:",accuracy(pred_y_2, y_test_2))




#%%
"""#Class 1 and Class 3: Filtering out Class 2 variables from the data"""

train_df_3 = train_df[train_df['tgt']!='class-2']

#Replacing the target variable as -1 and +1

train_df_3.loc[train_df_3['tgt'] == 'class-1', 'tgt_enc'] = -1
train_df_3.loc[train_df_3['tgt'] != 'class-1', 'tgt_enc'] = 1

X_train_3 = train_df_3.drop(['tgt','tgt_enc'], axis=1)
y_train_3 = train_df_3['tgt_enc']

# Fitting the model on training data
w,b= train_perceptron(X_train_3, y_train_3, MaxIter)

# Running the model on training data
pred_train_y_3 = test_perceptron(X_train_3,w,b)

# Checking accuracy on the training model
print("\nThe accuracy of the Class 1 vs Class 3 training model is:",accuracy(pred_train_y_3, y_train_3))

# Running the model on test data
test_df_3 = test_df[test_df['tgt']!='class-2']

#Replacing the target variable as -1 and +1

test_df_3.loc[test_df_3['tgt'] == 'class-1', 'tgt_enc'] = -1
test_df_3.loc[test_df_3['tgt'] != 'class-1', 'tgt_enc'] = 1

X_test_3 = test_df_3.drop(['tgt','tgt_enc'], axis=1)
y_test_3 = test_df_3['tgt_enc']

# Running the model on test data
pred_y_3 = test_perceptron(X_test_3,w,b)

# Calculating Accuracy
print("The accuracy of the Class 1 vs Class 3 testing model is:",accuracy(pred_y_3, y_test_3))



#%%
"""
Q4.
One vs rest: 
"""

def test_perceptron_ovr(X_test, w1, b1, w2, b2, w3, b3):
    pred_y=[0.0 for x in range(len(X_test))]
    
    # It is the same as test_perceptron. The only difference is we run three models and compare their activation scores.
    # The model with the highest score is chosen as prediction for that row
    for i in range(len(X_test)):
        
        a_1, a_2, a_3= 0,0,0
        for j in range(len(X_test.columns)):
            a_1+= w1[j]*X_test.iloc[i,j] + b1
            a_2+= w2[j]*X_test.iloc[i,j] + b2
            a_3+= w3[j]*X_test.iloc[i,j] + b3
        
        pred_y[i]= 1 if (a_1>a_2) and (a_1>a_3) else 2 if (a_2>a_1) and (a_2>a_3) else 3
        
    return pd.Series(pred_y)


#%%
"""
One vs Rest
"""
train_df_4 = train_df.copy(deep=True)
train_df_5 = train_df.copy(deep=True)
train_df_6 = train_df.copy(deep=True)

#Keeping class in interest as 1 and others as -1. Making 3 models for three different classes.
# Each class will become 1 once and -1 the other 2 times.

train_df_4.loc[train_df_4['tgt'] == 'class-1', 'tgt_enc'] = 1
train_df_4.loc[train_df_4['tgt'] != 'class-1', 'tgt_enc'] = -1
X_train_4 = train_df_4.drop(['tgt','tgt_enc'], axis=1)
y_train_4 = train_df_4['tgt_enc']


train_df_5.loc[train_df_5['tgt'] == 'class-2', 'tgt_enc'] = 1
train_df_5.loc[train_df_5['tgt'] != 'class-2', 'tgt_enc'] = -1
X_train_5 = train_df_5.drop(['tgt','tgt_enc'], axis=1)
y_train_5 = train_df_5['tgt_enc']


train_df_6.loc[train_df_6['tgt'] == 'class-3', 'tgt_enc'] = 1
train_df_6.loc[train_df_6['tgt'] != 'class-3', 'tgt_enc'] = -1
X_train_6 = train_df_6.drop(['tgt','tgt_enc'], axis=1)
y_train_6 = train_df_6['tgt_enc']

# Creating the y_train dataset. Just picking the class number from below eg: (1,2,3)
y_train_ovr=train_df.copy(deep=True)
y_train_ovr['enc_tgt'] = y_train_ovr['tgt'].apply(lambda x: int(x.split('-')[1]))

# Fitting the model on training data
w1,b1 = train_perceptron(X_train_4, y_train_4, MaxIter)
w2,b2 = train_perceptron(X_train_5, y_train_5, MaxIter)
w3,b3 = train_perceptron(X_train_6, y_train_6, MaxIter)

# Running the model on training data. As X_train_4, X_train_5, X_train_6 are exactly the same, we can use any to get accuracy on the training data
pred_train_y_ovr = test_perceptron_ovr(X_train_6, w1,b1, w2, b2, w3, b3)

# Checking accuracy on the training model
print("\nThe accuracy of the one vs rest training model is:",accuracy(pred_train_y_ovr, y_train_ovr['enc_tgt']))

# Running the model on test data
test_df_4 = test_df.copy(deep=True)

# Picking the class number from below eg: (1,2,3)
X_test_ovr = test_df_4.drop(['tgt'], axis=1)
y_test_ovr = test_df_4['tgt'].apply(lambda x: int(x.split('-')[1]))

# Running the model on test data
pred_y_ovr = test_perceptron_ovr(X_test_ovr, w1, b1, w2, b2, w3, b3)

# Calculating Accuracy
print("The accuracy of the one vs rest testing model is:",accuracy(pred_y_ovr, y_test_ovr))


#%%

def train_perceptron_loss(X_train, y_train, lamb, MaxIter):
    """ This is the same function as train_perceptron
    The difference is we penalise the weights with a lambda coefficient whenever a wrong classification is done.
    Also, if the classification is done correctly, we change the weights as opposed to the normal classification
    """
    w= [0.0 for x in range(len(X_train.columns))]
    b=0  
    for _ in range(MaxIter):
        
        for i in range(len(X_train)):
            a=0
            for j in range(len(X_train.columns)):
                a+= w[j]* X_train.iloc[i,j] + b
            
            if y_train.iloc[i]*a <=0:
                for j in range(len(X_train.columns)):
                    w[j]= (1-2*lamb)*w[j] + y_train.iloc[i]*X_train.iloc[i,j]
                    b+=y_train.iloc[i]
            else:
                for j in range(len(X_train.columns)):
                    w[j]= (1-2*lamb)*w[j]
    return w,b
            

#%%

"""
One vs Rest with L2
"""

lambda_list=[0.01, 0.1, 1.0, 10.0, 100.0]
# Using the same datasets created above for One vs Rest and putting them in the new training function

# Fitting the model on training data

# Creating dictinaries to save the accuracies of the output
training_acc_dict={}
test_acc_dict={}

for lamb in lambda_list:
    # 3 models for 3 different classes
    w1,b1 = train_perceptron_loss(X_train_4, y_train_4, lamb, MaxIter)
    w2,b2 = train_perceptron_loss(X_train_5, y_train_5, lamb, MaxIter)
    w3,b3 = train_perceptron_loss(X_train_6, y_train_6, lamb, MaxIter)
    
    # Testing the model on training data
    pred_train_y_loss = test_perceptron_ovr(X_test_ovr, w1, b1, w2, b2, w3, b3)
    # Calculating training accuracy
    training_acc_dict[lamb] = accuracy(pred_train_y_loss, y_train_ovr['enc_tgt'])
    
    # Running on test data
    pred_y_loss = test_perceptron_ovr(X_test_ovr, w1, b1, w2, b2, w3, b3)
    #Calculating testing accuracy 
    test_acc_dict[lamb] = accuracy(pred_y_loss, y_test_ovr)

print("\nTraining Data Accuracy: ", training_acc_dict)
print("Test Data Accuracy: ", test_acc_dict)


