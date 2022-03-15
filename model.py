#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# In[2]:


dataset=pd.read_csv('hiring.csv')
dataset.head()


# In[3]:


def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
X = dataset.iloc[:, :3]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


# In[5]:


regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print(y_pred)


# In[6]:


#save the model in disk


# In[9]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[10]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




