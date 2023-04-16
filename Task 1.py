#!/usr/bin/env python
# coding: utf-8

# # TASK 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('http://bit.ly/w-data')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df.corr()


# In[7]:


df.boxplot(figsize=(10,6))
plt.show()


# In[8]:


df.sort_values('Hours').plot(x='Hours',y='Scores',kind='bar')
plt.show()


# In[9]:


from sklearn.model_selection import train_test_split as slt
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[10]:


y=df['Scores']
x=df[['Hours']]


# In[11]:


x_train,x_test,y_train,y_test=slt(x,y,test_size=0.25,random_state=2529)


# In[12]:


lr.fit(x_train,y_train)


# In[13]:


y_pred=lr.predict(x_test)


# In[14]:


plt.figure(figsize=(8,8))
plt.scatter(x,y)
plt.plot(x,lr.predict(x),color='red')
plt.show()


# In[15]:


hours=[[7.50]]
pred_of_input=lr.predict(hours)
print("HOURS STUDIED ", hours[0][0])
print("PREDICTED SCORE ",pred_of_input[0])


# In[16]:


from sklearn.metrics import mean_absolute_percentage_error as mape
mape(y_test,y_pred)


# In[17]:


from sklearn.metrics import mean_squared_error as mse
import math
math.sqrt(mse(y_test,y_pred))


# In[18]:


from sklearn.metrics import mean_absolute_error as mae
print("MEAN ASBOLUTE ERROR ", mae(y_test,y_pred))

