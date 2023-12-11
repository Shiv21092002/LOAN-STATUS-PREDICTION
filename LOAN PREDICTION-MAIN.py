#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[3]:


shiv=pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")


# In[4]:


shiv.head()


# In[5]:


type(shiv)


# In[6]:


shiv.shape


# In[7]:


shiv.describe()


# In[8]:


shiv.isnull().sum()


# In[9]:


shiv.dropna(inplace=True)


# In[10]:


shiv.isnull().sum()


# In[11]:


shiv.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)


# In[12]:


shiv.head()


# In[13]:


shiv["Dependents"].value_counts()


# In[14]:


shiv.replace(to_replace="3+",value=4,inplace=True)


# In[15]:


shiv["Dependents"].value_counts()


# In[16]:


sns.countplot(x="Education",hue="Loan_Status",data=shiv)


# In[17]:


sns.countplot(x="Married",hue="Loan_Status",data=shiv)


# In[18]:


sns.countplot(x="Gender",hue="Loan_Status",data=shiv)


# In[19]:


shiv.replace({"Married":{"Yes":0,"No":1},"Gender":{"Male":0,"Female":"1"},"Self_Employed":{"No":0,"Yes":1},
            "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},"Education":{"Not Graduate":0,"Graduate":1}},inplace =True)


# In[20]:


shiv.head()


# In[21]:


x=shiv.drop(columns=["Loan_ID","Loan_Status"],axis=1)
y=shiv["Loan_Status"]


# In[22]:


print(x)


# In[23]:


print(y)


# In[24]:


scaler=StandardScaler()


# In[25]:


scaler.fit(x)


# In[26]:


standerized_data=scaler.transform(x)


# In[27]:


print(standerized_data)


# In[28]:


x=standerized_data
print(x)


# In[29]:


y=shiv["Loan_Status"]
print(y)


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)


# In[31]:


print(x.shape,x_train.shape,x_test.shape)


# In[32]:


classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)


# In[33]:


x_train_prediction=classifier.predict(x_train)


# In[34]:


training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[35]:


print("Accuracy on training data:",training_data_accuracy)


# In[36]:


x_test_prediction=classifier.predict(x_test)


# In[37]:


test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[38]:


print("Accuracy on test data:",test_data_accuracy)


# Making a predictive  system

# In[46]:


input_data=(0,0,2,1,0,4006,1526,168,360,1,2)
a=np.asarray(input_data)
b=a.reshape(1,-1)
std_data=scaler.transform(b)
print(std_data)
prediction=classifier.predict(std_data)
if prediction == 1:
    print('Loan is approved')
else:
    print('Loan is not approved')


# In[47]:


input_data=(1,1,0,1,0,3510,0,76,360,0,2)
a=np.asarray(input_data)
b=a.reshape(1,-1)
std_data=scaler.transform(b)
print(std_data)
prediction=classifier.predict(std_data)
if prediction == 1:
    print('Loan is approved')
else:
    print('Loan is not approved')


# In[ ]:





# In[ ]:




