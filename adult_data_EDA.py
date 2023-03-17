#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
warnings.filterwarnings("ignore")


# In[56]:


data=pd.read_csv('adult_data.csv',header=None)


# In[57]:


data.head()


# In[58]:


column_list=['age',
'workclass',
'fnl_wgt',
'education',
'education_num',
'marital_status',
'occupation',
'relationship',
'race',
'sex',
'capital_gain',
'capital_loss',
'hrs_per_week',
'native_country',
'salary']


# In[59]:


data.columns=column_list


# In[60]:


data.head()


# In[61]:


data.shape


# In[62]:


data.columns


# In[63]:


data.head(50)


# ### Observation
# there are some invalid data ('?') in workclass and occupation features
# 
# and there may be '?' in other features, so 
# 
# Let's look at the unique value and its count for each features

# In[64]:


data[data.duplicated()]


# In[65]:


data[data.duplicated()].shape


# ### Observation
# There are 24 duplicate row in the dataset

# In[66]:


data.info()


# ### Observation
# there are 6 integer datatype and 9 object datatype
# 
# Thus, there are 6 numerical features and 9 categorical features

# In[67]:


data.columns


# ### Checking the unique values of each features

# In[68]:


data['age'].value_counts()


# In[69]:


data['workclass'].value_counts()


# In[70]:


data['fnl_wgt'].value_counts()


# In[71]:


data['education'].value_counts()


# In[72]:


data['education_num'].value_counts()


# In[73]:


data['marital_status'].value_counts()


# In[74]:


data['occupation'].value_counts()


# In[75]:


data['relationship'].value_counts()


# In[76]:


data['race'].value_counts()


# In[77]:


data['sex'].value_counts()


# In[78]:


data['capital_gain'].value_counts()


# In[79]:


data['capital_loss'].value_counts()


# In[80]:


data['hrs_per_week'].value_counts()


# In[81]:


data['native_country'].value_counts()


# In[82]:


data['salary'].value_counts()


# ### Observation
# There is '?' (questionmark) in "workclass", "occupation" and "native_country" features and having 1836, 1843 and 583 no. of values respectively
# 
# So, we can't drop the rows containing '?'
# 
# Thus let's replace '?' with mean, median or mode (which one of them are more suitable for the the dataset)
# 
# Since all of the three features having '?' (questionmark) are categorical features.
# 
# Thus, we will replace '?' (questionmark) with mode (i.e we will the most common value(highest frequent value) in each features).
# 
# In salary feature there are some charector, so we have to cleen it

# In[83]:


#Lets find the frequency for all features


# In[84]:


data.describe(include='all').T


# ### Observation
# Thus from above description table of data, we find that
# 
# Highest freq. in workclass features is "Private"
# 
# Highest freq. in occupation features is "Prof-specialty"
# 
# Highest freq. in native_country features is "United-States"

# ### Let's Replace '?' (questionmark) with mod of each features

# In[85]:


data.mode()


# In[86]:


mode_workclass=data['workclass'].mode().iloc[0]
mode_occupation=data['occupation'].mode().iloc[0]
mode_native_country=data['native_country'].mode().iloc[0]


# In[87]:


data['workclass']=data['workclass'].str.replace('?',mode_workclass)
data['occupation']=data['occupation'].str.replace('?',mode_occupation)
data['native_country']=data['native_country'].str.replace('?',mode_native_country)


# In[88]:


data.head(50)


# In[89]:


data['workclass'].value_counts()


# In[90]:


data['occupation'].value_counts()


# In[91]:


data['native_country'].value_counts()


# ### Cleening the salary features

# In[92]:


data['salary'].unique()


# In[93]:


#Replacing '1' where '>50K' (where salary is greater then 50k)
#and Replacing '0' where '<=50K' (where salary is less then 50k)


# In[97]:


data['salary']=data['salary'].str.replace(' <=50K','0')
data['salary']=data['salary'].str.replace(' >50K','1')


# In[98]:


data['salary'].unique()


# In[99]:


#Converting the object dtype to int
data['salary']=data['salary'].astype(int)


# In[102]:


data.info()


# ### Handeling the missing values

# In[103]:


data.isnull().sum()


# ### Observation
# There is not any missing value in dataset

# ### Checking Duplicate values in dataset

# In[113]:


data[data.duplicated()]


# In[ ]:





# In[ ]:





# ### Finding the categorical and numerical featuress

# In[63]:


data.info()


# ### Observation
# there are 6 integer datatype and 9 object datatype
# 
# Thus, there are 6 numerical features and 9 categorical features

# In[107]:


numerical_fea=[features for features in data.columns if data[features].dtype!='O' ]


# In[108]:


numerical_fea


# In[111]:


data[numerical_fea]


# In[109]:


categorical_fea=[features for features in data.columns if data[features].dtype=='O' ]


# In[110]:


categorical_fea


# In[112]:


data[categorical_fea]


# ### Univariate Analysis of Numerical Features

# In[118]:


plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numerical_fea)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=data[numerical_fea[i]],shade=True, color='b')
    plt.xlabel(numerical_fea[i])
    plt.tight_layout()


# ### Univariate Analysis of Categorical Features

# In[120]:


len(categorical_fea)


# In[127]:


# categorical columns
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.9, y=1.)
category = [ 'Type', 'Content Rating']
for i in range(0, len(categorical_fea)):
    plt.subplot(5, 2, i+1)
    sns.countplot(x=data[categorical_fea[i]],palette="Set2")
    plt.xlabel(categorical_fea[i])
    plt.xticks(rotation=45)
    plt.tight_layout() 
   


# In[ ]:




