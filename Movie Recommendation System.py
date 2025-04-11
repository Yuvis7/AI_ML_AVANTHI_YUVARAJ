#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# In[2]:


movies=pd.read_csv('movies.zip')
movies.head(10)


# In[3]:


movies.describe()


# In[4]:


movies.info()


# In[5]:


movies.isnull().sum()


# In[6]:


movies.columns


# In[7]:


movies=movies[['id', 'title', 'overview', 'genre','rating']]
movies


# In[8]:


movies['tags'] = movies['overview']+movies['genre']
movies


# In[9]:


new_data  = movies.drop(columns=['overview', 'genre'])
new_data


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


cv=CountVectorizer(max_features=10000, stop_words='english')
cv


# In[12]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
vector.shape


# In[13]:


similarity=cosine_similarity(vector)
similarity


# In[14]:


new_data[new_data['title']=="The Godfather"].index[0]


# In[15]:


distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


# In[16]:


def recommand(movies):
    index=new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)


# In[17]:


recommand("Iron Man")


# In[18]:


import pickle


# In[19]:


pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[20]:


pickle.load(open('movies_list.pkl', 'rb'))


# In[ ]:





# In[ ]:




