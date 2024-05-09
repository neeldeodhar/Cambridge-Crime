#!/usr/bin/env python
# coding: utf-8

# In[7]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering


# In[8]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('Cambridge Crime Data.csv')
#printing first 5 rows
df.head()



# In[9]:


#printing columns list
df.columns


# In[10]:


# Encoding, using ONE HOT ENCODER 
enc = OneHotEncoder()

df_onehot = pd.get_dummies(df[['Crime', 'Neighborhood']])

#df = enc.fit_transform(df[['Crime', 'Neighborhood']])
df_onehot['Reporting Area'] = df['Reporting Area']



# In[11]:


display(df)


# In[12]:


df_onehot.columns


# In[13]:


# dropping null values 
df_onehot.dropna(inplace=True) 


# In[14]:


# defining x
x_columns = 68

#no selected features as one hot encoding being used.
x = df_onehot.iloc[:, 0:x_columns].values


# In[15]:


#implementing Elbow method in Python
crimeelbow = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters = k)
    kmeanModel.fit(df_onehot)
    crimeelbow.append(kmeanModel.inertia_)


# In[16]:


#plotting Elbow method vs values of K
plt.figure(figsize = (16,8))
plt.plot(K, crimeelbow, 'bx-')
plt.xlabel('k values')
plt.ylabel ('elbow')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[17]:


#observation:

print("As per elbow method graph above, the curve is taking bend when clusters are 2")


# In[18]:


#defining a KMEANS model with 2 clusters
model = KMeans(n_clusters =2, random_state = 0)
model.fit(x)  


# In[19]:


#scaling and preprocessing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)


# In[20]:


#defining clusters for model
clusters = model.predict(x)


# In[21]:


#printing model inertia
model.inertia_


# In[22]:


# trying a different optimal value of k, i.e k = 3
model1 = KMeans(n_clusters =3, random_state = 0)
model1.fit(x)  


# In[23]:


#printing model1 inertia
model1.inertia_


# In[24]:


# printing first 10 data samples for model
y = model.fit_predict(x)
print(y[0:10])


# In[25]:


# printing first 10 data samples for model1
y1 = model1.fit_predict(x)
print(y1[0:10])


# In[26]:


#defining centres for model
centres = model.cluster_centers_
print(centres)


# In[27]:


df_onehot


# In[28]:


df_onehot = df_onehot.drop(df_onehot.columns[[0]],axis = 1)


# In[34]:


#plotting KMeans cluster between Crime and Reporting Area columns
import matplotlib.pyplot as plt
colors = ['magenta', 'blue', 'yellow']
for i in range(3):
    plt.scatter(x[y1 == i, 1], x[y1 == i, 67], c=colors[i])
plt.scatter(model1.cluster_centers_[:, 1], model1.cluster_centers_[:, 67], color='red', marker='+', s=300)
plt.title('K-Means Clustering')
plt.xlabel('Crime')
plt.ylabel('Reporting Area')


# In[33]:


#HIERARCHICAL clustering part of Agglomerative clustering:

# down loading Scipy library


# generating Agglomerative Clustering, (i.e. a Dendogram)


#SciPy Dendogram (structuring
plt.figure(figsize=(18,10))
plt.title('Dendrogram')
plt.xlabel('Crime')
plt.ylabel('Euclidean distances')
dendrogram = sch.dendrogram(sch.linkage(x, method ='ward'),
                            color_threshold=200, 
                            above_threshold_color='red') 
plt.show()



# In[31]:


# SciKitLearn HIERARCHICAL clustering part of Agglomerative clustering -WITH 3 clusters
modelHC = AgglomerativeClustering(n_clusters = 3, affinity ='euclidean',
                                 linkage ='ward')
yHC = modelHC.fit_predict(df_onehot)

plt.scatter(x[:, 1], x[:, 67], c=yHC, cmap="rainbow")
plt.xlabel('Crime Accident')
plt.ylabel('Reporting area')
plt.title("SciKitLearn Agglomerative Clustering")
plt.show()


# In[32]:


#MEMORY ERRORS

print("HIERARCHICAL CLUSTERING IS GIVING MEMORY ERRORS ABOVE, DUE TO NATURE OF THE DATASET")


# In[ ]:




