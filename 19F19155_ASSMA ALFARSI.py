#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn import metrics
from  sklearn.model_selection import train_test_split


# In[22]:


data = pd.read_csv("Iris.csv")
data.sample(5)


# In[23]:


data.head(5)


# In[41]:


data=pd.read_csv('Iris.csv')

y=data.pop('Species')
data.pop('Id')


species=np.unique(y)

y=y.map(dict(zip(np.unique(y),np.arange(len(np.unique(y))))))


# In[42]:


corr=df.corr()
sns.heatmap(corr,annot=True)
plt.show()


# In[24]:


sns.pairplot( data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species' )


# In[25]:


data.describe()


# In[26]:


df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


# In[28]:


df_norm.describe()


# In[29]:


target = data[['Species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
target.sample(n=5)


# In[30]:


df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)


# In[31]:


train, test = train_test_split(df, test_size = 0.3)
trainX = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
trainY=train.Species# output of our training data
testX= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
testY =test.Species   #output value of test data
trainX.head(5)


# In[32]:


trainY.head(5)


# In[33]:


testX.head(5)


# In[34]:


testY.head(5)


# In[35]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)


# In[36]:


clf.fit(trainX, trainY)


# In[37]:


prediction = clf.predict(testX)
print(prediction)


# In[38]:


print(testY.values)


# In[39]:


print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))

