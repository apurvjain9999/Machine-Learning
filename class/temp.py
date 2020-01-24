#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[3]:


#load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("pima-indians-diabetes.csv", header = None, names = col_names)
pima.head()


# In[5]:


#Feature selection
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  #Selected features
X = X[1:]
y = pima.label          #Class/output variable
y = y[1:]


# In[9]:


#Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
X_test


# In[11]:


#Building decision tree model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(X_train,y_train)
clf


# In[12]:


y_pred = clf.predict(X_test)


# In[13]:


#Evaluate your model
print("Accuracy = ", metrics.accuracy_score(y_test,y_pred))


# In[21]:


# To visualize the generated decision tree in tree form, you can use
# graphviz function

# pip install graphviz and then conda install graphviz 
# pip install pydotplus and then conda install pydotplus, if you have anaconda

#After successfully installed these two packages, run the following script

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, 
                 filled = True, rounded = True,
               special_characters=True, feature_names=feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('pima-indians-diabetes-entropy.png')
Image(graph.create_png())


# In[ ]:




