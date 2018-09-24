
# coding: utf-8

# In[1]:


import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy.stats as stat
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# In[2]:


titanic = pd.read_csv('train.csv')
titanic.head()


# In[3]:


titanic.describe()


# In[4]:


#Using decision tree without any data cleaning on the dataset


# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


# In[6]:


from sklearn.preprocessing import Imputer


# In[7]:


imp = Imputer(axis=0,missing_values='NaN',strategy='median')
temp = imp.fit_transform(titanic['Age'].reshape(-1,1))
titanic['Age'] = temp
titanic.head()


# In[8]:


Embarked_Dummy = pd.get_dummies(titanic.Embarked,prefix='Embarked')
titanic = pd.concat([titanic,Embarked_Dummy],axis=1)
titanic.head()


# In[9]:


Pclass_Dummy = pd.get_dummies(titanic.Pclass,prefix='Pclass')
titanic = pd.concat([titanic,Pclass_Dummy],axis=1)


# In[10]:


Sex_Dummy = pd.get_dummies(titanic.Sex,prefix='Sex')
titanic = pd.concat([titanic,Sex_Dummy],axis=1)


# In[11]:


SibSp_Dummy = pd.get_dummies(titanic.SibSp,prefix='SibSp')
titanic = pd.concat([titanic,SibSp_Dummy],axis=1)


# In[12]:


Parch_Dummy = pd.get_dummies(titanic.Parch,prefix='Parch')
titanic = pd.concat([titanic,Parch_Dummy],axis=1)


# In[13]:


titanic.columns


# In[14]:


titanicDF = titanic.iloc[:,(list(range(1,2)) + list(range(5,6)) + list(range(9,10)) + list(range(12,titanic.shape[1])))]
titanicDF.columns


# In[15]:


Titanictrain1, Titanictest1 = train_test_split(titanicDF,test_size=0.2,random_state=10)


# In[16]:


Titanictrain1X = Titanictrain1.iloc[:,1:titanicDF.shape[1]]
Titanictrain1X.head()


# In[17]:


Titanictrain1Y = Titanictrain1.Survived
Titanictrain1Y.head()


# In[18]:


Titanictest1X = Titanictest1.iloc[:,1:titanicDF.shape[1]]
Titanictest1X.head()


# In[19]:


Titanictest1Y = Titanictest1.Survived
Titanictest1Y.head()


# In[20]:


dt1 = DecisionTreeClassifier(max_depth=6,min_samples_split=10,random_state=15)


# In[25]:


dt1.fit(Titanictrain1X,Titanictrain1Y)


# In[26]:


predicted1 = dt1.predict(Titanictest1X)


# In[27]:


print (metrics.classification_report(Titanictest1Y,predicted1))


# In[28]:


auc = metrics.roc_auc_score(Titanictest1Y,predicted1)
print (auc)


# In[29]:


#Using GridSearch cross validation


# In[30]:


from sklearn.grid_search import GridSearchCV
from time import time
from operator import itemgetter


# In[31]:


#This function performs a grid search with cross validation and returns the accuracy values 
def GridSearch_BestParam(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = grid_search.grid_scores_
    return  top_params


# In[32]:


param_grid = {"criterion":["gini","entropy"],"min_samples_split":[2,10,20],"max_depth":[None,2,5,10],
             "min_samples_leaf":[1,5,10]}


# In[33]:


cvDT = DecisionTreeClassifier()


# In[34]:


#cv refers to the number of subsets of data created by algorithm
topparams= GridSearch_BestParam(Titanictrain1X,Titanictrain1Y,cvDT,param_grid,cv =10)
topparams


# In[35]:


topScore = sorted(topparams,key = itemgetter(1),reverse = True)
paramCV = topScore[0].parameters
paramCV


# In[36]:


dt2=DecisionTreeClassifier(max_depth=paramCV['max_depth'],
                                   min_samples_leaf=paramCV['min_samples_leaf'],
                                  min_samples_split = paramCV['min_samples_split'],
                                  criterion =paramCV['criterion'] )


# In[37]:


dt2.fit(Titanictrain1X,Titanictrain1Y)


# In[38]:


predicted2 = dt2.predict(Titanictest1X)


# In[39]:


#Improvement of 1% over simple decision tree implementation
print (metrics.classification_report(Titanictest1Y,predicted2))


# In[40]:


#Print importance of features
tempDF = pd.DataFrame()
tempDF['Features'] = Titanictrain1X.columns
tempDF['Importance'] = dt2.feature_importances_
tempDF = tempDF.iloc[np.argsort(tempDF['Importance'])[::-1], :]
tempDF


# In[41]:


#Using ensemble methods, starting with random forest and then gradient boost


# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[43]:


#RF without GridSearchCV
rf1 = RandomForestClassifier(max_features=10,n_estimators=500)


# In[44]:


rf1.fit(Titanictrain1X,Titanictrain1Y)


# In[45]:


predicted3 = rf1.predict(Titanictest1X)


# In[46]:


print (metrics.classification_report(Titanictest1Y,predicted3))


# In[47]:


#RF with GridSearchCV


# In[48]:


param_grid = {"criterion":["gini","entropy"],"max_features":[3,4,5,6,7],"max_depth":[None,2,5,10]}


# In[49]:


rf2 = RandomForestClassifier(max_features=10,n_estimators=500)


# In[50]:


gridsearchcv = GridSearch_BestParam(Titanictrain1X,Titanictrain1Y,rf2,param_grid,cv =10)


# In[51]:


topScore = sorted(gridsearchcv,key = itemgetter(1),reverse = True)
paramCV = topScore[0].parameters
paramCV


# In[52]:


rf3 = RandomForestClassifier(max_features=paramCV['max_features'],max_depth=paramCV['max_depth'],criterion=paramCV['criterion'])
rf3.fit(Titanictrain1X,Titanictrain1Y)


# In[53]:


predicted4 = rf3.predict(Titanictest1X)


# In[54]:


print (metrics.classification_report(Titanictest1Y,predicted4))


# In[55]:


#Using gradient boost classifier


# In[56]:


from sklearn.ensemble import GradientBoostingClassifier


# In[57]:


gbTitanic = GradientBoostingClassifier(n_estimators =500, max_depth=6,learning_rate=0.02)
gbTitanic.fit(Titanictrain1X,Titanictrain1Y)


# In[58]:


predicted5 = gbTitanic.predict(Titanictest1X)


# In[59]:


print (metrics.classification_report(Titanictest1Y,predicted5))


# In[1]:


#Using SVC


# In[2]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[60]:


svc = SVC(C=1.0,gamma=0.5,kernel='rbf')


# In[61]:


svc.fit(Titanictrain1X,Titanictrain1Y)


# In[62]:


predicted6 = svc.predict(Titanictest1X)


# In[63]:


print (metrics.classification_report(Titanictest1Y,predicted6))


# In[64]:


auc = metrics.roc_auc_score(Titanictest1Y,predicted6)
print (auc)


# In[65]:


#Using LinearSVC


# In[66]:


linearsvc = LinearSVC(loss='squared_hinge',C=4)
linearsvc.fit(Titanictrain1X,Titanictrain1Y)


# In[67]:


predicted7 = linearsvc.predict(Titanictest1X)


# In[68]:


print (metrics.classification_report(Titanictest1Y,predicted7))


# In[69]:


auc = metrics.roc_auc_score(Titanictest1Y,predicted7)
print (auc)

