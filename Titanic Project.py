
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


np.sum(titanic.Survived)


# In[4]:


titanic.describe()


# In[5]:


titanicDF = titanic.iloc[:,(list(range(1,3)) + list(range(4,8)) + list(range(9,12)))]
titanicDF.head()


# In[6]:


titanicDF.corr()


# In[7]:


#Useful in finding missing values all at once in df
titanicDF.info()


# In[8]:


print (len(titanicDF.Embarked.unique()))


# In[9]:


print (len(titanicDF.Cabin.unique()))


# In[10]:


print (len(titanicDF.Pclass.unique()))


# In[11]:


#Checking if age is normally distributed
print (titanicDF.Age.median())


# In[12]:


print (titanicDF.Age.mode())


# In[13]:


print (titanicDF.Cabin.mode())


# In[14]:


print ((titanicDF.Fare.isnull().sum()))
#No missing value in Fare


# In[15]:


titanicDF.describe()


# In[16]:


#get row at a particular index
print (titanicDF.iloc[33])


# In[17]:


#Outlier treatment
age_99 = titanicDF.Age.quantile(0.99)
list1 = titanicDF[(titanicDF['Age']>age_99)].Age.index
titanicDF.iloc[list1]


# In[18]:


from scipy import stats


# In[19]:


#tDF = titanicDF[(np.abs(stats.zscore(titanicDF)) < 3).all(axis=1)]
#tDF.describe()


# In[20]:


#MIssing value treatment
from sklearn.preprocessing import Imputer


# In[21]:


imp = Imputer(axis=0,strategy='median',missing_values='NaN')
temp = imp.fit_transform(titanicDF.Age.reshape(-1,1))
titanicDF['Age'] = temp
titanicDF.head(10)


# In[22]:

titanicDF1 = titanicDF[(titanicDF['Age']<=age_99)]
titanicDF1.describe()


# In[23]:


titanicDF1.tail()


# In[24]:

sibsp_99 = titanicDF1.SibSp.quantile(0.99)
titanicDF2 = titanicDF1[(titanicDF1['SibSp']<=sibsp_99)]
titanicDF2.describe()


# In[25]:


#titanicDF2.reset_index()
parch_99 = titanicDF2.Parch.quantile(0.99)
titanicDF3 = titanicDF2[(titanicDF2['Parch']<=parch_99)]
titanicDF3.describe()


# In[26]:


Fare_99 = titanicDF3.Fare.quantile(0.99)
titanicDF4 = titanicDF3[(titanicDF3['Fare']<=Fare_99)]
titanicDF4.describe()


# In[27]:


titanicDF = titanicDF4
titanicDF.describe()


# In[28]:


titanicDF.head()


# In[29]:


#Perform cross tabulation
class_dummy = pd.get_dummies(titanicDF['Pclass'],prefix='Pclass_')
titanicDF = pd.concat([titanicDF,class_dummy],axis=1)
titanicDF.head()


# In[30]:


class_dummy = pd.get_dummies(titanicDF['Sex'],prefix='Sex_')
titanicDF = pd.concat([titanicDF,class_dummy],axis=1)
titanicDF.head()


# In[31]:


print (len(titanicDF.SibSp.unique()))


# In[32]:


class_dummy = pd.get_dummies(titanicDF['SibSp'],prefix='SibSp_')
titanicDF = pd.concat([titanicDF,class_dummy],axis=1)
titanicDF.head()


# In[33]:


class_dummy = pd.get_dummies(titanicDF['Parch'],prefix='Parch_')
titanicDF = pd.concat([titanicDF,class_dummy],axis=1)
titanicDF.head()


# In[34]:


class_dummy = pd.get_dummies(titanicDF['Embarked'],prefix='Embarked_')
titanicDF = pd.concat([titanicDF,class_dummy],axis=1)
titanicDF.head()


# In[35]:


titanicDF.describe()


# In[36]:


titanicDF.columns


# In[37]:


del titanicDF['Cabin']


# In[38]:


del titanicDF['Embarked']


# In[39]:


from sklearn import linear_model


# In[40]:


titanicTrain, titanicTest = train_test_split(titanicDF,test_size=0.3,random_state=45)


# In[41]:


titanicTrain_X = titanicTrain.iloc[:,range(6,titanicTrain.shape[1])]
titanicTrain_X.head()


# In[42]:


titanicTrain_Y = titanicTrain.Survived
titanicTrain_Y.describe()


# In[43]:


titanicTest_X = titanicTest.iloc[:,range(6,titanicTest.shape[1])]
titanicTest_X.head()


# In[44]:


titanicTest_Y = titanicTest.Survived
titanicTest_Y.describe()


# In[45]:


tmodel1 =  linear_model.LogisticRegression()


# In[46]:


titanicTrain_X.columns


# In[47]:


tmodel1.fit(titanicTrain_X,titanicTrain_Y)


# In[48]:


predicted1 = tmodel1.predict(titanicTest_X)
predicted1


# In[49]:


from sklearn import metrics


# In[50]:


print (metrics.classification_report(titanicTest_Y,predicted1))


# In[51]:


print (metrics.confusion_matrix(titanicTest_Y, predicted1))


# In[52]:


#ROC Curve analysis
fpr, tpr, threshold = metrics.roc_curve(titanicTest_Y,predicted1,pos_label=1)


# In[53]:


print (threshold)


# In[54]:


#Building CAP Curve to find fitment of model
from scipy import integrate


# In[55]:


def capcurve(y_values, y_preds_proba):
	num_pos_obs = np.sum(y_values)
	num_count = len(y_values)
	rate_pos_obs = float(num_pos_obs) / float(num_count)
	ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
	xx = np.arange(num_count) / float(num_count - 1)

	y_cap = np.c_[y_values,y_preds_proba]
	y_cap_df_s = pd.DataFrame(data=y_cap)
	y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(drop=True)

	print(y_cap_df_s.head(20))

	yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
	yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

	percent = 0.5
	row_index = np.trunc(num_count * percent)

	val_y1 = yy[row_index]
	val_y2 = yy[row_index+1]
	if val_y1 == val_y2:
		val = val_y1*1.0
	else:
		val_x1 = xx[row_index]
		val_x2 = xx[row_index+1]
		val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

	sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
	sigma_model = integrate.simps(yy,xx)
	sigma_random = integrate.simps(xx,xx)

	ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
	#ar_label = 'ar value = %s' % ar_value

	fig, ax = plt.subplots(nrows = 1, ncols = 1)
	ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
	ax.plot(xx,yy, color='red', label='User Model')
	#ax.scatter(xx,yy, color='red')
	ax.plot(xx,xx, color='blue', label='Random Model')
	ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
	ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

	plt.xlim(0, 1.02)
	plt.ylim(0, 1.25)
	plt.title("CAP Curve - a_r value ="+str(ar_value))
	plt.xlabel('% of the data')
	plt.ylabel('% of positive obs')
	plt.legend()
	plt.show()


# In[56]:


y_pred_proba = tmodel1.predict_proba(titanicTest_X)
#capcurve(titanicTest_Y, y_pred_proba[:,1])


# In[57]:


#AUC score works only when the data is binary i.e 0,1
auc = metrics.roc_auc_score(titanicTest_Y,predicted1)
print (auc)


# In[58]:


plt.plot(fpr,tpr,'b',label='AUC = %0.2f'% auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.show()


# In[59]:


#AUC is 0.77 which is not so good. Need to do regularization
#penalty=l1 makes it a Lasso regression


# In[60]:


tmodel_lasso = linear_model.LogisticRegression(penalty='l1')


# In[61]:


tmodel_lasso.fit(titanicTrain_X,titanicTrain_Y)


# In[62]:


titanicTrain_X.columns


# In[63]:


tmodel_lasso.coef_


# In[64]:


predicted2 = tmodel_lasso.predict(titanicTest_X)


# In[65]:


print (metrics.classification_report(titanicTest_Y,predicted2))


# In[66]:


print (metrics.roc_auc_score(titanicTest_Y,predicted2))


# In[67]:


#Using Naive Bayes


# In[68]:


from sklearn.naive_bayes import GaussianNB


# In[69]:


tmodel_nb = GaussianNB()


# In[70]:


tmodel_nb.fit(titanicTrain_X,titanicTrain_Y)


# In[71]:


predicted3 = tmodel_nb.predict(titanicTest_X)


# In[72]:


print (metrics.classification_report(titanicTest_Y, predicted3))


# In[73]:


predicted2


# In[74]:


titanic1 = pd.read_csv('train.csv')
titanic1.head()


# In[75]:


del titanic1['Cabin']
del titanic1['Ticket']
del titanic1['Embarked']
del titanic1['Name']


# In[76]:


del titanic1['PassengerId']


# In[77]:


#Doing cross tabulation as NB doesnt understand chars
Sex_dummy = pd.get_dummies(titanic1['Sex'], prefix='Sex_')
titanicDF1 = pd.concat([titanic1,Sex_dummy],axis=1)
titanicDF1.head()


# In[78]:


del titanicDF1['Sex']


# In[79]:


titanicDF1.head()


# In[80]:


imp1 = Imputer(axis=0,strategy='median',missing_values='NaN')
temp = imp1.fit_transform(titanicDF1.Age.reshape(-1,1))
titanicDF1['Age'] = temp


# In[81]:


print (titanicDF1['Age'].isnull().sum())


# In[82]:


titanicDF1.info()


# In[83]:


titanic1_train, titanic1_test = train_test_split(titanicDF1, test_size=0.2, random_state=10)


# In[84]:


titanic1_train_X = titanic1_train.iloc[:,range(1,titanicDF1.shape[1])]
titanic1_train_X.head()


# In[85]:


titanic1_train_Y = titanic1_train.Survived
titanic1_train_Y.head()


# In[86]:


titanic1_test_X = titanic1_test.iloc[:,range(1,titanicDF1.shape[1])]
titanic1_test_Y = titanic1_test.Survived


# In[87]:


tmodel_nb1 = GaussianNB()


# In[88]:


tmodel_nb1.fit(titanic1_train_X,titanic1_train_Y)


# In[89]:


predicted4 = tmodel_nb1.predict(titanic1_test_X)


# In[90]:


print (metrics.classification_report(titanic1_test_Y, predicted4))


# In[91]:


tmodel_nb1.score(titanic1_test_X,titanic1_test_Y)


# In[92]:


#Using KNN
from sklearn.neighbors import KNeighborsClassifier


# In[93]:


score = []
for i in range(1,20):
    knn = KNeighborsClassifier(i)
    knn.fit(titanicTrain_X, titanicTrain_Y)
    score.append(knn.score(titanicTest_X, titanicTest_Y))    


# In[94]:


plt.plot(range(1,20), score)
plt.show()


# In[95]:


knn1 = KNeighborsClassifier(n_neighbors=3)


# In[96]:


knn1.fit(titanicTrain_X, titanicTrain_Y)


# In[97]:


predicted5 = knn1.predict(titanicTest_X)


# In[98]:


print (metrics.classification_report(titanicTest_Y, predicted5))


# In[99]:


print (metrics.roc_auc_score(titanicTest_Y, predicted5))


# In[100]:


#Using gridsearchcv to find best parameters for KNN
from sklearn.model_selection import GridSearchCV


# In[101]:


knn2 = KNeighborsClassifier()
params = [{"n_neighbors":range(1,20)}]
gridsearch = GridSearchCV(estimator = knn2, cv=10, param_grid=params)


# In[102]:


gridsearch.fit(titanicTrain_X, titanicTrain_Y)


# In[103]:


gridsearch.best_params_


# In[104]:


gridsearch.grid_scores_


# In[105]:


#Using gradient descent with regularization


# In[106]:


titanicTrain, titanicTest = train_test_split(titanicDF,test_size=0.2,random_state=45)


# In[107]:


titanicTrain_X_Proper = titanicTrain.iloc[:,list(range(6,9)) + list(range(10,11)) + list(range(12,17)) + 
                                                 list(range(18,22)) + list(range(23,25))]
titanicTrain_X_Proper.head()


# In[108]:


titanicTest_X_Proper = titanicTest.iloc[:,list(range(6,9)) + list(range(10,11)) + list(range(12,17)) + 
                                                 list(range(18,22)) + list(range(23,25))]
titanicTest_X_Proper.columns


# In[109]:


titanicTrain_gradient_X = titanicTrain_X_Proper
titanicTest_gradient_X = titanicTest_X_Proper


# In[110]:


titanicTrain_Y = titanicTrain.Survived
titanicTrain_Y.head()


# In[111]:


titanicTrain_gradient_Y = np.array([titanicTrain_Y]).transpose()
titanicTrain_gradient_Y.shape


# In[112]:

titanicTrain_gradient_Y.min()


# In[123]:


titanicTrain_gradient_Y.shape[0]


# In[124]:


from scipy.special import expit


# In[125]:


titanicTrain_gradient_X.shape


# In[126]:


def getJTheta(p_theta, p_titanicTrain_gradient_X, p_titanicTrain_gradient_Y):
    thetaX = np.dot(p_theta.transpose(),p_titanicTrain_gradient_X())
    eThetaX = expit(-thetaX)
    denom = np.add(1,eThetaX)
    hx = np.divide(1,denom)
    p_hxNewLogSafe = np.subtract(hx,epsilon)
    jTheta = np.add(np.multiply(p_titanicTrain_gradient_Y.transpose(),np.log(p_hxNewLogSafe)),
                      np.multiply(np.subtract(1,p_titanicTrain_gradient_Y.transpose()),np.log(np.subtract(1,p_hxNewLogSafe))))
    return jTheta     


# In[127]:


def calculateGradientOfJ(p_theta,p_titanicTrain_gradient_X, p_titanicTrain_gradient_Y, p_epsilon):    
    JPlus = getJTheta(np.add(p_theta,epsilon), p_titanicTrain_gradient_X, p_titanicTrain_gradient_Y)
    JMinus = getJTheta(np.subtract(p_theta,epsilon), p_titanicTrain_gradient_X, p_titanicTrain_gradient_Y)
    JDiff = np.divide(np.subtract(JPlus,JMinus),titanicTrain_gradient_Y.shape[0])
    JDiff
    return JDiff


# In[128]:


from sklearn.preprocessing import StandardScaler


# In[129]:


scalar = StandardScaler()


# In[133]:


titanicTrain_gradient_X.head()


# In[134]:


titanicTrain_gradient_sc_T_X = scalar.fit_transform(titanicTrain_gradient_X)
print (titanicTrain_gradient_sc_T_X.shape)
titanicTrain_gradient_sc_X = titanicTrain_gradient_sc_T_X

# In[139]:


titanicTrain_gradient_sc_X[:,0]


# In[140]:


X_gradient1 = np.ones(titanicTrain_gradient_sc_X.shape[0])
X_gradient2 = np.c_[X_gradient1,titanicTrain_gradient_sc_X]
X_gradient2.shape


# In[145]:


#X_gradient2[:,0]


# In[141]:


titanicTrain_gradient_sc_X = X_gradient2.transpose
titanicTrain_gradient_sc_X().shape


# In[142]:


titanicTrain_gradient_sc_X()[0,:]


# In[143]:


#calculateGradientOfJ(theta1_practice,X_practice,Y_practice,0.1).shape


# In[290]:


iteration = 30000
initialThetaSeed = 10
np.random.seed(initialThetaSeed)
thetanew = np.random.randint(0,initialThetaSeed,size=(titanicTrain_gradient_sc_X().shape[0],1))
theta = thetanew
errorLog = np.empty(iteration)
gradientDiffLog = np.empty(iteration)
#lambdaa = 0.48
#alpha = 0.0043
lambdaa = 0.4
alpha = 0.07
epsilon = 0.001
for i in range(iteration):
    thetaX = np.dot(theta.transpose(),titanicTrain_gradient_sc_X())
    eThetaX = expit(-thetaX)
    denom = np.add(1,eThetaX)
    hx = np.divide(1,denom)
    diff = np.subtract(hx,titanicTrain_gradient_Y.transpose())    
    derivative = np.dot(diff,titanicTrain_gradient_sc_X().transpose())
    #Implementing regularization    
   
    #print ("before", thetaRegularized)
    thetaRegularized = thetanew*(1-(alpha*lambdaa/titanicTrain_gradient_sc_X().shape[1]))
    #print ("after", thetaRegularized)
    #Not regularizing theta0 (the constant)
    thetaRegularized[0][0] = thetaRegularized[0][0]/(1-(alpha*lambdaa/titanicTrain_gradient_sc_X().shape[1]))
    #print ("after adjustment", thetaRegularized)
    theta = np.subtract(thetaRegularized,np.multiply((alpha/titanicTrain_gradient_sc_X().shape[1]),derivative.transpose()))
    thetaXNew = np.dot(theta.transpose(),titanicTrain_gradient_sc_X())
    eThetaXNew = expit(-thetaXNew)
    denomNew = np.add(1,eThetaXNew)
    hxNew = np.divide(1,denomNew)
    hxNewLogSafe = np.subtract(hxNew,epsilon)
    costerror = -np.divide(np.add(np.multiply(titanicTrain_gradient_Y.transpose(),np.log(hxNewLogSafe)),
                      np.multiply(np.subtract(1,titanicTrain_gradient_Y.transpose()),np.log(np.subtract(1,hxNewLogSafe)))),
                          titanicTrain_gradient_Y.shape[0])
    #print (costerror.shape)
    errorLog[i] = costerror.sum()
    thetanew = theta
    


# In[291]:


100*(errorLog[iteration-2] - errorLog[iteration-1])/errorLog[iteration-2]


# In[292]:


plt.plot(range(iteration),errorLog)
plt.show()


# In[259]:


def getPredictedFromGradient(theta_p,testX_p):
    thetaX = np.dot(theta_p.transpose(),testX_p())
    denom = np.add(1,expit(-thetaX))
    hx = np.divide(1,denom)
    #print (hx)
    hxT = hx.transpose()          
    return hxT


# In[260]:


titanicTrain_gradient_X.transpose().shape


# In[261]:


titanicTest_gradient_X.transpose().shape


# In[262]:


titanicTest_gradient_sc_T_X = scalar.transform(titanicTest_gradient_X)
#print (titanicTest_gradient_sc_T_X.shape)
titanicTest_gradient_sc_X = titanicTest_gradient_sc_T_X
#titanicTest_gradient_sc_X = titanicTest_gradient_sc_T_X.transpose()
print (titanicTest_gradient_sc_X.shape)


# In[263]:


X_t_gradient1 = np.ones(titanicTest_gradient_sc_X.shape[0])
X_t_gradient2 = np.c_[X_t_gradient1,titanicTest_gradient_sc_X]
titanicTest_gradient_sc_X = X_t_gradient2.transpose


# In[264]:


titanicTest_gradient_sc_X().shape


# In[265]:


titanicTest_Y = titanicTest.Survived
titanicTest_gradient_Y = np.array([titanicTest_Y]).transpose()
titanicTest_gradient_Y.shape


# In[266]:


predicted_gradient = getPredictedFromGradient(theta,titanicTest_gradient_sc_X)


# In[267]:


for i in range(0,predicted_gradient.shape[0]):
    if predicted_gradient[i][0] > 0.51:
        predicted_gradient[i][0] = 1
    else:
        predicted_gradient[i][0] = 0


# In[271]:


print (metrics.classification_report(titanicTest_gradient_Y, predicted_gradient))


# In[269]:


print (metrics.roc_auc_score(titanicTest_gradient_Y, predicted_gradient))


# In[ ]:


#Using Logistic regression
tmodel_log1 = linear_model.LogisticRegression(C=0.2)

tmodel_log1.fit(titanicTrain_X_Proper,titanicTrain_Y)

predicted_log1 = tmodel_log1.predict(titanicTest_X_Proper)

print (metrics.classification_report(titanicTest_Y, predicted_log1))


# In[ ]:


#Using Neural Networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[258]:


titanicTrain_X_Proper.shape


# In[259]:


np.random.seed(10)
dropoutrate = 0.2
def buildClassifier():
    classifier = Sequential()
    classifier.add(Dense(8, activation='relu', input_dim=15, kernel_initializer='uniform'))
    classifier.add(Dropout(dropoutrate))
    classifier.add(Dense(8, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(dropoutrate))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier


# In[260]:


classifier = KerasClassifier(build_fn=buildClassifier, epochs=200, batch_size=5)

# In[261]:

classifier.fit(scalar.fit_transform(titanicTrain_X_Proper), titanicTrain_Y)
predicted_NN = classifier.predict(scalar.transform(titanicTest_X_Proper))


# In[263]:


print (metrics.classification_report(titanicTest_Y, predicted_NN))


# In[270]:


print (metrics.roc_auc_score(titanicTest_gradient_Y, predicted_NN))

