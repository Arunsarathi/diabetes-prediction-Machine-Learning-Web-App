#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("pima-data.csv")


# In[3]:


data.shape


# In[4]:


data.head(5)


# In[5]:


# check if any null value is present
data.isnull().values.any()


# In[6]:


## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[7]:


data.corr()


# ## Changing the diabetes column data from boolean to number

# In[8]:


diabetes_map = {True: 1, False: 0}


# In[9]:


data['diabetes'] = data['diabetes'].map(diabetes_map)


# In[10]:


data.head(5)


# In[11]:


diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])


# In[12]:


(diabetes_true_count,diabetes_false_count)


# In[13]:


data.columns


# In[14]:


## Train Test Split

from sklearn.model_selection import train_test_split
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']


# In[15]:


X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# ## Check how many other missing(zero) values

# In[16]:


print("total number of rows : {0}".format(len(data)))

print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


# In[32]:


# using for find null value
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[17]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[18]:


## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())


# In[19]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics
import pickle

print("Accuracy =",format(metrics.accuracy_score(y_test, predict_train_data)))


pickle.dump(random_forest_model, open('diabeties.pkl','wb'))

model = pickle.load( open('diabeties.pkl','rb'))




print(model.predict([[4,154,72,126,31,0.33,37,1.14]]))

result= model.predict([[4,154,72,126,31,0.33,37,1.14]])
if result == 0:
    print('you dont have diabetes')

else:
    print('you have diabetes')
