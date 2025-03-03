#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


train_data = pd.read_csv('train_ctrUa4K.csv')
train_data.head()


# In[9]:


print(train_data.shape)


# In[10]:


train_data.describe()


# In[14]:


train_data.info()


# In[15]:


def missing_values(df) :
    a = num_null_values = df.isnull().sum()
    return a


# In[16]:


missing_values(train_data)


# In[17]:


train_data.drop(["Loan_ID","Dependents"], axis=1, inplace=True)


# In[18]:


train_data


# In[19]:


cols = train_data[["Gender","Married","Self_Employed"]]
for i in cols:
    train_data[i].fillna(train_data[i].mode().iloc[0], inplace=True)


# In[20]:


train_data.isnull().sum()


# In[21]:


n_cols = train_data[["LoanAmount", "Loan_Amount_Term","Credit_History"]]
for i in n_cols:
    train_data[i].fillna(train_data[i].mean(axis=0), inplace=True)


# In[22]:


def bar_chart(col):
    Approved = train_data[train_data["Loan_Status"]=="Y"] [col].value_counts()
    Disapproved = train_data[train_data["Loan_Status"]=="N"] [col].value_counts()
    
    df1 = pd.DataFrame([Approved, Disapproved])
    df1.index = ["Approved", "Disapproved"]
    df1.plot(kind="bar")


# In[23]:


bar_chart("Gender")


# In[24]:


bar_chart("Married")


# In[25]:


bar_chart("Education")


# In[26]:


bar_chart("Self_Employed")


# In[27]:


from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
train_data[["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]] = ord_enc.fit_transform(
    train_data[["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]]
)
train_data.head()


# In[28]:


train_data[['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']] = train_data[
    ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'] ].astype(int)


# In[29]:


train_data


# In[30]:


from sklearn.model_selection import train_test_split
X = train_data.drop("Loan_Status",axis=1)
y = train_data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[32]:


from sklearn.naive_bayes import GaussianNB

gfc = GaussianNB()
gfc.fit(X_train, y_train)
pred1 = gfc.predict(X_test)


# In[33]:


from sklearn.metrics import precision_score, recall_score, accuracy_score
def loss(y_true, y_pred):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    print(pre)
    print(rec)
    print(acc)


# In[34]:


loss(y_test, pred1)


# In[37]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


param_grid = {'C': [0.1, 1, 10, 100, 1000],
             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose = 3)
grid.fit(X_train, y_train)


# In[39]:


grid.best_params_


# In[41]:


svc = SVC(C= 0.1, gamma= 1, kernel= 'rbf')
svc.fit(X_train, y_train)
pred2 = svc.predict(X_test)
loss(y_test,pred2)


# In[47]:


get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier

xgb = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
xgb.fit(X_train, y_train)
pred3 = xgb.predict(X_test)
loss(y_test, pred3)


# In[66]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def randomized_search (params, runs=20, clf=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2)
    rand_clf.fit(X_train, y_train)
    best_model = rand_clf.best_estimator_

    # Extract best score
    best_score = rand_clf.best_score_

    #Print best score
    print("Training score: {:.3f}".format(best_score))

    #Predict test set labels
    y_pred = best_model.predict(X_test)

    #Compute accuracy
    accuracy = accuracy_score (y_test, y_pred)

    #Print accuracy
    print('Test score: {:.3f}'.format(accuracy))

    return best_model


# In[67]:


randomized_search(params={'criterion':['entropy', 'gini'],
                                'splitter':['random', 'best'],
                        'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01],
                        'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
                        'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04],
                        'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                        'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                        'max_features':['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                        'max_depth':[None, 2,4,6,8],
                        'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                                          })                  


# In[68]:


ds = DecisionTreeClassifier(max_depth=8, max_features=0.9, max_leaf_nodes=30,
                       min_impurity_decrease=0.05, min_samples_leaf=0.02,
                       min_samples_split=10, min_weight_fraction_leaf=0.005,
                       random_state=2, splitter='random')
ds.fit(X_train, y_train)
pred4 =ds.predict(X_test)
loss(y_test, pred4)


# In[69]:


import joblib
joblib.dump(ds, "model.pkl")
model = joblib.load('model.pkl')
model.predict(X_test)


# In[ ]:




