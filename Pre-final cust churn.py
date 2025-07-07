#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


churn = pd.read_excel(r'D:\Mak\Jain study\Customer_Churn.xlsx', sheet_name = 'Data for DSBA')
churn.head()


# In[3]:


churn.shape


# In[4]:


churn.info()


# In[5]:


churn.describe().T


# In[6]:


special_chars = ['$', '@', '&', '#', '*']

# Replace special characters with NaN in the entire DataFrame
churn = churn.applymap(lambda x: np.nan if isinstance(x, str) and any(char in x for char in special_chars) else x)

missing_values = churn.isna().sum()
print(missing_values)


# In[7]:


churn["account_segment"] = churn["account_segment"].replace('Super +','Super Plus').replace('Regular +','Regular Plus')

churn.account_segment.unique()


# In[8]:


print("kurtosis and skewness of dataste is as below")
pd.DataFrame(data = [churn.kurtosis(), churn.skew()], index=['Kurtosis','Skewness']).T.round(2)


# In[9]:


print("standard deviation of variables")
print(churn.std())


# In[10]:


churn.hist(figsize=(20,15));


# In[11]:


#Univariate analysis
sns.countplot(x='Payment', data=churn)
plt.show()


# In[12]:


sns.countplot(x='account_segment', data=churn)
plt.show()


# In[13]:


churn["Gender"] = churn["Gender"].replace("M",'Male').replace("F",'Female')


# In[14]:


#Bivairiate analysis
sns.catplot(y="Gender", hue='Churn', kind='count', data=churn)
plt.show()


# In[15]:


sns.catplot(y="Marital_Status", hue="Churn", kind="count", data=churn)
plt.show()


# In[16]:


churn.dtypes


# In[17]:


missing_info = churn.isna().sum()
missing_percent = (missing_info / len(churn)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_info, 'Percent': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0])


# In[18]:


churn = churn.loc[:, churn.isnull().mean() < 0.3]

churn = churn.dropna()


# In[19]:


churn.isna().sum().sum()


# In[20]:


cat_cols = churn.select_dtypes(include=['object', 'category']).columns.tolist()
churn[cat_cols] = churn[cat_cols].astype(str)
churn = pd.get_dummies(churn, columns=cat_cols, drop_first=True)

print(churn.columns.tolist())


# In[21]:


X = churn.drop('Churn', axis=1)
y = churn['Churn']


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[25]:


#LogisticRegression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# In[27]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
plt.title("Confusion Matrix for Log Reg model")
plt.show()


# In[28]:


#RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[29]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix for RandForest Classifier model")
plt.show()


# In[30]:


#SupportVectorClassifier model
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Precision:", precision_score(y_test, y_pred_svc))
print("Recall:", recall_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


# In[31]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test)
plt.title("Confusion Matrix for SVClassifier model")
plt.show()


# In[32]:


#KNearestNeighborsClassifier model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# In[33]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.title("Confusion Matrix for KNN Classifier model")
plt.show()


# In[34]:


#XGBoostCLassifier model
import xgboost as xgb

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# In[35]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title("Confusion Matrix for XGB Classifier model")
plt.show()


# In[36]:


from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Plot ROC Curve
RocCurveDisplay.from_estimator(
    xgb_model, 
    X_test, 
    y_test
)
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')  # Baseline
plt.title("ROC Curve for XGBoost Model")
plt.legend()
plt.show()

# Calculate AUC score
from sklearn.metrics import roc_auc_score
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for Class 1
auc_score = roc_auc_score(y_test, y_prob_xgb)
print(f"AUC Score: {auc_score:.4f}")

