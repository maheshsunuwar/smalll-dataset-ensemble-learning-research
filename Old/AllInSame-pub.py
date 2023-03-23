# %%
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# %%
clinical_data = pd.read_csv('data\\NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv', index_col=False)

# %%
# clinical_data = clinical_data.drop(['Case ID','Patient affiliation','%GG', 'Survival Status','Date of Last Known Alive', 'Date of Last Known Alive', 'CT Date', 'PET Date', 'Date of Recurrence'], axis=1)
clinical_data.head(2)

# %%
clinical_data['Quit Smoking Year'] = pd.to_datetime(clinical_data['Quit Smoking Year'], format='%Y')

# %%
clinical_data['Quit Smoking Year']

# %%
#replace Not Collected with NaN
clinical_data.replace('Not Collected', 'NaN', inplace=True)
clinical_data.replace('Not collected', 'NaN', inplace=True)
clinical_data.replace('Not Recorded In Database', 'NaN', inplace=True)
clinical_data.replace('NaN', np.nan, inplace=True)

# %%
#weight to numerical value
clinical_data['Weight (lbs)'] = pd.to_numeric(clinical_data['Weight (lbs)'])
# clinical_data.astype({'Weight (lbs)':'Int32'})

# %%
clinical_data['Quit Smoking'] = clinical_data['Quit Smoking Year'].isnull()
clinical_data['Quit Smoking Year'] = clinical_data.drop('Quit Smoking Year', axis=1)

# %%
clinical_data['Weight (lbs)'].fillna(-999, inplace=True)

# %%
dm = pd.DataFrame(pd.get_dummies(clinical_data))

# %%
dm = dm.drop(['Recurrence_no'], axis=1)

# %%
clinical_data['Weight (lbs)']

# %%
clinical_data['Quit Smoking']

# %%
#define x and y
X = dm.iloc[:, dm.columns != 'Recurrence_yes']
y = dm.iloc[:,-1].values

# %%
#clinical_data.iloc[:,-1]

# %% [markdown]
# Train Test Split

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# %%
X_train

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
y_train

# %%
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

nb_y_predict = nb_classifier.predict(X_test)
nb_y_predict

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score
cm = confusion_matrix(y_test, nb_y_predict)
ac = accuracy_score(y_test,nb_y_predict)
rs = recall_score(y_test,nb_y_predict)
ps = precision_score(y_test,nb_y_predict)

print(cm)
print(ac)
print(rs)
print(ps)

# %%
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC()
svm_classifier.fit(X_train, y_train)

svm_y_predict = svm_classifier.predict(X_test)
svm_y_predict

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score
cm = confusion_matrix(y_test, svm_y_predict)
ac = accuracy_score(y_test,svm_y_predict)
rs = recall_score(y_test,svm_y_predict)
ps = precision_score(y_test,svm_y_predict)

print(cm)
print(ac)
print(rs)
print(ps)

# %%
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

dt_y_predict = dt_classifier.predict(X_test)
dt_y_predict

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score
cm = confusion_matrix(y_test, dt_y_predict)
ac = accuracy_score(y_test,dt_y_predict)
rs = recall_score(y_test,dt_y_predict)
ps = precision_score(y_test,dt_y_predict)

print(cm)
print(ac)
print(rs)
print(ps)

# %%
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)

gb_y_predict = gb_classifier.predict(X_test)
gb_y_predict

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score
cm = confusion_matrix(y_test, gb_y_predict)
ac = accuracy_score(y_test,gb_y_predict)
rs = recall_score(y_test,gb_y_predict)
ps = precision_score(y_test,gb_y_predict)

print(cm)
print(ac)
print(rs)
print(ps)

# %% [markdown]
# k-fold validation

# %%
from sklearn.model_selection import cross_val_score

# %%
cross_val_score(GaussianNB(), X, y)

# %%
cross_val_score(SVC(), X, y)

# %%
cross_val_score(DecisionTreeClassifier(), X, y)

# %%
cross_val_score(GradientBoostingClassifier(), X, y)

# %%
X

# %%