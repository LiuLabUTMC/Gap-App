

#Libraries
import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from itertools import cycle
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


#check the current work directory
os.getcwd()

#importing the excel datasheet
all_data = pd.read_excel("Pancreatic_dataset after initial gene filtration.xlsx", 
                          sheet_name = 'Sheet1')

all_data


all_data = all_data.drop(['submitter_id.samples','OS','OS.time'],axis=1)

X = all_data.drop(['lymph_node_examined_count','final_censor'], axis=1)
y = all_data['final_censor']


#missing values imputing for features
X["Gender"].fillna("not reported", inplace = True)
X["race.demographic"].fillna("not reported", inplace = True)
X["ethnicity.demographic"].fillna("not reported", inplace = True)
X["Tumor_Stage"].fillna("not reported", inplace = True)
X["Diabetes_history"].fillna("not reported", inplace = True)
X["surgery_performed_type"].fillna("not reported", inplace = True)


#label encode categorical features
label_encoder = LabelEncoder()
category_columns = ['Gender', 'race.demographic', 'ethnicity.demographic',
                    'Tumor_Stage', 'Diabetes_history',
                    'surgery_performed_type']

mapping={}

for cols in category_columns:
    X[cols] = label_encoder.fit_transform(X[cols])
    le_mapping = dict(zip(label_encoder.classes_,
                         label_encoder.transform(label_encoder.classes_)))
    mapping[cols]=le_mapping

print(mapping)


X

X_new = X.drop(['Gender', 'race.demographic', 'ethnicity.demographic', 'Age',
       'Tumor_Stage', 'Diabetes_history', 'surgery_performed_type'], axis=1)


X_new = X_new.to_numpy()
y = y.to_numpy()


#Random Forest model

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']
plt.figure(figsize=(9, 9))

for i, (train, test) in enumerate(kf.split(X_new,y)):
    model = RandomForestClassifier(max_depth=2, max_leaf_nodes=2, max_samples=0.25,
                       n_estimators=75, random_state=42)
    model.fit(X_new[train], y[train])
    y_pred = model.predict_proba(X_new[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, lw = 2, alpha = 1, label=r'RandomForest, AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),)


#AdaBosst classifier
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

for i, (train, test) in enumerate(kf.split(X_new,y)):
    model = AdaBoostClassifier(learning_rate=1.2742749857031335, n_estimators=100,
                   random_state=42)
    model.fit(X_new[train], y[train])
    y_pred = model.predict_proba(X_new[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, lw = 2, alpha = 1, label=r'AdaBoost, AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),)


#Decision Tree
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

for i, (train, test) in enumerate(kf.split(X_new,y)):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features='sqrt',
                       min_samples_leaf=3, random_state=42)
    model.fit(X_new[train], y[train])
    y_pred = model.predict_proba(X_new[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, lw = 2, alpha = 1, label=r'DecisionTree, AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),)


#Logistic Regression
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

for i, (train, test) in enumerate(kf.split(X_new,y)):
    model = LogisticRegression(C=0.01, random_state=42, max_iter=2000)
    model.fit(X_new[train], y[train])
    y_pred = model.predict_proba(X_new[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, lw = 2, alpha = 1, label=r'LogisticRegression, AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),)


#Linear SVC
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

for i, (train, test) in enumerate(kf.split(X_new,y)):
    model = LinearSVC(C=0.01, random_state=42, dual='auto', max_iter=3000)
    model.fit(X_new[train], y[train])
    y_pred = model._predict_proba_lr(X_new[test])[:,1]
    fpr, tpr, _ = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, lw = 2, alpha = 1, label=r'LinearSVC, AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),)


plt.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'black', alpha= 1)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate', fontsize=22, weight='bold')
plt.xlabel('False Positive Rate', fontsize=22, weight='bold')
plt.xticks(fontsize = 22, weight='bold')
plt.yticks(fontsize = 22, weight='bold')
plt.legend(loc="lower right", prop = {'size' : 14, 'weight' : 'bold'})
plt.title(' ')
plt.savefig('5-fold CV ROC.png')
plt.show()


