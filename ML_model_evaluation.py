

#Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import flask
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


#importing the excel datasheet
train_data = pd.read_excel("Final pancreratic dataset after feature selection.xlsx", 
                          sheet_name = 'Sheet1')

train_data


train_data = train_data.drop(['submitter_id.samples','OS','OS.time'],axis=1)
train_data


X = train_data.drop(['lymph_node_examined_count','final_censor'], axis=1)
y = train_data['final_censor']

X_new = X.drop(['Gender', 'race.demographic', 'ethnicity.demographic', 'Age',
       'Tumor_Stage', 'Diabetes_history', 'surgery_performed_type'], axis=1)


cols = X_new.columns.values
new_cols= [re.split(r'[.:]',item)[0] for item in cols]
X_new.columns = new_cols

#splitting into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42, stratify=y)


ind_test_data = pd.read_csv("Independent test dataset.csv")

X_test_unseen=ind_test_data.drop(['submitter_id.samples','Gender', 'Age',
       'Race','OS','OS.time','final_censor'], axis=1)

y_test_unseen=ind_test_data['final_censor']


common_columns = X_new.columns.intersection(X_test_unseen.columns)

missing_cols = set(X_new.columns) - set(X_test_unseen.columns)
if missing_cols:
    print(f"Missing columns in test dataset: {missing_cols}")

for col in missing_cols:
    X_test_unseen[col] = 0

X_test_unseen = X_test_unseen[X_new.columns]



#Building the optimized models
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)


#Random forest model
#Building the model
random_forest = RandomForestClassifier("Use the best parameters obtained after optimization for respective models")

random_forest.fit(X_train, y_train)


#calculate training accuracy
y_pred=random_forest.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")


#calculate test accuracy
y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


#Calculate unseen data accuracy
y_pred = random_forest.predict(X_test_unseen)

accuracy = accuracy_score(y_test_unseen, y_pred)
print(f"Accuracy on unseen test data: {accuracy:.4f}")

precision = precision_score(y_test_unseen, y_pred)
recall = recall_score(y_test_unseen, y_pred)
f1 = f1_score(y_test_unseen, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")



#Unseen data AUC
y_prob = random_forest.predict_proba(X_test_unseen)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test_unseen, y_prob)

roc_auc = auc(fpr, tpr)
print("AUC Score for unseen test dataset: ", roc_auc)

brier_score = brier_score_loss(y_test_unseen, y_prob)
print(f"Brier Score (lower is better): {brier_score:.4f}")

#Plotting ROC curve
plt.figure(figsize=(9, 9))

plt.plot(fpr, tpr, alpha=1, lw=2, 
         label=f'RandomForest, AUC={roc_auc:.2f}')


###############
#AdaBoost model
###############

#Building the model
ada_boost = AdaBoostClassifier("Use the best parameters obtained after optimization for respective models")

ada_boost.fit(X_train, y_train)

#calculate training accuracy
y_pred=ada_boost.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")


#calculate test accuracy
y_pred = ada_boost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


#Calculate unseen data accuracy
y_pred = ada_boost.predict(X_test_unseen)

accuracy = accuracy_score(y_test_unseen, y_pred)
print(f"Accuracy on unseen test data: {accuracy:.4f}")

precision = precision_score(y_test_unseen, y_pred)
recall = recall_score(y_test_unseen, y_pred)
f1 = f1_score(y_test_unseen, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#Unseen data AUC
# Predict probabilities
y_prob = ada_boost.predict_proba(X_test_unseen)[:, 1] 
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_unseen, y_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

print("AUC Score for unseen test dataset: ", roc_auc)

brier_score = brier_score_loss(y_test_unseen, y_prob)
print(f"Brier Score (lower is better): {brier_score:.4f}")

plt.plot(fpr, tpr, alpha=1, lw=2, 
         label=f'AdaBoost, AUC={roc_auc:.2f}')




####################
#Decision Tree model
####################

#Building the model
decision_tree = DecisionTreeClassifier("Use the best parameters obtained after optimization for respective models")

decision_tree.fit(X_train, y_train)

#calculate training accuracy
y_pred=decision_tree.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")


#calculate test accuracy
y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


#Calculate unseen data accuracy
y_pred = decision_tree.predict(X_test_unseen)

accuracy = accuracy_score(y_test_unseen, y_pred)
print(f"Accuracy on unseen test data: {accuracy:.4f}")

precision = precision_score(y_test_unseen, y_pred)
recall = recall_score(y_test_unseen, y_pred)
f1 = f1_score(y_test_unseen, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#Unseen data AUC
# Predict probabilities
y_prob = decision_tree.predict_proba(X_test_unseen)[:, 1] 
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_unseen, y_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

print("AUC Score for unseen test dataset: ", roc_auc)

brier_score = brier_score_loss(y_test_unseen, y_prob)
print(f"Brier Score (lower is better): {brier_score:.4f}")

plt.plot(fpr, tpr, alpha=1, lw=2, 
         label=f'DecisionTree, AUC={roc_auc:.2f}')



##########################
#Logistic Regression model
##########################

#Building the model
log_reg = LogisticRegression("Use the best parameters obtained after optimization for respective models")

log_reg.fit(X_train, y_train)

#calculate training accuracy
y_pred=log_reg.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")


#calculate test accuracy
y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


#Calculate unseen data accuracy
y_pred = log_reg.predict(X_test_unseen)

accuracy = accuracy_score(y_test_unseen, y_pred)
print(f"Accuracy on unseen test data: {accuracy:.4f}")

precision = precision_score(y_test_unseen, y_pred)
recall = recall_score(y_test_unseen, y_pred)
f1 = f1_score(y_test_unseen, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#Unseen data AUC
# Predict probabilities
y_prob = log_reg.predict_proba(X_test_unseen)[:, 1] 
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_unseen, y_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

print("AUC Score for unseen test dataset: ", roc_auc)

brier_score = brier_score_loss(y_test_unseen, y_prob)
print(f"Brier Score (lower is better): {brier_score:.4f}")

plt.plot(fpr, tpr, alpha=1, lw=2, 
         label=f'LogisticRegression, AUC={roc_auc:.2f}')



#################
#Linear SVC model
#################

#Building the model
linear_SVC = LinearSVC("Use the best parameters obtained after optimization for respective models")

linear_SVC.fit(X_train, y_train)

#calculate training accuracy
y_pred=linear_SVC.predict(X_train)

accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data: {accuracy:.4f}")


#calculate test accuracy
y_pred = linear_SVC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")


#Calculate unseen data accuracy
y_pred = linear_SVC.predict(X_test_unseen)

accuracy = accuracy_score(y_test_unseen, y_pred)
print(f"Accuracy on unseen test data: {accuracy:.4f}")

precision = precision_score(y_test_unseen, y_pred)
recall = recall_score(y_test_unseen, y_pred)
f1 = f1_score(y_test_unseen, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#Unseen data AUC
# Predict probabilities
y_prob = linear_SVC._predict_proba_lr(X_test_unseen)[:, 1] 
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_unseen, y_prob)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

print("AUC Score for unseen test dataset: ", roc_auc)

brier_score = brier_score_loss(y_test_unseen, y_prob)
print(f"Brier Score (lower is better): {brier_score:.4f}")

plt.plot(fpr, tpr, alpha=1, lw=2, 
         label=f'LinearSVC, AUC={roc_auc:.2f}')


plt.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'black', alpha= 1) 
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate', fontsize=22, weight='bold')
plt.xlabel('False Positive Rate', fontsize=22, weight='bold')
plt.xticks(fontsize = 22, weight='bold')
plt.yticks(fontsize = 22, weight='bold')
plt.title(' ')
plt.legend(loc="lower right", prop = {'size' : 14, 'weight' : 'bold'})
plt.savefig('ROC curve for unseen test data.png')
plt.show()


