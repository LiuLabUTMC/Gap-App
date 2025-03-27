

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss



#importing the excel datasheet
all_data = pd.read_excel("Please use the appropriate dataset for initial models or refined models accordingly.xlsx", 
                          sheet_name = 'Sheet1')

all_data


all_data = all_data.drop(['submitter_id.samples','OS','OS.time'],axis=1)
all_data

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


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)


#Using Random Forest Classifier

#optimizing the random forest model
criterion = ['gini', 'entropy', 'log_loss']
n_estimators = [25, 50, 75, 100, 125, 150]
max_features = ['sqrt', 'log2', None]
max_depth = [2, 3, 6, 9, 12]
max_leaf_nodes = [2, 3, 6, 9, 12]
max_samples = [0.20,0.25,0.40,0.5,0.60,0.75,0.80]

param_grid = {
    'criterion': criterion,
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'max_leaf_nodes': max_leaf_nodes,
    'max_samples' : max_samples
}


grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1
                          )
grid_search.fit(X_new, y)


print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#scores after optimization
np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy'))

scores=cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='roc_auc'))


np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision'))

scores=cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall'))

scores=cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1'))

scores=cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='neg_brier_score'))



#Using AdaBoost Classifier

#model optimization
n_estimators = [10, 25, 50, 75, 100]
learning_rate = np.logspace(-2,2,30)
algorithm = ['SAMME', 'SAMME.R']

param_grid = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'algorithm': algorithm
}


grid_search = GridSearchCV(AdaBoostClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1
                          )
grid_search.fit(X_new, y)


print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#scores after optimization

#after changing the cv in gridsearch
np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='accuracy'))

scores=cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='accuracy')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='roc_auc'))


np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision'))

scores=cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='precision')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall'))

scores=cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='recall')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1'))

scores=cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='f1')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(AdaBoostClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='neg_brier_score'))


#Using Decision tree classifier

#model_optimization

max_features = ['sqrt', 'log2', None]
max_depth = [3, 6, 9, 12]
criterion = ['gini', 'entropy', 'log_loss']
min_samples_leaf = [1,2,3,4,5]

param_grid = {
    'criterion': criterion,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_leaf' : min_samples_leaf
}


grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1
                          )
grid_search.fit(X_new, y)


print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#scores after optimization

np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='accuracy'))

scores=cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='roc_auc'))


np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision'))

scores=cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall'))

scores=cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1'))

scores=cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(DecisionTreeClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='neg_brier_score'))


#Logistic Regression classifier

#model optimization

param_grid=[
    {'penalty' : ['l1','l2','elasticnet','none'],
     'C' : np.logspace(-2,2,30),
     'solver' : ['lbfgs', 'newton-cg','liblinear','saga']
     }
]

grid_search = GridSearchCV(LogisticRegression(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1
                          )
grid_search.fit(X_new, y)


print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#scores after optimization

np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='accuracy'))


scores=cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),X_new,y,cv=cv,scoring='roc_auc'))


np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='precision'))

scores=cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='recall'))

scores=cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='f1'))

scores=cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LogisticRegression("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='neg_brier_score'))


#LinearSVC classifier

#model_optimization
param_grid=[
    {'penalty' : ['l1','l2'],
     'loss' : ['hinge','squared_hinge'],
     'C' : np.logspace(-2,2,30)
     }
]


#from sklearn.model_selection import HalvingGridSearchCV

grid_search = GridSearchCV(LinearSVC(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1
                          )
grid_search.fit(X_new, y)


print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


#scores after optimization

np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='accuracy'))

scores=cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy')

print(scores)
print(min(scores))
print(max(scores))


#np.mean(cross_val_score(LinearSVC(C=0.01, random_state=42),X_new_scaled,y,cv=cv,scoring='roc_auc'))

np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='roc_auc'))



np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='precision'))

scores=cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='precision')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='recall'))

scores=cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='recall')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                        X_new,y,cv=cv,scoring='f1'))

scores=cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='f1')

print(scores)
print(min(scores))
print(max(scores))


np.mean(cross_val_score(LinearSVC("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='neg_brier_score'))



