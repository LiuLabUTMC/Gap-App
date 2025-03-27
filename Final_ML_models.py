
#Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import flask
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV


final_data = pd.read_excel("Final pancreratic dataset after feature selection.xlsx", 
                          sheet_name = 'Sheet1')

final_data


final_data = final_data.drop(['submitter_id.samples','OS','OS.time'],axis=1)
final_data


X = final_data.drop(['lymph_node_examined_count','final_censor'], axis=1)
y = final_data['final_censor']


X_new = X.drop(['Gender', 'race.demographic', 'ethnicity.demographic', 'Age',
       'Tumor_Stage', 'Diabetes_history', 'surgery_performed_type'], axis=1)

cols = X_new.columns.values
new_cols= [re.split(r'[.:]',item)[0] for item in cols]
X_new.columns = new_cols


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

np.mean(cross_val_score(RandomForestClassifier("Use the best parameters obtained after optimization for respective models"),
                       X_new,y,cv=cv,scoring='accuracy'))



random_forest = RandomForestClassifier("Use the best parameters obtained after optimization for respective models")
calibrated_rf = CalibratedClassifierCV(random_forest, method='sigmoid', cv=5)
calibrated_rf.fit(X_new, y)

pickle.dump(calibrated_rf, open('Final prediction model.pkl', 'wb'))


