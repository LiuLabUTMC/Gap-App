
#Libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
import numpy as np
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


#check the current work directory
os.getcwd()

#importing the excel datasheet
all_data = pd.read_excel("Pancreatic_dataset after initial gene filtration.xlsx", 
                          sheet_name = 'Sheet1')

all_data

all_data = all_data.drop(['submitter_id.samples','OS','OS.time'],axis=1)
all_data


X = all_data.drop(['lymph_node_examined_count','final_censor'], axis=1)
y = all_data['final_censor']


X.columns

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

#Here make sure to use the appropriate model that you find best from metrics calculated earlier for initial models.
#This feature importance will work for both Random Forest and AdaBoost models.

cv_output=cross_validate(RandomForestClassifier("Use the best parameters obtained after optimization"),
                       X_new,y,cv=cv,scoring='accuracy',
                                                return_estimator=True)


feature_scores_df = pd.DataFrame(columns=['importance'])

 
for idx,estimator in enumerate(cv_output['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index = X_new.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)   
    separator = pd.DataFrame({'importance': ['Separating the folds']})
    feature_importances=pd.concat([feature_importances,separator])
    feature_scores_df = pd.concat([feature_scores_df,feature_importances])
    print(feature_importances)


filtered_feature_scores_df = feature_scores_df[feature_scores_df['importance'] != 0]

filtered_feature_scores_df.to_excel('Feature scores with CV.xlsx',
                           sheet_name='RandomForest',index=True)


