# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:18:10 2020

@author: Gianluca
"""


import numpy as np
import pandas as pd
from Radiomics.Utility import PreProcessing, Features_Extraction
from Radiomics.Analysis import Remove_Correlated, Lasso_Selection, Forest_Selection
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


path = r'C:\Users\Gianluca\Desktop\Gianluca\Pattern_Recognition\Radiomics_LS_SS.csv'

df_features = PreProcessing(path)
df_labels = pd.read_csv(r'C:\Users\Gianluca\Desktop\Gianluca\Pattern_Recognition\Radiomics_outcome.csv')

features_CT, features_PET, labels_CT, labels_PET = Features_Extraction(df_features,
                                                                       df_labels, divide = True)

for i in range(len(labels_PET)):
    if labels_PET[i] > 1:
        labels_PET[i] = 1

features_PET, correlated_PET = Remove_Correlated(df_features, features_PET, verbose = True)

features_PET = Lasso_Selection(features_PET, labels_PET)

features_array = features_PET.to_numpy()

features_forest, idx = Forest_Selection(features_array, labels_PET,
                                                            how_many = 15)
        
X_train, X_test, Y_train, Y_test = train_test_split(features_forest, labels_PET, test_size = 0.3,
                                                    random_state = 7)


count0, count1 = 0, 0
for i in range(len(Y_test)):
    if Y_test[i] == 0:
        count0 = count0 + 1
    else:
        count1 = count1 + 1

weights = {0: 1, 1: 1*(count0/count1)}

scaler = StandardScaler()
scaler.fit(X_train)

forest = RandomForestClassifier()


param_grid = {'n_estimators': [10, 100, 200, 300, 500], 'max_features':[3, 5, 7, 9, 12],
                'max_leaf_nodes': [16, 32, 64, 128]}

grid_search = GridSearchCV(forest, param_grid, cv = 3, scoring = 'f1',
                            return_train_score = True)
grid_search.fit(scaler.transform(X_train), Y_train.ravel())

best_params = grid_search.best_params_

forest = RandomForestClassifier(max_features = 7, max_leaf_nodes = 128, n_estimators = 200)

forest.fit(scaler.transform(X_train), Y_train.ravel())

y_pred_forest = forest.predict(scaler.transform(X_test))


precision_forest = precision_score(Y_test, y_pred_forest)
recall_forest = recall_score(Y_test, y_pred_forest)

predicted_proba = forest.predict_proba(scaler.transform(X_test))
scores = predicted_proba[:, 1]

precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(Y_test, scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    [...] # highlight the threshold and add the legend, axis label, and grid

plot_precision_recall_vs_threshold(precisions_forest, recalls_forest, thresholds_forest)
plt.show()

threshold = 0.74

predicted = (predicted_proba[:, 1] >= threshold).astype('int')

precision_threshold = precision_score(Y_test, predicted)
recall_threshold = recall_score(Y_test, predicted)


fpr, tpr, thresholds = roc_curve(Y_test, scores)

def plot_roc_curve(fpr, tpr, score):
    
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    
    textstr = ('ROC AUC = ' + str(score)[:6])
    
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(b = True)
    plt.xlabel('False Positive Rate', fontsize = 'x-large')
    plt.ylabel('True Positive Rate', fontsize = 'x-large')
    plt.title('PET Images ROC curve', fontsize = 'x-large')
    plt.text(0.05, 0.95, textstr, fontsize = 14, verticalalignment='top', bbox=props)


roc = roc_auc_score(Y_test, scores)

plot_roc_curve(fpr, tpr, roc)
plt.show()


f1_score(Y_test, y_pred_forest)
