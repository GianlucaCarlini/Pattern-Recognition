# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:16:56 2020

@author: Gianluca
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def Remove_Correlated(df, features_array, threshold = 0.85, verbose = False):
    """
    

    Parameters
    ----------
    df : Dataframe
        The dataframe (without info) from which the features names are extracted.
    features_array : numpy array
        The array of features values.
    threshold : float, optional
        The correlation value beyond which two features are considered correlated.
        The default value is 0.85.
    verbose : Bool, optional
        If true, the function returns the list of the correlated features; if false
        it will not.
        The default value is False

    Returns
    -------
    pandas dataframe, list(optional)
        If verbose is false, the function returns the dataframe containing only the
        uncorrelated features.
        If verbose is true, the function returns the dataframe containing only the
        uncorrelated features and a list of the correlated features
        
    """
    
    feature_names = df['Feature Name'].head(105)
    feature_names = feature_names.tolist()
            
    """
    Some columns have the same name
    In order to make a correlation study and remove the correlated features we want
    them to have different names
    """
    
    for i in range(int(len(feature_names))):
        count = 1
        for j in range((i + 1), len(feature_names)):
            if feature_names[i] == feature_names[j]:
                feature_names[j] = feature_names[j] + str(count)
                count = count + 1
             
    features = pd.DataFrame(features_array)
    features.columns = feature_names
          
    """
    Correlation
    """
    
    correlation_matrix = features.corr()
    
    correlated_features = set()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    correlated_features = list(correlated_features)
    
    features = features.drop(correlated_features, axis = 1)
    
    
    if verbose:
        return features, correlated_features
    else:
        return features
    

def Lasso_Selection(features, labels_array):
    """
    

    Parameters
    ----------
    features : dataframe
        The dataframe containing features.
    labels_array : numpy array
        the array containing labels.

    Returns
    -------
    selected_features : dataframe
        The dataframe containing only the features survived to the LASSO regression

    """
    
    features_array = features.to_numpy()
    features_array = features_array.astype(float)
    
    #scaling the features
    scaler = StandardScaler()
    scaler.fit(features_array)
    
    #selecting features
    sel_ = SelectFromModel(LogisticRegression(penalty = 'l1', solver = 'liblinear',
                                              max_iter = 1000))
    sel_.fit(scaler.transform(features_array), labels_array.ravel())
    
        
    #vector of selected features
    bool_features = sel_.get_support()
    
    selected_features = features.loc[:, bool_features]
    
    return selected_features
    

def Forest_Selection(features, labels, how_many = 15, trees  = 100, leafs = 100):
    """
    

    Parameters
    ----------
    features : numpy array
        The array of features.
    labels : numpy array
        The array of labels.
    how_many : int, optional
        The number of important features to select. The default is 15.
    trees : int, optional
        The number of trees . The default is 100.
    leafs : int, optional
        The maximum number of leafs per tree. The default is 100.

    Returns
    -------
    meaningful_features : numpy array
        The array with the selected features only.
    idx : numpy array
        The index of the selected features.

    """
    
    scaler = StandardScaler()
    scaler.fit(features)
    
    rfc = RandomForestClassifier(n_estimators = trees, max_leaf_nodes = leafs)
    rfc.fit(scaler.transform(features), labels.ravel())
    rfc.fit(scaler.transform(features), labels.ravel())
        
    important_features = rfc.feature_importances_
       
    idx = (-important_features).argsort()[:how_many]
    idx = np.sort(idx)
                
    meaningful_features = features[:, idx]
    
    return meaningful_features, idx 
    
    

    
        

    
    
    
    
    
