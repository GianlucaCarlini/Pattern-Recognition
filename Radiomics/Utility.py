# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:17:38 2020

@author: Gianluca
"""

import pandas as pd
import numpy as np

def PreProcessing(path):
    """
    

    Parameters
    ----------
    path : String
        The path to the dataframe.

    Returns
    -------
    df : pandas dataframe
        The processed dataframe.

    """
    
    
    df = pd.read_csv(path)
    df = df[df['Feature Class'] != 'info']
    df.reset_index(drop = True, inplace = True)      
    df['pz'].fillna(value = 0, inplace = True)
    df = df.drop(df[df['pz'] <= 0].index)
    df.fillna(value = 0, inplace = True)
    
    #There are some patients with a number of features different from 105. We are going to drop them
    bad_ones = [18, 19, 21, 24, 39]
        
    for i in range(len(bad_ones)):
        df = df.drop(df[df['pz'] == bad_ones[i]].index)
        
    return df
        
def Features_Extraction(df_features, df_labels, divide = False):
    """
    

    Parameters
    ----------
    df_features : Dataframe
        The dataframe containing the features.
    df_labels : Dataframe
        The outcome dataframe, from which the labels will be extracted.
    divide : Bool, optional
        If True, the features will be splitted in PET and CT, if False they will not.
        Default: False.

    Returns
    -------
    numpy array
        If divide is True the function returns 4 arrays, 2 for CT and PET features and
        2 for CT and PET labels.
        If divide is False the function returns 2 arrays, features and labels.

    """
    
    df_labels_array = df_labels[['Pt', 'Recurrence']].to_numpy()
    
    if divide:
        
        
        CT = df_features['img'] == 'CT'
        PET = df_features['img'] == 'PET'  
        df_features_CT = df_features[CT]
        df_features_PET = df_features[PET]
            
        group_CT = df_features_CT.groupby('pz').count()/105
        group_CT.reset_index(level = 0, inplace = True)
        Patients_Images_CT = group_CT[['pz', 'img']].to_numpy()
        
        group_PET = df_features_PET.groupby('pz').count()/105
        group_PET.reset_index(level = 0, inplace = True)
        Patients_Images_PET = group_PET[['pz', 'img']].to_numpy()
                
        features_CT = df_features_CT['Value'].tolist()
        features_PET = df_features_PET['Value'].tolist()
        
        features_CT = np.array(features_CT).reshape(-1, 105)
        features_CT = features_CT.astype(float)
        features_PET = np.array(features_PET).reshape(-1, 105)
        features_PET = features_PET.astype(float)
                
        labels_CT = []
        labels_PET = []
        
        for i in range(len(Patients_Images_CT)):
            for j in range(int(Patients_Images_CT[i][1])):
                labels_CT.append(df_labels_array[int(Patients_Images_CT[i][0] - 1)][1])
            
        labels_CT = np.array(labels_CT).reshape(-1, 1)
        labels_CT = labels_CT.astype(float)
        
        for i in range(len(Patients_Images_PET)):
            for j in range(int(Patients_Images_PET[i][1])):
                labels_PET.append(df_labels_array[int(Patients_Images_PET[i][0] - 1)][1])
            
        labels_PET = np.array(labels_PET).reshape(-1, 1)
        labels_PET = labels_PET.astype(float)
        
        return features_CT, features_PET, labels_CT, labels_PET
        
        
    else:
        
        grouped = df_features.groupby('pz').count()/105
        grouped.reset_index(level = 0, inplace = True)
        Patients_Images = grouped[['pz', 'img']].to_numpy()
        
        features = df_features['Value'].tolist()
        features = np.array(features).reshape(-1, 105)
        features = features.astype(float)
        
        labels = []
        
        for i in range(len(Patients_Images)):
            for j in range(int(Patients_Images[i][1])):
                labels.append(df_labels_array[int(Patients_Images[i][0] - 1)][1])
            
        labels = np.array(labels).reshape(-1, 1)
        
        return features, labels
        
        
        
        
        
        
        
        
                        