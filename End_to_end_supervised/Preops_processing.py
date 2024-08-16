"""
Preprocessing module for preops
1) Does one hot encoding
2) Normalization
3) Imputation
4) Separate functions for training and inference time

Outputs:
1) returns train, val, test partition of the preops and the corresponding indices
2) saves the meta data like values used for normalization
"""

import json
import os
import sys, argparse
import glob
import pickle
import numpy as np
import pandas as pd
import math
import pdb
from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
from datetime import datetime
import pickle
from scipy.stats import qmc

def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()
    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std
    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v
    return data

def preprocess_train(preops,task, y_outcome=None, binary_outcome=False, test_size=0.2, valid_size=0.1,random_state=101, input_dr=None, output_dr=None):
    if 'orlogid_encoded' in preops.columns:  # this is being done to have generality when this code is ran on its own and the data will have orlogids in comparison to when this function will be called fomr the model training file
        preops.rename({"orlogid_encoded": "person_integer"}, axis=1, inplace=True)

    preops.reset_index(drop=True, inplace=True)

    preops_mask = preops.copy()

    lab_cats = pd.read_csv(input_dr+'mapping_info/categories_labs.csv')

    ordinal_variables = list(pd.read_csv(input_dr+'mapping_info/ordinal_vars.txt', delimiter= "\t",header=None)[0])

    preop_labs_categorical = lab_cats[lab_cats['all_numeric'] == 0.0]['LAB_TEST'].unique()
    num_lab_cats = [i for i in lab_cats['LAB_TEST'].unique() if
                    (i in preops.columns) and (i not in preop_labs_categorical) and (i not in ordinal_variables)]
    # for thisvar in data.keys():
    #     transform_dict = dict(zip(list(data[thisvar].keys()), np.arange(len(data[thisvar].keys())) + 1))
    #     labs_non_numeric.loc[:, thisvar] = labs_non_numeric.loc[:, thisvar].map(transform_dict)

    ordinal_variables = [i for i in ordinal_variables if (i in preops.columns)]
    # ordinal_variables = ['LVEF', 'ASA', 'CHF_class', 'FunctionalCapacity', 'DiastolicFunction']
    # ordinal_variables = ['LVEF', 'ASA', 'Pain Score', 'FunctionalCapacity', 'TOBACCO_USE', 'AR', 'AS']

    # making sure that Sex variable has 0 and 1 values instead of 1 and 2
    preops.loc[preops['Sex'] == 1, 'Sex'] = 0
    preops.loc[preops['Sex'] == 2, 'Sex'] = 1


    # encoding the plannedDispo from text to number
    # {"OUTPATIENT": 0, '23 HOUR ADMIT': 1, "FLOOR": 1, "OBS. UNIT": 2, "ICU": 3}
    preops.loc[preops['plannedDispo'] == 'Outpatient', 'plannedDispo'] = 0
    preops.loc[preops['plannedDispo'] == 'Floor', 'plannedDispo'] = 1
    preops.loc[preops['plannedDispo'] == 'Obs. unit', 'plannedDispo'] = 2
    preops.loc[preops['plannedDispo'] == 'ICU', 'plannedDispo'] = 3
    if '' in list(preops['plannedDispo'].unique()):
        preops.loc[preops['plannedDispo'] == '', 'plannedDispo'] = np.nan
    preops['plannedDispo'] = preops['plannedDispo'].astype('float') # needed to convert this to float because the nans were not getting converted to int and this variable is object type


    categorical_variables = [i for i in preop_labs_categorical if i in preops.columns] + ['Sex', 'PlannedAnesthesia']
    binary_variables = []

    # continuous_variables = ['Secondary Diagnosis']
    continuous_variables = []

    for i in num_lab_cats:
        if len(preops[i].value_counts().index) == 2 and (preops[i].value_counts(dropna=False).iloc[0] / len(
                preops) > 0.2):  # checking for missingness as the first element of valuecounts is the missing flag of nan
            categorical_variables.append(i)
        elif len(preops[i].value_counts().index) == 2 and (
                preops[i].value_counts(dropna=False).iloc[0] / len(preops) < 0.2):
            preops_mask[i] = (preops_mask[i].notnull()).astype('int')
            preops[i].fillna(0, inplace=True)
            binary_variables.append(i)
        elif len(preops[i].value_counts(
                dropna=False).index) == 2:  # the variables that are reported only when present ([NA, value] form) can be traansformed to binary
            preops_mask[i] = (preops_mask[i].notnull()).astype('int')
            preops[i].fillna(0, inplace=True)
            binary_variables.append(i)
        elif len(preops[i].unique()) > 2 and len(preops[i].value_counts().index) < 10:  # for the variables that have more than 2 categories
            categorical_variables.append(i)
        elif len(preops[i].value_counts().index) > 10:
            continuous_variables.append(i)
        else:
            ordinal_variables.append(i)

    for a in preops.columns:
        if preops[a].dtype == 'bool':
            preops[a] = preops[a].astype('int32')
        if preops[a].dtype == 'int32' or preops[a].dtype == 'int64':
            if len(preops[a].unique()) < 10 and len(preops[a].unique()) > 2 and (a not in ordinal_variables):
                preops[a] = preops[a].astype('category')
                categorical_variables.append(a)
        if len(preops[a].unique()) <= 2 and (a not in ordinal_variables+binary_variables+categorical_variables+continuous_variables):
            binary_variables.append(a)
        # this change in dtype is not possible because there are missing values.
        # Not changing the dtype is not affecting anything down the line because the imputation for ordinal variables is anyway done seperately.
        # if a in ordinal_variables:
        #   preops[a] = preops[a].astype('int32')
        if preops[a].dtype == 'O' and (a not in ordinal_variables+binary_variables+categorical_variables+continuous_variables):
            preops[a] = preops[a].astype('category')
            categorical_variables.append(a)

    # following inf is more or less hardcoded based on how the data was at the training time.
    categorical_variables.append('SurgService_Name')
    # preops['SurgService_Name'].replace(['NULL', ''], [np.NaN, np.NaN], inplace=True)

    dif_dtype = [a for a in preops.columns if preops[a].dtype not in ['int32', 'int64', 'float64',
                                                                      'category', 'O']]  # columns with non-numeric datatype; this was used at the code development time
    for a in dif_dtype:
        preops[a] = preops[a].astype('category')
        categorical_variables.append(a)

    # this is kind of hardcoded; check your data beforehand for this; fixed this
    temp_list = [i for i in preops['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in preops[
        'PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list != []:
        preops['PlannedAnesthesia'].replace(temp_list, np.NaN,
                                            inplace=True)  # this is done because there were two values for missing token (nan anf -inf)
    categorical_variables.append('PlannedAnesthesia')

    # remove if there are any duplicates in any of the variable name lists
    categorical_variables = [*set(categorical_variables)]

    continuous_variables = continuous_variables + [i for i in preops.columns if
                                                   i not in (
                                                               binary_variables + categorical_variables + ordinal_variables)]
    continuous_variables = [*set(continuous_variables)]

    continuous_variables.remove(
        'person_integer')  # this var went into categorical due to its initial datatype of being an object

    # masking operation; need not worry about bow_na variable as it is already a mask
    preops_mask[continuous_variables] = (preops_mask[continuous_variables].notnull()).astype('int')
    preops_mask[ordinal_variables] = (preops_mask[ordinal_variables].notnull()).astype('int')

    # since the categorical labs were float type earlier
    for name in categorical_variables:
        preops[name] = preops[name].astype('category')

    # one hot encoding
    meta_Data = {}

    meta_Data["levels"] = {}

    preops_ohe = preops.copy()
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    import itertools
    encoded_variables = list()
    for i in categorical_variables:
        meta_Data["levels"][i] = list(preops[i].cat.categories)
        temp = pd.get_dummies(preops[i], dummy_na=True, prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))

    # masking operation for categorical observations: make everything zero except for the null column, that is supposed to be as it is
    preops_mask.drop(columns=categorical_variables, inplace=True)
    preops_mask[encoded_variables] = preops_ohe[encoded_variables].copy()
    encoded_variables_null = [name for name in encoded_variables if name.split("_")[-1] == 'nan']
    preops_mask[encoded_variables_null] = 0
    preops_mask.drop(columns='person_integer', inplace=True)
    for name in preops_mask.columns:
        preops_mask[name] = preops_mask[name].astype('int')


    # partitioning the data into train, valid and test
    if False:
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train0, test = train_test_split(preops_ohe, test_size=test_size, random_state=random_state,
                                            stratify=y_outcome)
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state,
                                            stratify=y_outcome[train0.index])
        else:
            train0, test = train_test_split(preops_ohe, test_size=test_size, random_state=random_state)
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state)
    if True:
        upto_test_idx = int(test_size * len(preops_ohe))
        test = preops_ohe.iloc[:upto_test_idx]
        train0 = preops_ohe.iloc[upto_test_idx:]
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state,
                                            stratify=y_outcome[train0.index])
        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=random_state)

    train_index = train.index
    valid_index = valid.index
    test_index = test.index

    # print( list(train.columns) )
    train.drop(columns="person_integer", inplace=True)
    valid.drop(columns="person_integer", inplace=True)
    test.drop(columns="person_integer", inplace=True)

    # mean imputing and scaling the continuous variables
    train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    valid[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or valid[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # median Imputing_ordinal variables

    # imputing
    for i in ordinal_variables:
        if np.isnan(preops[i].unique().min()) == True:
            train[i].replace(train[i].unique().min(), train[i].median(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].median(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].median(), inplace=True)

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord

    if (sum(valid.isna().any()) > 0) or (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["encoded_var"] = encoded_variables

    meta_Data["binary_var_name"] = binary_variables

    meta_Data["categorical_name"] = categorical_variables
    meta_Data["ordinal_variables"] = ordinal_variables
    meta_Data["continuous_variables"] = continuous_variables
    meta_Data["column_all_names"] = list(preops_ohe.columns)

    # meta_Data["bow"] = bow_cols

    output_file_name = output_dr + 'preops_metadata_' + str(task) + "_" + datetime.now().strftime("%y-%m-%d") +'.json'

    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile)

    return train, valid, test, train_index, valid_index, test_index, preops_mask

def preprocess_inference(preops, metadata):
    """

    preops is the input data from EHR with some checks (?)
    metadata is the .json metadata file created during training available in B the box folder
    In the Box there is a file called fitter_feature_names_outcome.txt which has the column names after preprocessing from this code snippet.
    The last 101 elements in that list are the names of the procedure text embedded features. For now you can set them to 0 when feeding it into the prediction model.

    """
    preops.reset_index(drop=True, inplace=True)
    binary_variables = metadata["binary_var_name"]
    categorical_variables = metadata["categorical_name"]
    ordinal_variables = metadata["ordinal_variables"]
    continuous_variables = metadata["continuous_variables"]
    all_var = binary_variables + categorical_variables + ordinal_variables + continuous_variables

    # this is done because there are some variable which are absent in the wave2 data and hence setting them to nan here so that they can get imputed later
    if len(set(list(all_var)).difference(set(preops.columns))) != 0:  # orlogid_encoded will always be there in the otherway difference
        for i in list(set(list(all_var)).difference(set(preops.columns))):
            preops[i]=np.nan
            if i in categorical_variables: # this needs to be done otherwise the algo doesn't know any levels; ultimately all of them except the number of level examples will be in the nan categrory of the variable
                for j in range(len(metadata['levels'][i])):
                    preops.at[j, i] = metadata['levels'][i][j]


    # encoding the plannedDispo from text to number
    # {"OUTPATIENT": 0, '23 HOUR ADMIT': 1, "FLOOR": 1, "OBS. UNIT": 2, "ICU": 3}
    preops.loc[preops['plannedDispo'] == 'Outpatient', 'plannedDispo'] = 0
    preops.loc[preops['plannedDispo'] == 'Floor', 'plannedDispo'] = 1
    preops.loc[preops['plannedDispo'] == 'Obs. unit', 'plannedDispo'] = 2
    preops.loc[preops['plannedDispo'] == 'ICU', 'plannedDispo'] = 3
    if '' in list(preops['plannedDispo'].unique()):
        preops.loc[preops['plannedDispo'] == '', 'plannedDispo'] = np.nan
    preops['plannedDispo'] = preops['plannedDispo'].astype('float') # needed to convert this to float because the nans were not getting converted to int and this variable is object type



    preops_ohe = preops.copy()[set(binary_variables + categorical_variables + ordinal_variables + continuous_variables)]

    for i in binary_variables:
        preops_ohe[i].fillna(0, inplace=True)
        preops_ohe[i] = preops_ohe[i].astype('int32') # some object type are getting passed later on

    # this is kind of hardcoded; check your data beforehand for this; fixed this
    # this is done because there were two values for missing token (nan and -inf)
    # NOTE: try the isfinite function defined above
    # this section creates NaNs only to be filled in later. it harmonizes the different kinds of not-a-number representations
    temp_list = [i for i in preops_ohe['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in preops_ohe[
        'PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list != []:
        preops_ohe['PlannedAnesthesia'].replace(temp_list, np.NaN, inplace=True)

    if 'plannedDispo' in preops_ohe.columns:
        preops_ohe['plannedDispo'].replace('', np.NaN, inplace=True)

    for name in categorical_variables:
        preops_ohe[name] = preops_ohe[name].astype('category')
    for a in preops_ohe.columns:
        if preops_ohe[a].dtype == 'bool':
            preops_ohe[a] = preops_ohe[a].astype('int32')
        if preops_ohe[a].dtype == 'int32':
            if (a in categorical_variables) and (a not in ordinal_variables):
                preops_ohe[a] = pd.Series(
                    pd.Categorical(preops_ohe[a], categories=metadata['levels'][a], ordered=False))

    # one hot encoding
    # this is reverse from how I would have thought about it. It starts with the list of target columns, gets the value associated with that name, then scans for values matching the target
    # i probably would have used pd.get_dummies, concat, drop cols not present in the original, add constant 0 cols that are missing. I think this works as-is
    encoded_var = metadata['encoded_var']
    for ev in encoded_var:
        preops_ohe[ev] = 0
        ev_name = ev.rsplit("_", 1)[0]
        ev_value = ev.rsplit("_", 1)[1]
        if ev_value != 'nan':
            if len(preops[ev_name].unique()) < 2:
                dtype_check = preops[ev_name].unique()[0]
            else:
                dtype_check = preops[ev_name].unique()[1]
            if type(dtype_check) == np.float64 or type(dtype_check) == np.int64:
                preops_ohe[ev] = np.where(preops_ohe[ev_name].astype('float') == float(ev_value), 1, 0)
            elif type(dtype_check) == bool:
                preops_ohe[ev] = np.where(preops[ev_name].astype('str') == ev_value, 1, 0)
            else:
                preops_ohe[ev] = np.where(preops_ohe[ev_name] == ev_value, 1, 0)
    # this for loop checks if the categorical variable doesn't have 1 in any non-NAN value column and then assigns 1 in the nan value column
    # this is done because the type of nans in python are complicated and different columns had different type of nans
    for i in categorical_variables:
        name = str(i) + "_nan"
        lst = [col for col in encoded_var if (i == col.rsplit("_", 1)[0]) and (col != name)]
        preops_ohe['temp_col'] = preops_ohe[lst].sum(axis=1)
        preops_ohe[name] = np.where(preops_ohe['temp_col'] == 1, 0, 1)
        preops_ohe.drop(columns=['temp_col'], inplace=True)
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    # mean imputing and scaling the continuous variables
    preops_ohe[continuous_variables].fillna(
        dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"])),
        inplace=True)  ## warning about copy
    # this is done because nan that are of float type is not recognised as missing by above commands
    for i in continuous_variables:
        if preops_ohe[i].isna().any() == True:
            preops_ohe[i].replace(preops_ohe[i].unique().min(),
                                  dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"]))[i],
                                  inplace=True)
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_cont'], continuous_variables)
    # median Imputing_ordinal variables
    # imputing
    for i in ordinal_variables:
        preops_ohe.loc[:, i] = pd.to_numeric(preops_ohe[i], errors='coerce').fillna(
            dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i])
        # replace(preops_ohe[i].unique().min(), dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i], inplace=True) # min because nan is treated as min
    # normalization
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_ord'], ordinal_variables)
    preops_ohe = preops_ohe.reindex(metadata["column_all_names"], axis=1)

    if "person_integer" in preops_ohe.columns:
        preops_ohe.rename({"person_integer":"orlogid_encoded"}, axis=1, inplace=True)

    preops_ohe['orlogid_encoded'] = preops['orlogid_encoded']

    return preops_ohe


if False:
    preops = feather.read_feather('/mnt/ris/ActFastExports/v1.1/preops_reduced_for_training.feather')
    # outcomes = pd.read_csv('/mnt/ris/ActFastExports/v1.1/epic_outcomes.csv')
    task = 'endofcase'

    # currently not restratining the preops to any task; endofcase is just a placeholder
    # if task == 'endofcase':
    #     end_of_case_times = feather.read_feather('/mnt/ris/ActFastData/Epic_TS_Prototyping/end_of_case_times.feather')
    #     # updating the end_of_case_times targets for bigger distribution;
    #     """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
    #     end_of_case_times['true_test'] = end_of_case_times['endtime'] - 25
    #     end_of_case_times['t1'] = end_of_case_times['true_test'] -40
    #     end_of_case_times['t2'] = end_of_case_times['true_test'] -50
    #     end_of_case_times['t3'] = end_of_case_times['true_test'] -60
    #
    #
    # end_of_case_times = end_of_case_times[end_of_case_times['t3'] > 10]
    # end_of_case_times = end_of_case_times[end_of_case_times['true_test'] > 10]
    # end_of_case_times = end_of_case_times[end_of_case_times['endtime'] < 512]
    #
    # with_val_Cases = list(end_of_case_times['person_integer'])
    # preops = preops[preops['person_integer'].isin(with_val_Cases)]


    preops_tr, preops_val, preops_te, tr_idx, val_idx, te_idx = preprocess_train(preops, False)

    md_f = open('./preops_metadata.json')
    md = json.load(md_f)

    preops_proc = preprocess_inference(preops,False, md)