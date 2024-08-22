"""
Code to perform one time imputation on two types of flowsheets data (very dense and other)
1) For the very dense, the imputation is LOCF. The technique involves taking the diff so that ater the to_dense and cumsum we get the original tensor back
2) for the other sparse, the method would involve running a regression of preops on the first observed value of each time series

"""

import json
import os
import sys, argparse
import glob
import pickle
import numpy as np
import pandas as pd
import math
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
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime

# starting time of the script
start_time = datetime.now()

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

def preprocess_train(preops, data_dir, test_size=0.2):
    preops.reset_index(drop=True, inplace=True)

    preops_mask = preops.copy()

    lab_cats = pd.read_csv(data_dir + 'mapping_info/categories_labs.csv')

    ordinal_variables = list(pd.read_csv(data_dir + 'mapping_info/ordinal_vars.txt', delimiter= "\t", header=None)[0])

    preop_labs_categorical = lab_cats[lab_cats['all_numeric'] == 0.0]['LAB_TEST'].unique()
    num_lab_cats = [i for i in lab_cats['LAB_TEST'].unique() if
                    (i in preops.columns) and (i not in preop_labs_categorical) and (i not in ordinal_variables)]

    ordinal_variables = [i for i in ordinal_variables if (i in preops.columns)]

    # making sure that Sex variable has 0 and 1 values instead of 1 and 2
    preops.loc[preops['Sex'] == 1, 'Sex'] = 0
    preops.loc[preops['Sex'] == 2, 'Sex'] = 1

    # breakpoint()
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
    # breakpoint()
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
        #if a in ordinal_variables:
        #  preops[a] = preops[a].astype('int32')
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
        'orlogid_encoded')  # this var went into categorical due to its initial datatype of being an object

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
    preops_mask.drop(columns='orlogid_encoded', inplace=True)
    for name in preops_mask.columns:
        preops_mask[name] = preops_mask[name].astype('int')

    # partitioning the data into train and a holdout set so that there is no leakage when training the final model
    if True:
        upto_test_idx = int(test_size * len(preops_ohe))
        test = preops_ohe.iloc[:upto_test_idx]
        train = preops_ohe.iloc[upto_test_idx:]

    train_index = train.index
    test_index = test.index

    train_orlogids = train['orlogid_encoded']
    test_orlogids = test['orlogid_encoded']

    # print( list(train.columns) )
    train.drop(columns="orlogid_encoded", inplace=True)
    test.drop(columns="orlogid_encoded", inplace=True)

    # mean imputing and scaling the continuous variables

    train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont

    # median Imputing_ordinal variables

    # imputing
    for i in ordinal_variables:
        if np.isnan(preops[i].unique().min()) == True:
            train[i].replace(train[i].unique().min(), train[i].median(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].median(), inplace=True)
        if (train[i].dtype == 'O') or (test[i].dtype=='O'):
            train[i] = train[i].astype('int')
            test[i] = test[i].astype('int')

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = [int(i) for i in train[ordinal_variables].min(axis=0).values]
    normalizing_values_ord['max'] = [int(i) for i in train[ordinal_variables].max(axis=0).values]


    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord

    if  (sum(test.isna().any()) > 0) or (sum(train.isna().any()) > 0):
        raise AssertionError("Processed data has nans")

    meta_Data["encoded_var"] = encoded_variables

    meta_Data["binary_var_name"] = binary_variables

    meta_Data["categorical_name"] = categorical_variables
    meta_Data["ordinal_variables"] = ordinal_variables
    meta_Data["continuous_variables"] = continuous_variables
    meta_Data["column_all_names"] = list(preops_ohe.columns)

    # output_file_name = '/home/trips/PeriOperative_RiskPrediction/Xgboost_model_Other_flow/preops_metadata_fromflowsheet_imp.json'
    output_file_name = data_dir+ 'flow_ts/Xgboost_model_Other_flow_wave2/preops_metadata_fromflowsheet_imp.json'

    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile)

    train['orlogid_encoded'] = train_orlogids
    test['orlogid_encoded'] = test_orlogids

    return train, test

def flowsheet_imputer_estimate_generator_training(first_flow, preops, inp_data_dir):
    """
    trains xgboost models that predicts the 0-time estimator for flowsheet timeseries using the preops as predictors and the first non observed value in the time series as y
    saves those models and saves the generated predictions also at the time of training
    """
    preops_train, preops_test = preprocess_train(preops, inp_data_dir)

    preops_ohe= pd.concat([preops_test, preops_train], axis=0)  # this is being done because we want the imputation for the whole dataset but the imputer will be trained only the train dataset

    # regression bit here; currently running each regression seperately
    """ OTHER FLOW"""
    other_flow_est = pd.DataFrame(index=preops_ohe.index, columns=['orlogid_encoded'])
    other_flow_est['orlogid_encoded'] = preops_ohe['orlogid_encoded']
    preops_other_flow_X = preops_train[preops_train['orlogid_encoded'].isin(first_flow['orlogid_encoded'])]
    preops_other_flow_X.set_index('orlogid_encoded', inplace=True)
    first_flow.set_index('orlogid_encoded', inplace=True)
    train_idx_otherflow = list(set(first_flow.index).intersection(preops_other_flow_X.index))
    for i in first_flow.columns:
        temp_tr_y = first_flow.loc[train_idx_otherflow][i].dropna()
        temp_tr_x = preops_other_flow_X.loc[temp_tr_y.index]
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror').fit(temp_tr_x,temp_tr_y)
        other_flow_est[i] = xgb_reg.predict(preops_ohe.drop(columns=['orlogid_encoded']))
        # xgb_reg.save_model("/home/trips/PeriOperative_RiskPrediction/Xgboost_model_Other_flow/Measure_"+str(i)+".json")
        xgb_reg.save_model(inp_data_dir + "flow_ts/Xgboost_model_Other_flow_wave2/Measure_"+str(i)+".json")

    # other_flow_est.to_csv("/home/trips/PeriOperative_RiskPrediction/Xgboost_model_Other_flow/Other_flow_0time_imputedvalues.csv")
    other_flow_est.to_csv(inp_data_dir + "flow_ts/Xgboost_model_Other_flow_wave2/Other_flow_0time_imputedvalues.csv")
    return

# this has a lot of issues so wrote a new function which is cleaner
def flowsheet_imputation_training_old(very_dense_flow, other_intra_flow_wlabs, inp_data_dir, imputer_other_flow =None):

    """ INTERPOLATION for GAPS -> RECODING -> SPARSIFY -> TO DENSE -> CUMSUM (took care of imputing the initial nans)  """

    """ OTHER FLOW DATA """

    # # reading the imputers
    if (not isinstance(imputer_other_flow, pd.DataFrame)):
        imputer_other_flow = pd.read_csv(inp_data_dir+ "flow_ts/Xgboost_model_Other_flow/Other_flow_0time_imputedvalues.csv")
    # breakpoint()

    # converting the imputers in coordinate format
    imputer_other_flow_coord = pd.melt(imputer_other_flow, id_vars=['orlogid_encoded'],
                                       value_vars=[i for i in imputer_other_flow.columns if
                                                   i not in ['Unnamed: 0', 'orlogid_encoded']], var_name='measure_index',
                                       value_name='VALUE')

    imputer_other_flow_coord['measure_index'] = imputer_other_flow_coord['measure_index'].astype('int32')

    imputer_other_flow_coord['timepoint'] = 0


    other_intra_flow_wlabs['timepoint'] = other_intra_flow_wlabs['timepoint'] + 1
    # fill in the nans (gaps in between) with linear interpolation for a combination of person and measure
    other_intra_flow_wlabs.groupby(by=['orlogid_encoded', 'measure_index'], group_keys=True).apply( lambda group: group.interpolate(method='linear', limit_direction='forward'))

    # appending the 0-time estimate
    other_intra_flow_wlabs = pd.concat([other_intra_flow_wlabs, imputer_other_flow_coord[imputer_other_flow_coord['orlogid_encoded'].isin(other_intra_flow_wlabs['orlogid_encoded'])]], ignore_index=True)
    other_intra_flow_wlabs.sort_values(by=['orlogid_encoded', 'timepoint'], inplace=True)

    other_intra_flow_wlabs = other_intra_flow_wlabs.set_index(['orlogid_encoded', 'measure_index', 'timepoint']).groupby(by=['orlogid_encoded','measure_index'], group_keys=True).diff().dropna().reset_index()  # by this point there shouldn't be any nans except for the starting ones which will be carried forward anyway

    # this is repeated here because the 0 time has become nan now and removed but we still want its value
    other_intra_flow_wlabs_imputed = pd.concat([other_intra_flow_wlabs, imputer_other_flow_coord[
        imputer_other_flow_coord['orlogid_encoded'].isin(other_intra_flow_wlabs['orlogid_encoded'])]], ignore_index=True)

    if True:
        """  Saving the recoded (coordinate format) and imputed data to feather files so that it is easily available """
        other_intra_flow_wlabs_imputed.to_feather('/home/trips/PeriOperative_RiskPrediction/Imputed_other_flow.feather')
    # #
    # breakpoint()

    """ VERY DENSE DATA """

    if False:
        # old method for imputation
        mask = very_dense_flow.copy()  # for masking later

        # reading the imputers
        if (not isinstance(imputer_very_dense_flow, pd.DataFrame)):
            imputer_very_dense_flow = pd.read_csv(
                "/home/trips/Epic-Time-SeriesModels/Xgboost_model_dense_flow/dense_flow_0time_imputedvalues.csv")
            imputer_very_dense_flow.drop(columns=['Unnamed: 0'], inplace=True)
        imputer_very_dense_flow['timepoint'] = 0

        # temporary
        # subset_df = very_dense_flow[very_dense_flow['person_integer'].isin(np.arange(500))]

        # filling in the gaps
        very_dense_flow = very_dense_flow.groupby(by=['person_integer']).apply(
            lambda group: group.interpolate(method='linear', limit_direction='forward'))

        # this is temporary; to avoid the overwriting in the next step; will be reversed after the initial estimate is dropped form the df.
        very_dense_flow['timepoint'] = very_dense_flow['timepoint'] + 1

        # appending the 0-time estimate
        very_dense_flow = pd.concat([very_dense_flow, imputer_very_dense_flow[
            imputer_very_dense_flow['person_integer'].isin(very_dense_flow['person_integer'])]], ignore_index=True)

        # doing an ffill now here because the data is already
        very_dense_flow.sort_values(by=['person_integer', 'timepoint'], inplace=True)
        very_dense_flow = very_dense_flow.groupby(by=['person_integer']).apply(
            lambda group: group.interpolate(method='ffill', limit_direction='forward'))

        # after this only nans that should be left are the ones where a particular measure is not recorded for the patient throughout the case
        very_dense_flowsheet_measures = list(very_dense_flow.columns)
        very_dense_flowsheet_measures.remove('timepoint')
        very_dense_flowsheet_measures.remove('person_integer')
        mapping_flowsheet_measurename = dict(
            zip(very_dense_flowsheet_measures, range(len(very_dense_flowsheet_measures))))
        very_dense_flowsheet_measures_mean = very_dense_flow[very_dense_flowsheet_measures].mean()
        very_dense_flow = very_dense_flow.fillna(value=very_dense_flowsheet_measures_mean)

        # taking the diff
        very_dense_flow = very_dense_flow.set_index(['person_integer', 'timepoint']).groupby(
            by=['person_integer']).diff().dropna().reset_index()

        # this is repeated here because the 0 time has become nan now and removed but we still want its value
        very_dense_flow = pd.concat([very_dense_flow, imputer_very_dense_flow[
            imputer_very_dense_flow['person_integer'].isin(very_dense_flow['person_integer'])]], ignore_index=True)

        # getting the coordinate format so that converting to sparse and then dense is easy
        very_dense_flowsheet_coord_imputed = pd.melt(very_dense_flow, id_vars=['person_integer', 'timepoint'],
                                                     value_vars=very_dense_flowsheet_measures,
                                                     var_name='measure_index', value_name='VALUE')
        very_dense_flowsheet_coord_imputed.replace(mapping_flowsheet_measurename, inplace=True)

        ## mask for the very dense data
        mask = pd.concat([mask, imputer_very_dense_flow[
            imputer_very_dense_flow['person_integer'].isin(mask['person_integer'])]], ignore_index=True)
        mask[very_dense_flowsheet_measures] = (mask[very_dense_flowsheet_measures].isnull()).astype('int')
        mask_coord = pd.melt(mask, id_vars=['person_integer', 'timepoint'], value_vars=very_dense_flowsheet_measures,
                             var_name='measure_index', value_name='VALUE')
        mask_coord.replace(mapping_flowsheet_measurename, inplace=True)

    # breakpoint()

    very_dense_flowsheet_measures = list(very_dense_flow.columns)
    very_dense_flowsheet_measures.remove('timepoint')
    very_dense_flowsheet_measures.remove('orlogid_encoded')
    mapping_flowsheet_measurename = dict(
        zip(very_dense_flowsheet_measures, range(len(very_dense_flowsheet_measures))))

    very_dense_flow_coord = pd.melt(very_dense_flow, id_vars=['orlogid_encoded', 'timepoint'],
                                                 value_vars=very_dense_flowsheet_measures,
                                                 var_name='measure_index', value_name='VALUE')
    very_dense_flow_coord.replace(mapping_flowsheet_measurename, inplace=True)
    very_dense_flow_coord.dropna(inplace=True)

    print("cross1")

    first_rec_index = very_dense_flow_coord.groupby(by=['orlogid_encoded', 'measure_index'], group_keys=True)['timepoint'].min().reset_index().set_index(['orlogid_encoded', 'measure_index', 'timepoint'])

    print("cross2")

    a0 = very_dense_flow_coord.set_index(['orlogid_encoded', 'measure_index', 'timepoint'])
    print("cross3")

    first_rec_values = a0.loc[first_rec_index.index]
    first_rec_values.drop(0, level=2, axis=0, inplace=True) # dropping the observations for which there already exists the 0 time values; level =2 is the time level in multiindex
    first_rec_values.reset_index(inplace=True)

    print("cross4")

    # this takes a lot of time
    very_dense_flowsheet_coord_imputed = very_dense_flow_coord.set_index(['orlogid_encoded', 'measure_index', 'timepoint']).groupby(
        by=['orlogid_encoded',
            'measure_index'], group_keys=True).diff().dropna().reset_index()

    very_dense_flow_coord_with0timeforall = pd.concat([first_rec_values, very_dense_flowsheet_coord_imputed])

    # breakpoint()
    print("cross6")

    if True:
        """  Saving the recoded (coordinate format) and imputed data to feather files so that it is easily available """
        very_dense_flow_coord_with0timeforall.to_feather('/home/trips/PeriOperative_RiskPrediction/Imputed_very_dense_flow.feather')

        end_time = datetime.now()

        timetaken = end_time - start_time
        print("time taken to run the imputation script", timetaken)

        exit()

    """  tensorize and sparsify the other flow data """
    index_med_other_flow = torch.tensor(other_intra_flow_wlabs_imputed[['person_integer', 'timepoint', 'measure_index']].values,
                                        dtype=int)
    value_med_other_flow = torch.tensor(other_intra_flow_wlabs_imputed['VALUE'].values)
    dense_other_flow_proc = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                               value_med_other_flow).to_dense()
    dense_other_flow_proc = torch.cumsum(dense_other_flow_proc, dim=1)

    """  tensorize and sparsify the very dense data """
    index_med_very_dense = torch.tensor(very_dense_flowsheet_coord_imputed[['person_integer', 'timepoint', 'measure_index']].values,
                                        dtype=int)
    value_med_very_dense = torch.tensor(very_dense_flowsheet_coord_imputed['VALUE'].values)
    very_dense_flowsheet_proc = torch.sparse_coo_tensor(torch.transpose(index_med_very_dense, 0, 1),
                                              value_med_very_dense).to_dense()
    very_dense_flowsheet_proc = torch.cumsum(very_dense_flowsheet_proc, dim=1)

    flowsheet_dense_comb = torch.cat((very_dense_flowsheet_proc, dense_other_flow_proc), dim=2)

    return other_intra_flow_wlabs_imputed, very_dense_flowsheet_coord_imputed, flowsheet_dense_comb


def flowsheet_imputation_training(very_dense_flow, other_intra_flow_wlabs, inp_data_dir, imputer_other_flow =None):
    # updated on Aug 5 2024 after discussion with Ryan (about the dense ones) and looking into the isolate inference branch of Epic codes (to confirm that linear innterpolation is not needed in between for other flow)
    # for dense: the output from this file will be coordinate format which will be converted to coo and then to dense and cumsum to ultimately obtain the LOCF verison with the initial values backfill imputed
    # for other flow: the output of this function will be coordinate format which will be converted to sparse tensors. The first value here is either preop predicted or actually recorded. Even though this could be used as it is in the sparse format, we perform the (cumsum(to_dense)) operation at the batch level in the collate function.

    # # reading the imputers
    if (not isinstance(imputer_other_flow, pd.DataFrame)):
        # imputer_other_flow = pd.read_csv(inp_data_dir+ "flow_ts/Xgboost_model_Other_flow/Other_flow_0time_imputedvalues.csv")
        imputer_other_flow = pd.read_csv(inp_data_dir+ "flow_ts/Xgboost_model_Other_flow_wave2/Other_flow_0time_imputedvalues.csv")

    """ OTHER FLOW DATA """
    imputer_other_flow_coord = pd.melt(imputer_other_flow, id_vars=['orlogid_encoded'],
                                       value_vars=[i for i in imputer_other_flow.columns if
                                                   i not in ['Unnamed: 0', 'orlogid_encoded']], var_name='measure_index',
                                       value_name='VALUE')

    imputer_other_flow_coord['measure_index'] = imputer_other_flow_coord['measure_index'].astype('int32')

    imputer_other_flow_coord['timepoint'] = 0
    other_intra_flow_wlabs.dropna(subset=["VALUE"])

    other_intra_flow_wlabs = pd.concat([other_intra_flow_wlabs, imputer_other_flow_coord], ignore_index=True).drop_duplicates(subset=["orlogid_encoded", "measure_index", "timepoint"] , keep="first")

    other_intra_flow_wlabs.sort_values(by=['orlogid_encoded', 'timepoint'], inplace=True)
    other_intra_flow_wlabs_imputed = pd.concat([
        other_intra_flow_wlabs[other_intra_flow_wlabs.timepoint == 0]
        , other_intra_flow_wlabs.set_index(['orlogid_encoded', 'measure_index', 'timepoint']).groupby(by=['orlogid_encoded','measure_index']).diff().dropna().reset_index()
        ], ignore_index=True)

    """  Saving the recoded (coordinate format) and imputed data to feather files so that it is easily available """
    # other_intra_flow_wlabs_imputed.to_feather('/home/trips/PeriOperative_RiskPrediction/Imputed_other_flow.feather')
    other_intra_flow_wlabs_imputed.to_feather(inp_data_dir + 'flow_ts/Imputed_other_flow_wave2.feather')

    """ VERY DENSE DATA """
    very_dense_flowsheet_measures = list(very_dense_flow.columns)
    very_dense_flowsheet_measures.remove('timepoint')
    very_dense_flowsheet_measures.remove('orlogid_encoded')
    mapping_flowsheet_measurename = dict(
        zip(very_dense_flowsheet_measures, range(len(very_dense_flowsheet_measures))))

    very_dense_flow_coord = pd.melt(very_dense_flow, id_vars=['orlogid_encoded', 'timepoint'],
                                                 value_vars=very_dense_flowsheet_measures,
                                                 var_name='measure_index', value_name='VALUE')
    very_dense_flow_coord.replace(mapping_flowsheet_measurename, inplace=True)
    very_dense_flow_coord.dropna(inplace=True)

    first_rec_index = very_dense_flow_coord.loc[very_dense_flow_coord.groupby(['measure_index', 'orlogid_encoded'])['timepoint'].idxmin()]
    first_rec_index = first_rec_index[first_rec_index.timepoint > 0 ]
    first_rec_index['timepoint'] = 0

    # this takes a lot of time
    very_dense_flowsheet_coord_imputed = very_dense_flow_coord.set_index(['orlogid_encoded', 'measure_index', 'timepoint']).groupby(by=['orlogid_encoded','measure_index'], group_keys=True).diff().dropna().reset_index()
    very_dense_flow_coord_with0timeforall = pd.concat([first_rec_index, very_dense_flowsheet_coord_imputed], ignore_index=True)

    """  Saving the recoded (coordinate format) and imputed data to feather files so that it is easily available """
    # very_dense_flow_coord_with0timeforall.to_feather('/home/trips/PeriOperative_RiskPrediction/Imputed_very_dense_flow.feather')
    very_dense_flow_coord_with0timeforall.to_feather(inp_data_dir + 'flow_ts/Imputed_very_dense_flow_wave2.feather')

    end_time = datetime.now()

    timetaken = end_time - start_time
    print("time taken to run the imputation script", timetaken)

if False:
    # reading files
    data_dir = '/mnt/ris/ActFastExports/v1.3.2/'
    # data_dir = '/input/'
    # breakpoint()

    # first_imputer_path = data_dir+ "flow_ts/Xgboost_model_Other_flow/Other_flow_0time_imputedvalues.csv"
    first_imputer_path = data_dir+ "flow_ts/Xgboost_model_Other_flow_wave2/Other_flow_0time_imputedvalues.csv"

    if(not os.path.exists(first_imputer_path)):
        # preops = pd.read_csv(data_dir + 'epic_preop.csv')
        preops = pd.read_csv(data_dir + 'epic_preop_wave2.csv')
        # to drop the old pmh and problem list
        to_drop_old_pmh_problist_with_others = ["MentalHistory_anxiety", "MentalHistory_bipolar",
                                                "MentalHistory_depression",
                                                "MentalHistory_schizophrenia", "PNA", "delirium_history",
                                                "MentalHistory_adhd",
                                                "MentalHistory_other", "opioids_count",
                                                "total_morphine_equivalent_dose",
                                                'pre_aki_status', 'preop_ICU', 'preop_los', 'URINE UROBILINOGEN',
                                                'MRN_encoded', 'time_of_day',
                                                'BACTERIA, URINE', 'CLARITY, URINE', 'COLOR, URINE',
                                                'EPITHELIAL CELLS, SQUAMOUS, URINE',
                                                'GLUCOSE, URINE, QUALITATIVE', 'HYALINE CAST',
                                                'LEUKOCYTE ESTERASE, URINE',
                                                'PROTEIN, URINE QUALITATIVE',
                                                'RED BLOOD CELLS, URINE', 'URINE BLOOD', 'URINE KETONES',
                                                'URINE NITRITE',
                                                'URINE UROBILINOGEN', 'WHITE BLOOD CELLS, URINE']
        to_drop_old_pmh_problist = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                    "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                    "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose",
                                    'pre_aki_status', 'preop_ICU', 'preop_los',
                                    'URINE UROBILINOGEN', 'time_of_day',
                                    'CLARITY, URINE', 'COLOR, URINE',
                                    'GLUCOSE, URINE, QUALITATIVE', 'URINE BLOOD', 'URINE KETONES', 'AnestStop']

        # preops = preops.drop(columns=to_drop_old_pmh_problist_with_others)
        preops = preops.drop(columns=to_drop_old_pmh_problist)

        # first values
        # first_flow = feather.read_feather(data_dir + 'flow_ts/first_flow.feather')
        first_flow = feather.read_feather(data_dir + 'flow_ts/first_flow_wave2.feather')
        flowsheet_imputer_estimate_generator_training(first_flow, preops, data_dir)

    # very_dense_flow = feather.read_feather(data_dir + 'flow_ts/very_dense_flow.feather')
    # other_intra_flow_wlabs = feather.read_feather(data_dir + 'flow_ts/other_intra_flow_wlabs.feather')
    # other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] < 0].index, inplace=True)

    very_dense_flow = feather.read_feather(data_dir + 'flow_ts/very_dense_flow_wave2.feather')
    other_intra_flow_wlabs = feather.read_feather(data_dir + 'flow_ts/other_intra_flow_wlabs_wave2.feather')
    other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] < 0].index, inplace=True)

    flowsheet_imputation_training(very_dense_flow, other_intra_flow_wlabs, data_dir)
