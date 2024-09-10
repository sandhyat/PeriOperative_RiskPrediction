"""
This file reads an external dataset and validates the existing tabular models on the external dataset.
Currently, the external dataset is one from a different time period which may not have a complete overlap with the training data in terms of the features.
The tabular models here are: XGBT, TabNet, Scarf

"""

# importing packages
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.stats import pearsonr
from xgboost import XGBClassifier, XGBRegressor
import sys, argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from datetime import datetime

from End_to_end_supervised import preprocess_inference
# import ./Preops_processing as pps
from Two_stage_selfsupervised.tasks.scarf_model_updated import *
import scipy
from datetime import datetime
import json
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import pickle

parser = argparse.ArgumentParser(description='Tabular modular model validation in wave2')

## modalities to select
parser.add_argument('--preops', default=True, action='store_true',
                    help='Whether to add preops and bow to ts representation')
parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
parser.add_argument('--homemeds', action="store_true",
                    help='Whether to add homemeds to ts representation')

## for the homemeds
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum'}


## task and setup parameters
parser.add_argument("--task",  default="icu") #
parser.add_argument("--randomSeed", default=100, type=int )


args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__)

modality_to_use = []
if eval('args.preops') == True:
    modality_to_use.append('preops')
    modality_to_use.append('cbow')

if eval('args.pmhProblist') == True:
    modality_to_use.append('pmh')
    modality_to_use.append('problist')

if eval('args.homemeds') == True:
    modality_to_use.append('homemeds')


# data_dir = '/mnt/ris/ActFastExports/v1.3.2/'
data_dir = '/input/'

# out_dir = './'
out_dir = '/output/'


to_drop_old_pmh_problist  = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                             "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                             "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose", 'pre_aki_status', 'preop_ICU', 'preop_los',
                             'URINE UROBILINOGEN', 'time_of_day',
                                 'CLARITY, URINE','COLOR, URINE',
                                'GLUCOSE, URINE, QUALITATIVE','URINE BLOOD', 'URINE KETONES', 'AnestStop']
preops_wave2 = pd.read_csv(data_dir + 'epic_preop_wave2.csv')

outcomes_wave2 = pd.read_csv(data_dir + 'epic_outcomes_wave2.csv')
outcomes_wave2 = outcomes_wave2.drop(index = preops_wave2[preops_wave2['age']<18].index)
outcomes_wave2 = outcomes_wave2.dropna(subset=['orlogid_encoded'])

outcomes_wave2 = outcomes_wave2.dropna(subset=['ICU'])


regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration', 'n_glu_high',
                           'low_sbp_time','aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                           'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                           'opioids_count_day0', 'opioids_count_day1']
binary_outcome = args.task not in regression_outcome_list
binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac', 'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']


if args.task in ['postop_del', 'severe_present_1', 'worst_pain_1', 'worst_pain_0']:
    # dropping the nans for postop_del and severe_present_1. Can't wait until later as these need to be converted into integer
    outcomes_wave2 = outcomes_wave2.dropna(subset=[args.task])

# outcome
icu_outcome = outcomes_wave2[['orlogid_encoded', 'ICU']]
icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)

mortality_outcome = outcomes_wave2[['orlogid_encoded', 'death_in_30']]
mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)


if args.task in ['aki1', 'aki2', 'aki3']:
    outcomes_wave2 = outcomes_wave2.dropna(subset=['post_aki_status'])
    aki_outcome = outcomes_wave2[['orlogid_encoded', 'post_aki_status']]
    if args.task == 'aki1':
        aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
        aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
    if args.task == 'aki2':
        aki_outcome.loc[aki_outcome[
                            'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
    if args.task == 'aki3':
        aki_outcome.loc[aki_outcome[
                            'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
    aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)

dvt_pe_outcome = outcomes_wave2[['orlogid_encoded', 'DVT_PE']]

if args.task in regression_outcome_list:
    outcomes_wave2['survival_time'] = np.minimum(outcomes_wave2['survival_time'], 90)
    # outcomes['readmission_survival'] = np.minimum(outcomes['readmission_survival'], 30) # this is being commented because we are going to predict everything as regression and then evaluate as classification by thresholding at 30
    # outcomes['n_glucose_high'] = outcomes['n_glucose_high'].fillna(0)  # this might not be needed as already taken of by the where statement
    outcomes_wave2['n_glu_high'] = np.where(outcomes_wave2['N_glu_measured'] > 0, outcomes_wave2['n_glu_high']/outcomes_wave2['N_glu_measured'], 0)
    # outcomes['total_blood'] = outcomes['total_blood'].fillna(0) # no null observations
    # outcomes['low_sbp_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_sbp_time']/outcomes['total_t'], 0)
    outcomes_wave2['low_relmap_time'] = np.where(outcomes_wave2['total_t'] > 0, outcomes_wave2['low_relmap_time']/outcomes_wave2['total_t'], 0)
    outcomes_wave2['low_map_time'] = np.where(outcomes_wave2['total_t'] > 0, outcomes_wave2['low_map_time']/outcomes_wave2['total_t'], 0)
    # to check the following
    # outcomes['aoc_low_sbp'] = np.where(outcomes['total_t'] > 0, outcomes['aoc_low_sbp']/outcomes['total_t'], 0)
    # outcomes['low_relmap_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_aoc']/outcomes['total_t'], 0)
    # outcomes['low_map_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_aoc']/outcomes['total_t'], 0)
    # outcomes['postop_vent_duration'] = outcomes['postop_vent_duration'].fillna(0)

    outcome_df_wave2 = outcomes_wave2[['orlogid_encoded', args.task]]
elif args.task in binary_outcome_list:
    if args.task == 'VTE':
        temp_outcome = outcomes_wave2[['orlogid_encoded']]
        temp_outcome[args.task] = np.where(outcomes_wave2['DVT'] == True, 1, 0) + np.where(outcomes_wave2['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[args.task] == 2, args.task] = 1    # not the most efficient but spent more time than it needed
    elif args.task in  ['n_glucose_low', 'n_glu_high', 'low_sbp_time']:  # the threshold of 0 for low_sbp_time was decided as the 75th percentile (outcomes_wave2[args.task].describe()['75%'])
        temp_outcome = outcomes_wave2[['orlogid_encoded', args.task]]
        temp_outcome[args.task] = temp_outcome[args.task].fillna(0)
        temp_outcome[args.task] = np.where(temp_outcome[args.task]>0, 1, 0)
    elif args.task in ['worst_pain_1', 'worst_pain_0']:
        temp_outcome = outcomes_wave2[['orlogid_encoded', args.task]]
        temp_outcome[args.task] = np.where(temp_outcome[args.task]>=7, 1, 0)
    else:
        temp_outcome = outcomes_wave2[['orlogid_encoded', args.task]]
        temp_outcome.loc[temp_outcome[args.task] == True, args.task] = 1
        temp_outcome.loc[temp_outcome[args.task] == False, args.task] = 0
    temp_outcome[args.task] = temp_outcome[args.task].astype(int)
    outcome_df_wave2 = temp_outcome
elif (args.task == 'dvt_pe'):
    outcome_df_wave2 = dvt_pe_outcome
elif (args.task == 'icu'):
    outcome_df_wave2 = icu_outcome
elif (args.task == 'mortality'):
    outcome_df_wave2 = mortality_outcome
elif (args.task == 'aki1' or args.task == 'aki2' or args.task == 'aki3'):
    outcome_df_wave2 = aki_outcome
else:
    raise Exception("outcome not handled")

cbow_wave2 = pd.read_csv(data_dir+'cbow_proc_text_wave2.csv')
cbow_wave2 = cbow_wave2.drop(index = preops_wave2[preops_wave2['age']<18].index)

preops_wave2 = preops_wave2.drop(columns=to_drop_old_pmh_problist)
preops_wave2 = preops_wave2.drop(index = preops_wave2[preops_wave2['age']<18].index)


if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
    sub_id = preops_wave2[preops_wave2['plannedDispo']!='ICU'].index
    preops_wave2 = preops_wave2.loc[sub_id]

if args.task == 'post_dialysis':
    sub_id = preops_wave2[(preops_wave2['ESRD'] != 1) & (preops_wave2['Dialysis'] != 1)].index   # dropping every row in which at least one value is 1
    preops_wave2 = preops_wave2[(preops_wave2['ESRD'] != 1) & (preops_wave2['Dialysis'] != 1)]

## intersect 3 mandatory data sources: preop, outcome
combined_case_set = list(set(outcome_df_wave2["orlogid_encoded"].values).intersection(set(preops_wave2['orlogid_encoded'].values)))

outcome_df_wave2 = outcome_df_wave2.loc[outcome_df_wave2['orlogid_encoded'].isin(combined_case_set)]
outcome_df_wave2.set_axis(["orlogid_encoded", "outcome"], axis=1, inplace=True)

new_index = outcome_df_wave2["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)  # this will serve as a good baseline to select cases with outcomes

preops_wave2 = preops_wave2.loc[preops_wave2['orlogid_encoded'].isin(combined_case_set)]

cbow_wave2 = cbow_wave2.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

cbow_cols = [col for col in cbow_wave2.columns if 'BOW' in col]
cbow_wave2['BOW_NA'] = np.where(np.isnan(cbow_wave2[cbow_cols[0]]), 1, 0)
cbow_wave2.fillna(0, inplace=True)

if 'homemeds' in modality_to_use:
    home_meds = pd.read_csv(data_dir + 'home_med_cui_wave2.csv', low_memory=False)
    Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

    home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded', 'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now

    home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left', on='rxcui')
    home_meds_embedded.drop(columns=['code', 'description', 'source'], inplace=True)

    # home meds basic processing
    home_meds_freq = home_meds[['orlogid_encoded', 'rxcui', 'Frequency']].pivot_table(index='orlogid_encoded',
                                                                                      columns='rxcui',
                                                                                      values='Frequency')
    rxcui_freq = home_meds["rxcui"].value_counts().reset_index()
    # rxcui_freq = rxcui_freq.rename({'count':'rxcui_freq', 'rxcui':'rxcui'}, axis =1)
    rxcui_freq = rxcui_freq.rename({'rxcui': 'rxcui_freq', 'index': 'rxcui'}, axis=1)
    home_meds_small = home_meds[home_meds['rxcui'].isin(list(rxcui_freq[rxcui_freq['rxcui_freq'] > 100]['rxcui']))]
    home_meds_small['temp_const'] = 1
    home_meds_ohe = home_meds_small[['orlogid_encoded', 'rxcui', 'temp_const']].pivot_table(index='orlogid_encoded',
                                                                                            columns='rxcui',
                                                                                            values='temp_const')
    home_meds_ohe = home_meds_ohe.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)
    home_meds_ohe.fillna(0, inplace=True)

    home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
    home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)
    home_meds_sum.fillna(0, inplace=True)  # setting the value for the ones that were added later

if 'pmh' in modality_to_use:
    pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert_wave2.csv')
    pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
    pmh_emb_sb = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)


if 'problist' in modality_to_use:
    prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert_wave2.csv')
    prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
    prob_list_emb_sb = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)


model_list = ['XGBT', 'TabNet', 'Scarf']
# model_list = ['TabNet']

# sav_dir = './Best_results/Preoperative/'
sav_dir = out_dir + 'Best_results/Preoperative/'
# best_file_name= path_to_dir + 'Best_trial_resulticu_TabNet_modal__preops_cbow_pmh_problist_homemeds174_24-07-17-10:55:55.json'
file_names = os.listdir(sav_dir)

## processing preops in a tabnet format



output_file_name = sav_dir + 'preops_metadata_' + str(args.task) + '.json'

# output_file_name = './preops_metadata' + str(args.task) + '.json'
md_f = open(output_file_name)
metadata = json.load(md_f)

if False:
    # encoding the plannedDispo from text to number
    # {"OUTPATIENT": 0, '23 HOUR ADMIT': 1, "FLOOR": 1, "OBS. UNIT": 2, "ICU": 3}
    preops_wave2.loc[preops_wave2['plannedDispo'] == 'Outpatient', 'plannedDispo'] = 0
    preops_wave2.loc[preops_wave2['plannedDispo'] == 'Floor', 'plannedDispo'] = 1
    preops_wave2.loc[preops_wave2['plannedDispo'] == 'Obs. unit', 'plannedDispo'] = 2
    preops_wave2.loc[preops_wave2['plannedDispo'] == 'ICU', 'plannedDispo'] = 3
    if '' in list(preops_wave2['plannedDispo'].unique()):
        preops_wave2.loc[preops_wave2['plannedDispo'] == '', 'plannedDispo'] = np.nan
    preops_wave2['plannedDispo'] = preops_wave2['plannedDispo'].astype('float') # needed to convert this to float because the nans were not getting converted to int and this variable is object type


    combined_with_outcome = outcome_df.merge(preops_wave2, on="orlogid_encoded", how="left")
    corr_outcome_features = combined_with_outcome.corr()
    corr_with_allfeatures = corr_outcome_features.loc['outcome'].sort_values(ascending=False)
    print(corr_with_allfeatures)
    print(list(corr_with_allfeatures.index)[-7:])
    exit()

processed_preops_wave2 = preprocess_inference(preops_wave2.copy(), metadata)

# Reading the models now
for m_name in model_list:
    modal_name = 'DataModal'
    for i in range(len(modality_to_use)):
        modal_name = modal_name + "_" + modality_to_use[i]
    dir_name = sav_dir + m_name + '/' + modal_name + "_" + str(args.task) +"/"

    if m_name=='TabNet':

        preops_wave2_t = preops_wave2.copy(deep=True)
        types = preops_wave2_t.dtypes


        output_file_name = dir_name + 'tabnet_feat_' + str(args.task) + '.pickle'
        with open(output_file_name, 'rb') as file: metadata_features = pickle.load(file)


        categorical_val_map = metadata_features['cat_dict_levels']
        nunique = metadata_features['n_col_unique']

        categorical_columns = metadata_features['cat_var']
        continuous_columns = metadata_features['cont_var']

        temp_list=[]
        for col in categorical_columns:
            if col in list(preops_wave2_t.columns):
                preops_wave2_t[col] = preops_wave2_t[col].replace(categorical_val_map[col])
            else:
                temp_list.append(col)

        for colname in continuous_columns:
            if colname in list(preops_wave2_t.columns):
                preops_wave2_t.fillna(preops_wave2_t[colname].mean(), inplace=True)
        preops_wave2_t = preops_wave2_t.reindex(columns=metadata_features['all_feat'], fill_value=0).reset_index(drop=True) # column that were not recorded in wave2
        for col in temp_list:
            preops_wave2_t[col] = max(list(categorical_val_map[col].values()))

    metadata_best_run_file = dir_name + '/Best_runs_metadata.pickle'
    with open(metadata_best_run_file, 'rb') as file:
        existing_data = pickle.load(file)

    best_5_random_number = [int(num.split("_")[-1]) for num in list(existing_data.keys())]

    if binary_outcome:
        perf_metric = np.zeros((len(best_5_random_number), 2))  # 2 is for the metrics auroc and auprc
    else:
        perf_metric = np.zeros((len(best_5_random_number), 5))  # 5 is for the metrics corr, corr_p, R2, MAE, MSE

    for runNum in range(len(best_5_random_number)):

        test_set = []
        if 'preops' in modality_to_use:
            if m_name=='TabNet':
                test_set.append(preops_wave2_t)
            else:
                test_set.append(processed_preops_wave2)
            test_set.append(cbow_wave2)

        if 'homemeds' in modality_to_use:
            hm_reading_form = existing_data['run_randomSeed_'+str(int(best_5_random_number[runNum]))]['hm_form']
            if hm_reading_form == 'ohe':
                home_meds_final = home_meds_ohe
            if hm_reading_form == 'embedding_sum':
                home_meds_final = home_meds_sum.copy().drop(["rxcui"], axis=1)

            test_set.append(home_meds_final)

        if 'pmh' in modality_to_use:
            new_name_pmh = ['pmh_sherbet'+str(num) for num in range(len(pmh_emb_sb.columns))]
            dict_name = dict(zip(pmh_emb_sb.columns,new_name_pmh ))
            test_set.append(pmh_emb_sb.copy().rename(columns=dict_name))

        if 'problist' in modality_to_use:
            new_name_prbl = ['prbl_sherbet'+str(num) for num in range(len(prob_list_emb_sb.columns))]
            dict_name = dict(zip(prob_list_emb_sb.columns,new_name_prbl ))
            test_set.append(prob_list_emb_sb.copy().rename(columns=dict_name))


        y_test = outcome_df_wave2["outcome"].values

        if m_name=='XGBT':
            feature_filename = dir_name + 'FittedFeatureNames_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".txt"
            file1 = open(feature_filename, "r+")
            feature_order = file1.read()
            extracted_list = ast.literal_eval(feature_order)
            if 'pmh' in modality_to_use:
                extracted_list = extracted_list[:-256] + new_name_pmh + new_name_prbl

            test_data = pd.concat(test_set, axis=1)
            test_data = test_data.reindex(columns=extracted_list)

            if (sum(test_data.isna().any()) > 0):  # this means that the column overlap between two waves is not full (mainly the homemeds as each homemed is a column)
                test_data.fillna(0, inplace=True)
            saving_path_name = dir_name + 'XGBT_BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".json"
            if binary_outcome:
                model = XGBClassifier()
            else:
                model = XGBRegressor()
            model.load_model(saving_path_name)

            # prediction on the loaded model
            if binary_outcome:
                pred_y_test = model.predict_proba(test_data)
                pred_y_test = pred_y_test[:, 1]
            else:
                pred_y_test = model.predict(test_data)

        if m_name =='TabNet':

            output_file_name = dir_name + 'tabnet_feat_' + str(args.task) + '_'+str(int(best_5_random_number[runNum]))+ '.json'
            md_features = open(output_file_name)
            metadata_features = json.load(md_features)

            tabnet_features = metadata_features['all_feat']

            saved_filepath = dir_name + 'BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name +".zip"

            tabnet_params = existing_data['run_randomSeed_' + str(int(best_5_random_number[runNum]))]['model_params']
            test_data = pd.concat(test_set, axis=1)

            test_data = test_data.reindex(columns=tabnet_features)
            if (sum(test_data.isna().any()) > 0):  # this means that the column overlap between two waves is not full (mainly the homemeds as each homemed is a column)
                test_data.fillna(0, inplace=True)

            if binary_outcome:
                loaded_clf = TabNetClassifier()
            else:
                loaded_clf = TabNetRegressor()
            loaded_clf.load_model(saved_filepath)
            # for i in range(len(loaded_clf.cat_idxs)): print(test_data.columns[loaded_clf.cat_idxs[i]], loaded_clf.cat_dims[i], len(np.unique(test_data1[:,loaded_clf.cat_idxs[i]])))

            if binary_outcome:
                pred_y_test = loaded_clf.predict_proba(test_data.values)
                pred_y_test = pred_y_test[:, 1]
            else:
                pred_y_test = loaded_clf.predict(test_data.values)

        if m_name=='Scarf':
            output_file_name = dir_name + 'scarf_feat_' + str(args.task) + '_'+str(int(best_5_random_number[runNum]))+ '.json'
            md_features = open(output_file_name)
            metadata_features = json.load(md_features)

            scarf_features = metadata_features['all_feat']

            if 'pmh' in modality_to_use:
                scarf_features = scarf_features[:-256] + new_name_pmh + new_name_prbl
            test_data = pd.concat(test_set, axis=1)
            test_data = test_data.reindex(columns=scarf_features)
            if (sum(test_data.isna().any()) > 0):  # this means that the column overlap between two waves is not full (mainly the homemeds as each homemed is a column)
                test_data.fillna(0, inplace=True)
            test_ds = SCARFDataset(test_data)

            param_values = existing_data['run_randomSeed_' + str(int(best_5_random_number[runNum]))]['model_params']
            batch_size_used = param_values['batchSize']
            device='cuda'
            model = SCARF(
                input_dim=param_values['input_dim'],
                features_low=param_values['features_low'],
                features_high=param_values['features_high'],
                dim_hidden_encoder=param_values['dim_hidden_encoder'],
                num_hidden_encoder=param_values['num_hidden_encoder'],
                dim_hidden_head=param_values['dim_hidden_head'],
                num_hidden_head=param_values['num_hidden_head'],
                corruption_rate=param_values['corruption_rate'],
                dropout=param_values['dropout'],
            ).to(device)

            saving_path_name = dir_name + 'BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".pkl"
            state_dict = torch.load(saving_path_name, map_location=device)
            model.load_state_dict(state_dict)

            ## this is needed because when the batch_size is 1, the batchnorm1d api breaks
            if divmod(test_ds.shape[0], batch_size_used)[1] ==1:
                test_ds = test_ds[:-1,:]
                y_test = y_test[:-1]
            test_loader = DataLoader(test_ds, batch_size=batch_size_used, shuffle=False)

            # get embeddings for training and test set
            test_embeddings = dataset_embeddings(model, test_loader, device)

            ## TODO: this xgb model is performing really bad in most cases. An alternative would be to train xgb on the test embeddings directly and report the performance.
            saving_xgb_file_name = dir_name + 'XGBT_BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".json"
            if binary_outcome:
                model_xgb = XGBClassifier()
            else:
                model_xgb = XGBRegressor()
            model_xgb.load_model(saving_xgb_file_name)

            # prediction on the loaded model
            if binary_outcome:
                pred_y_test = model_xgb.predict_proba(test_embeddings)
                pred_y_test = pred_y_test[:, 1]
            else:
                pred_y_test = model_xgb.predict(test_embeddings)

        if binary_outcome:
            test_auroc = roc_auc_score(y_test, pred_y_test)
            test_auprc = average_precision_score(y_test, pred_y_test)

            perf_metric[runNum, 0] = test_auroc
            perf_metric[runNum, 1] = test_auprc
        else:
            if pred_y_test.ndim == 2: pred_y_test = pred_y_test[:,0]  # to keep the shapes consistent for true and predicted values; mainly for tabnet
            corr_value = np.round(pearsonr(np.array(y_test), np.array(pred_y_test))[0], 3)
            cor_p_value = np.round(pearsonr(np.array(y_test), np.array(pred_y_test))[1], 3)
            print(str(args.task) + " prediction with correlation ", corr_value, ' and corr p value of ', cor_p_value)
            r2value = r2_score(np.array(y_test), np.array(pred_y_test))  # inbuilt function also exists for R2
            print(" Value of R2 ", r2value)
            temp_df = pd.DataFrame(columns=['true_value', 'pred_value'])
            temp_df['true_value'] = np.array(y_test)
            temp_df['pred_value'] = np.array(pred_y_test)
            temp_df['abs_diff'] = abs(temp_df['true_value'] - temp_df['pred_value'])
            temp_df['sqr_diff'] = (temp_df['true_value'] - temp_df['pred_value']) * (
                        temp_df['true_value'] - temp_df['pred_value'])
            mae_full = np.round(temp_df['abs_diff'].mean(), 3)
            mse_full = np.round(temp_df['sqr_diff'].mean(), 3)
            print("MAE on the test set ", mae_full)
            print("MSE on the test set ", mse_full)

            perf_metric[runNum, 0] = corr_value
            perf_metric[runNum, 1] = cor_p_value
            perf_metric[runNum, 2] = r2value
            perf_metric[runNum, 3] = mae_full
            perf_metric[runNum, 4] = mse_full

        print(perf_metric)

    print("Final performance metric", perf_metric)
    # saving the performance metrics from all best runs and all models in a pickle file
    perf_filename = sav_dir + str(args.task) + '_Best_perf_metrics_combined_preoperativeWave2.pickle'
    if not os.path.exists(perf_filename):
        data = {}
        data[str(m_name)] = {modal_name: perf_metric}
        with open(perf_filename, 'wb') as file:
            pickle.dump(data, file)
    else:
        with open(perf_filename, 'rb') as file:
            existing_data = pickle.load(file)

        try:
            existing_data[str(m_name)][modal_name] = perf_metric
        except(KeyError):  # this is to take care of the situation when a new model is added to the file
            existing_data[str(m_name)] = {}
            existing_data[str(m_name)][modal_name] = perf_metric

        # Save the updated dictionary back to the pickle file
        with open(perf_filename, 'wb') as file:
            pickle.dump(existing_data, file)

    print(" Model type :", m_name, " Modal name: ", modal_name, "  Finished for wave 2" )







