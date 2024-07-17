"""
This file will use the existing package of Tabnet from https://github.com/dreamquark-ai/tabnet and use it in a supervised manner for classifier.
The modalities will vary depending on the ablation need.

"""
# importing packages
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys, argparse
import scipy
from datetime import datetime
import json
from pytorch_tabnet.tab_model import TabNetClassifier


# starting time of the script
start_time = datetime.now()

parser = argparse.ArgumentParser(description='Tabular modular tabnet model training')

## modalities to select
parser.add_argument('--preops', default=True, action='store_true',
                    help='Whether to add preops and bow to ts representation')
parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
parser.add_argument('--homemeds', action="store_true",
                    help='Whether to add homemeds to ts representation')

## for the homemeds
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum'}

## learning parameters
parser.add_argument("--batchSize",  default=32, type=int) #
parser.add_argument("--epochs",  default=50, type=int) #


## task and setup parameters
parser.add_argument("--task",  default="icu") #
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--overSampling", default=True, action='store_true') # keep it as False when task is endofcase


## output parameters
parser.add_argument("--git",  default="") # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo",  default="") #
parser.add_argument("--outputcsv",  default="") #

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing

modality_to_use = []
if eval('args.preops') == True:
    modality_to_use.append('preops')
    modality_to_use.append('cbow')

if eval('args.pmhProblist') == True:
    modality_to_use.append('pmh')
    modality_to_use.append('problist')

if eval('args.homemeds') == True:
    modality_to_use.append('homemeds')

torch.manual_seed(args.randomSeed)
torch.cuda.manual_seed(args.randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.randomSeed)

data_dir = '/mnt/ris/ActFastExports/v1.3.2/'
# data_dir = '/input/'

out_dir = './'
# out_dir = '/output/'

preops = pd.read_csv(data_dir + 'epic_preop.csv')
outcomes = pd.read_csv(data_dir + 'epic_outcomes.csv')
end_of_case_times = outcomes[['orlogid_encoded', 'endtime']]


# end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration', 'n_glu_high',
                           'low_sbp_time','aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time', 'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1']
binary_outcome = args.task not in regression_outcome_list


outcomes = outcomes.dropna(subset=['ICU'])
outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

if args.task == 'endofcase':
    # updating the end_of_case_times targets for bigger distribution;
    """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 60] ## cases that are too short
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] < 25+511] ## cases that are too long
    end_of_case_times['true_test'] = end_of_case_times['endtime'] - 10
    end_of_case_times['t1'] = end_of_case_times['true_test'] -30
    end_of_case_times['t2'] = end_of_case_times['true_test'] -35 # temporary just to make sure nothing breaks; not being used
    end_of_case_times['t3'] = end_of_case_times['true_test'] -40 # temporary just to make sure nothing breaks; not being used
    overSampling = False  # TODO: there could be a better way to handle this.
else :
    end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 511)
    # end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 90)

binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac', 'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']

if args.task in regression_outcome_list:
    outcomes['survival_time'] = np.minimum(outcomes['survival_time'], 90)
    outcomes['readmission_survival'] = np.minimum(outcomes['readmission_survival'], 30)
    # outcomes['n_glucose_high'] = outcomes['n_glucose_high'].fillna(0)  # this might not be needed as already taken of by the where statement
    outcomes['n_glu_high'] = np.where(outcomes['N_glu_measured'] > 0, outcomes['n_glu_high']/outcomes['N_glu_measured'], 0)
    outcomes['total_blood'] = outcomes['total_blood'].fillna(0)
    outcomes['low_sbp_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_sbp_time']/outcomes['total_t'], 0)
    outcomes['low_relmap_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_time']/outcomes['total_t'], 0)
    outcomes['low_map_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_time']/outcomes['total_t'], 0)
    outcomes['aoc_low_sbp'] = np.where(outcomes['total_t'] > 0, outcomes['aoc_low_sbp'], 0)
    outcomes['low_relmap_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_aoc'], 0)
    outcomes['low_map_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_aoc'], 0)
    outcomes['postop_vent_duration'] = outcomes['postop_vent_duration'].fillna(0)
    outcomes['timew_pain_avg_0'] = outcomes['timew_pain_avg_0']/(outcomes['timew_pain_avg_0'].max() - outcomes['timew_pain_avg_0'].min())
    outcomes['median_pain_0'] = outcomes['median_pain_0']/(outcomes['median_pain_0'].max() - outcomes['median_pain_0'].min())
    outcomes['worst_pain_0'] = outcomes['worst_pain_0']/(outcomes['worst_pain_0'].max() - outcomes['worst_pain_0'].min())
    outcomes['worst_pain_1'] = outcomes['worst_pain_1']/(outcomes['worst_pain_1'].max() - outcomes['worst_pain_1'].min())

    outcome_df = outcomes[['orlogid_encoded', args.task]]
elif args.task in binary_outcome_list:
    if args.task == 'VTE':
        temp_outcome = outcomes[['orlogid_encoded']]
        temp_outcome[args.task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[args.task] == 2, args.task] = 1
    elif args.task == 'n_glucose_low':
        temp_outcome = outcomes[['orlogid_encoded', args.task]]
        temp_outcome[args.task] = temp_outcome[args.task].fillna(0)
        temp_outcome[args.task] = np.where(temp_outcome[args.task]>0, 1, 0)
    else:
        temp_outcome = outcomes[['orlogid_encoded', args.task]]
        temp_outcome.loc[temp_outcome[args.task] == True, args.task] = 1
        temp_outcome.loc[temp_outcome[args.task] == False, args.task] = 0
    temp_outcome[args.task] = temp_outcome[args.task].astype(int)
    outcome_df = temp_outcome
elif (args.task == 'dvt_pe'):
    dvt_pe_outcome = outcomes[['orlogid_encoded', 'DVT_PE']]
    outcome_df = dvt_pe_outcome
elif (args.task == 'icu'):
    icu_outcome = outcomes[['orlogid_encoded', 'ICU']]
    icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
    icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
    icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)
    outcome_df = icu_outcome
elif (args.task == 'mortality'):
    mortality_outcome = outcomes[['orlogid_encoded', 'death_in_30']]
    mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
    mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
    mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)
    outcome_df = mortality_outcome
elif (args.task == 'aki1' or args.task == 'aki2' or args.task == 'aki3'):
    aki_outcome = outcomes[['orlogid_encoded', 'post_aki_status']]
    aki_outcome = aki_outcome.dropna(subset=[
        'post_aki_status'])  # this is droping the patients with baseline kidney failure as they are now post_aki_status = NA_integer_
    if args.task == 'aki1':
        aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
        aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
    if args.task == 'aki2':
        aki_outcome.loc[aki_outcome[
                            'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
    if args.task == 'aki3':
        aki_outcome.loc[aki_outcome[
                            'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will become zero :(; there is a one liner too that can be used
        aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
    aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)
    outcome_df = aki_outcome
elif (args.task == 'endofcase'):
    outcome_df = end_of_case_times[['orlogid_encoded', 'true_test']]
else:
    raise Exception("outcome not handled")

## intersect 3 mandatory data sources: preop, outcome, case end times
combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(
    set(end_of_case_times['orlogid_encoded'].values)).intersection(
    set(preops['orlogid_encoded'].values)))

if False:
    combined_case_set = np.random.choice(combined_case_set, 5000, replace=False)

outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

# checking for NA and other filters
outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)   # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                     axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

train_set = []
test_set = []
valid_set = []

features = []
grouped_features = []
if 'preops' in modality_to_use:
    # this is being used because we will be adding the problem list and pmh as a seperate module in this file too
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

    preops = preops.drop(columns=to_drop_old_pmh_problist_with_others)

    # this is being done to remain consistent with other methods
    test_size = 0.2
    valid_size = 0.05  # change back to 0.00005 for the full dataset
    y_outcome = outcome_df["outcome"].values
    preops.reset_index(drop=True, inplace=True)
    upto_test_idx = int(test_size * len(preops))
    test = preops.iloc[:upto_test_idx]
    train0 = preops.iloc[upto_test_idx:]
    if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
        train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=args.randomSeed,
                                        stratify=y_outcome[train0.index])
    else:
        train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size), random_state=args.randomSeed)

    train_index = train.index
    valid_index = valid.index
    test_index = test.index

    if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
        test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 'ICU']['plannedDispo'].index


    nunique = preops.nunique()
    types = preops.dtypes

    categorical_columns = []
    continuous_columns = []
    categorical_dims = {}

    for col in preops.columns:
        if types[col] == 'object' or nunique[col] < 80:
            # print(col, preops[col].nunique(), preops[col].isna().any())
            if (preops[col].isna().any() == True) and types[col] == 'object':
                if preops[col].nunique() > 2:
                    preops[col] = preops[col].fillna("VV_likely")
            elif (preops[col].isna().any() == True) and types[col] == 'float':
                preops[col] = preops[col].fillna(float('nan'))
            l_enc = LabelEncoder()
            preops[col] = l_enc.fit_transform(preops[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            continuous_columns.append(col)

    for i in continuous_columns: preops.fillna(preops.loc[train_index, i].mean(), inplace=True)

    preops_tr = preops.iloc[train_index]
    preops_val = preops.iloc[valid_index]
    preops_te = preops.iloc[test_index]

    preops_tr.drop(columns="person_integer", inplace=True)
    preops_val.drop(columns="person_integer", inplace=True)
    preops_te.drop(columns="person_integer", inplace=True)

    input_shape_preops = len(preops_tr.columns)

    features = list(preops_tr.columns)

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    bow_input = pd.read_csv(data_dir + 'cbow_proc_text.csv')

    bow_input = bow_input.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(),preops.index.max()+1)),fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
    bow_cols = [col for col in bow_input.columns if 'BOW' in col]
    bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
    bow_input.fillna(0, inplace=True)

    bow_tr = bow_input.iloc[train_index]
    bow_val = bow_input.iloc[valid_index]
    bow_te = bow_input.iloc[test_index]

    input_shape_bow = len(bow_input.columns)

    grouped_features.append(list(np.arange(len(features), len(features)+input_shape_bow)))

    features = features + list(bow_input.columns)

    train_set.append(preops_tr)
    train_set.append(bow_tr)
    valid_set.append(preops_val)
    valid_set.append(bow_val)
    test_set.append(preops_te)
    test_set.append(bow_te)

if 'homemeds' in modality_to_use:
    # home meds reading and processing
    home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
    Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

    # home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
    # home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')
    home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded',
                                                  'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
    home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left', on='rxcui')
    home_meds_embedded.drop(columns=['code', 'description', 'source'], inplace=True)

    # breakpoint()
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
    home_meds_ohe.fillna(0, inplace=True)

    home_meds_ohe = home_meds_ohe.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"],
                                                                                                  axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    home_meds_ohe.fillna(0, inplace=True)  # setting the value for the ones that were added later

    home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
    home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"],
                                                                                                  axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    home_meds_sum.fillna(0, inplace=True)  # setting the value for the ones that were added later

    if args.home_medsform == 'ohe':
        home_meds_final = home_meds_ohe
    if args.home_medsform == 'embedding_sum':
        home_meds_sum = home_meds_sum.drop(["rxcui"], axis=1)
        home_meds_final = home_meds_sum

    hm_tr = home_meds_final.iloc[train_index]
    hm_te = home_meds_final.iloc[test_index]
    hm_val = home_meds_final.iloc[valid_index]
    hm_input_dim = len(home_meds_final.columns)

    grouped_features.append(list(np.arange(len(features), len(features)+hm_input_dim)))
    features = features + list(home_meds_final.columns)

    train_set.append(hm_tr)
    valid_set.append(hm_val)
    test_set.append(hm_te)

if 'pmh' in modality_to_use:

    pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert.csv')

    pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
    pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)

    pmh_tr = pmh_emb_sb_final.iloc[train_index]
    pmh_te = pmh_emb_sb_final.iloc[test_index]
    pmh_val = pmh_emb_sb_final.iloc[valid_index]
    pmh_input_dim = len(pmh_emb_sb_final.columns)

    train_set.append(pmh_tr)
    valid_set.append(pmh_val)
    test_set.append(pmh_te)

    grouped_features.append(list(np.arange(len(features), len(features)+pmh_input_dim)))

    features = features + list(pmh_emb_sb_final.columns)

if 'problist' in modality_to_use:
    prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')

    prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
    prob_list_emb_sb_final = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)

    problist_tr = prob_list_emb_sb_final.iloc[train_index]
    problist_te = prob_list_emb_sb_final.iloc[test_index]
    problist_val = prob_list_emb_sb_final.iloc[valid_index]
    problist_input_dim = len(prob_list_emb_sb_final.columns)

    train_set.append(problist_tr)
    valid_set.append(problist_val)
    test_set.append(problist_te)

    grouped_features.append(list(np.arange(len(features), len(features)+problist_input_dim)))

    features = features + list(prob_list_emb_sb_final.columns)

train_data = np.concatenate(train_set, axis=1)
valid_data = np.concatenate(valid_set,axis=1)
test_set = np.concatenate(test_set, axis=1)

tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":2,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax', # "sparsemax"
                 "grouped_features" : grouped_features,
                 'device_name':'cuda'
                }

clf = TabNetClassifier(**tabnet_params)

X_train = train_data
y_train = outcome_df.iloc[train_index]["outcome"].values

X_valid = valid_data
y_valid = outcome_df.iloc[valid_index]["outcome"].values

X_test = test_set
y_test = outcome_df.iloc[test_index]["outcome"].values


max_epochs = 50 if not os.getenv("CI", False) else 2

# This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
sparse_X_train = scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
sparse_X_valid = scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid

# Fitting the model
clf.fit(
    X_train=sparse_X_train, y_train=y_train,
    eval_set=[(sparse_X_train, y_train), (sparse_X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=max_epochs , patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
)

preds = clf.predict_proba(X_test)
test_auroc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
test_auprc = average_precision_score(y_score=preds[:,1], y_true=y_test)


preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

print(f"BEST VALID SCORE FOR : {clf.best_cost}")
print(f"FINAL TEST AUROC FOR : {test_auroc}")
print(f"FINAL TEST AUPRC FOR : {test_auprc}")

print(" Number of epochs that ran ", max_epochs)
print("Test AUROC and AUPRC values are ", np.round(test_auroc, 4), np.round(test_auprc, 4))
fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, preds[:,1], drop_intermediate=False)
precision_prc, recall_prc, thresholds_prc = precision_recall_curve(y_test, preds[:,1])
# interpolation in ROC
mean_fpr = np.linspace(0, 1, 100)
tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
mean_fpr = np.round(mean_fpr, decimals=2)
print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))

if args.task=='endofcase':
    outcome_rate = 0.5  # this is hardcoded here
else:
    outcome_rate = np.round(outcome_df.iloc[test_index]["outcome"].mean(), decimals=4)

end_time = datetime.now()  # only writing part is remaining in the code to time
timetaken = end_time-start_time
print("time taken to run the complete training script", timetaken)

csvdata = {
    'hp': json.dumps(vars(args)),
    'Initial_seed': args.randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
    'outcome_rate': outcome_rate,
    'AUROC': test_auroc,
    'AUPRC': test_auprc,
    'Sensitivity': tpr_inter[np.where(mean_fpr == 0.10)],
    'git': args.git,
    'name': args.nameinfo,
    'target': args.task,
    'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
    'time': timetaken
}

# breakpoint()
csvdata = pd.DataFrame(csvdata)
outputcsv = os.path.join(out_dir, args.outputcsv)
if (os.path.exists(outputcsv)):
    csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
else:
    csvdata.to_csv(outputcsv, header=True, index=False)