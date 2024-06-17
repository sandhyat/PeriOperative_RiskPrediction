import json
import os
import sys, argparse
import numpy as np
import pandas as pd

from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from datetime import datetime
from sklearn import ensemble
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from scipy.stats import qmc
from scipy.stats.stats import pearsonr
import Preops_processing as pps
import math

parser = argparse.ArgumentParser(description=' Tree models on preops, procedure text, home meds and possibly summarized intraop time series')

## task and setup parameters
parser.add_argument("--task", default="icu")  #
parser.add_argument("--binaryEOC", default=True, action='store_true')  #
parser.add_argument("--skipPreops", default=False, action='store_true')  # True value of this will only use bow
parser.add_argument("--sepPreopsBow", default=True,
                    action='store_true')  # True value of this variable would treat bow and preops as sep input and have different mlps for them. Also, both skipPreops and sepPreopsBow can't be True at the same time
parser.add_argument("--trainTime", default=True, action='store_true')
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum', 'None'}
parser.add_argument("--model", default='xgbt') # options {'gbdt', 'xgbt', 'lightgb', 'linear'} , linear represents the case where the booster in xgboost is linear
# parser.add_argument('--probType', default='reg') # options {''class', 'reg'}
# parser.add_argument('--morphineAsInput', default=False, action='store_true')  # can not be true when total morphine equivalent is being used as an outcome
parser.add_argument('--searchType', default='sobol') # options {'sobol', 'exhaustive'}
parser.add_argument('--poweroftwo_samples', default=6, type=int) # 2^(poweroftwo_samples) cases will be generated for HP tuning
parser.add_argument('--onlyProcedureTxt', default=False, action='store_true') # make sure the model is xgbt when running this because linear is just a baseline
parser.add_argument('--bestModel', default=True, action='store_true') # if you want to run HP or use the best model for which the HPS are hardcoded from earlier runs

# using time series summary
parser.add_argument('--TSMedAggType', default='None') # options {'None','SumOverTime', 'MedUnitComboSumOverTime'}
parser.add_argument("--endtimeMIN", default=30, type=int)  # cases filtered by min time duration
parser.add_argument("--FirstLastFlag", default=False, action='store_true') #  True if only first and last 45 minutes of the case need to be used for prediction; need endtimeMin to be atleast 90 mins

# using the pmmh and prob list embeddings from text
parser.add_argument("--BioGptProbPmh", default=True, action='store_true')
parser.add_argument("--biogptPromptType", default='Small')  # options {'Small', 'other'}
parser.add_argument("--BioGptProbPmh_agg", default='None')  # options {'None', 'Agg'}

## output parameters
parser.add_argument("--git", default="")  # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo", default="")  #
parser.add_argument("--outputcsv", default="")  #


args = parser.parse_args()
if __name__ == "__main__":
    globals().update(args.__dict__)  ## it would be better to change all the references to args.thing

np.random.seed(randomSeed)


data_dir = '/input/'

# reading the preop and outcome feather files
outcomes = pd.read_csv(data_dir + 'epic_outcomes.csv')

# home_meds
Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

if(BioGptProbPmh):
    preops = pd.read_csv(data_dir + 'epic_preop.csv')
    # to drop the old pmh and problem list
    to_drop_old_pmh_problist  = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                 "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                 "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose", 'pre_aki_status', 'preop_ICU', 'preop_los']

    preops = preops.drop(columns=to_drop_old_pmh_problist)
    breakpoint()
    # adding the biogpt embeddings for problem list and past medical history
    if biogptPromptType == 'Small':
        pmh_emb = pd.read_csv(data_dir+ 'pmh_small_biogpt.csv')
        prob_list_emb = pd.read_csv(data_dir+'preproblems_small_biogpt.csv')
    else:
        pmh_emb = pd.read_csv(data_dir+ 'pmh_biogpt.csv')
        prob_list_emb = pd.read_csv(data_dir+'preproblems_biogpt.csv')
else:
    if True:
        preops = pd.read_csv(data_dir + 'epic_preop.csv')
        # to drop the old pmh and problem list
        to_drop_old_pmh_problist_with_others = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                    "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                    "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose",
                                    'pre_aki_status', 'preop_ICU', 'preop_los', 'URINE UROBILINOGEN', 'MRN_encoded', 'time_of_day',
                                    'BACTERIA, URINE', 'CLARITY, URINE','COLOR, URINE','EPITHELIAL CELLS, SQUAMOUS, URINE',
                                    'GLUCOSE, URINE, QUALITATIVE','HYALINE CAST', 'LEUKOCYTE ESTERASE, URINE','PROTEIN, URINE QUALITATIVE',
                                    'RED BLOOD CELLS, URINE','URINE BLOOD', 'URINE KETONES', 'URINE NITRITE', 'URINE UROBILINOGEN','WHITE BLOOD CELLS, URINE']

        preops = preops.drop(columns=to_drop_old_pmh_problist_with_others)
    else:
        preops = feather.read_feather(data_dir + 'preops_reduced_for_training.feather')

preops = preops.drop(index = preops[preops['age']<18].index)

cbow_proc_text = pd.read_csv(data_dir+'cbow_proc_text.csv')

home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
# home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
# home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')


home_meds_embedded = home_meds[['orlogid_encoded','rxcui']].merge(Drg_pretrained_embedings, how='left', on='rxcui')
home_meds_embedded.drop(columns=['rxcui', 'code', 'description', 'source'], inplace= True)


home_meds_freq = home_meds[['orlogid_encoded','rxcui','Frequency']].pivot_table(index='orlogid_encoded', columns='rxcui', values='Frequency')
rxcui_freq = home_meds["rxcui"].value_counts().reset_index()
rxcui_freq = rxcui_freq.rename({'count':'rxcui_freq', 'rxcui':'rxcui'}, axis =1)
# rxcui_freq = rxcui_freq.rename({'rxcui':'rxcui_freq', 'index':'rxcui'}, axis =1)
home_meds_small =  home_meds[home_meds['rxcui'].isin(list(rxcui_freq[rxcui_freq['rxcui_freq']>100]['rxcui']))]
home_meds_small['temp_const'] = 1
home_meds_ohe = home_meds_small[['orlogid_encoded', 'rxcui','temp_const']].pivot_table(index='orlogid_encoded', columns='rxcui', values='temp_const')
home_meds_ohe.fillna(0, inplace=True)

# getting the person_integer ffor the outcomes large file
# outcomes.rename(columns={'orlogid': 'orlogid_encoded'}, inplace=True)
# outcomes = outcomes.join(epic_orlogids.set_index('orlogid_encoded'), on='orlogid_encoded')
# outcomes = outcomes.join(person_to_orlogid.set_index('orlogid'), on='orlogid')
# outcomes.drop(columns=epic_orlogids.columns, inplace=True)

# end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
end_of_case_times = outcomes[['orlogid_encoded', 'endtime']]

regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration',
                           'aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time', 'low_map_aoc', 'total_morphine_equivalent_dose_day1', 'case_duration']
binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac',
                       'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low','n_glu_high',
                       'severe_present_1', 'postop_del', 'low_sbp_time', 'worst_pain_1','worst_pain_0']

binary_outcome = task not in regression_outcome_list

outcomes = outcomes.dropna(subset=['ICU'])
# outcomes.drop_duplicates(subset=['orlogid_encoded'], inplace=True)
outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > endtimeMIN]

if task in ['postop_del', 'severe_present_1', 'worst_pain_1', 'worst_pain_0']:
    # dropping the nans for postop_del and severe_present_1. Can't wait until later as these need to be converted into integer
    outcomes = outcomes.dropna(subset=[task])


if task == 'endofcase':
    # updating the end_of_case_times targets for bigger distribution;
    """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 60] ## cases that are too short
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] < 25+511] ## cases that are too long
    end_of_case_times['true_test'] = end_of_case_times['endtime'] - 10
    end_of_case_times['t1'] = end_of_case_times['true_test'] -30
    end_of_case_times['t2'] = end_of_case_times['true_test'] -35 # temporary just to make sure nothing breaks; not being used
    end_of_case_times['t3'] = end_of_case_times['true_test'] -40 # temporary just to make sure nothing breaks; not being used
    overSampling = False  # TODO: there could be a better way to handle this.
    ## TODO: do something with very long cases
else :
    end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 511)
    # end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 90)


if FirstLastFlag == True: # this is being done to provide that extra information
    preops = preops.join(outcomes[['orlogid_encoded', 'case_duration']].set_index('orlogid_encoded'), on ='orlogid_encoded')

# outcome
icu_outcome = outcomes[['orlogid_encoded', 'ICU']]
icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)

mortality_outcome = outcomes[['orlogid_encoded', 'death_in_30']]
mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)

aki_outcome = outcomes[['orlogid_encoded', 'post_aki_status']]
if task == 'aki1':
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
    aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
if task == 'aki2':
    aki_outcome.loc[aki_outcome[
                        'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
if task == 'aki3':
    aki_outcome.loc[aki_outcome[
                        'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)

dvt_pe_outcome = outcomes[['orlogid_encoded', 'DVT_PE']]


if TSMedAggType in ['SumOverTime', 'MedUnitComboSumOverTime'] :
    # flowsheet data
    very_dense_flow = feather.read_feather(data_dir +"flow_ts/Imputed_very_dense_flow.feather")
    very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace=True)
    very_dense_flow = very_dense_flow.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
    very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint']]
    if FirstLastFlag == True:
        # dropping the rows that represent observations after first 45 minutes and before last 45 minutes
        very_dense_flow['temp'] = very_dense_flow['endtime']-45
        very_dense_flow = very_dense_flow.drop(very_dense_flow[(very_dense_flow['timepoint'] > 45) & (very_dense_flow['timepoint'] < very_dense_flow['temp'])].index)
        very_dense_flow.drop(["endtime", 'temp'], axis=1, inplace=True)
    else:
        very_dense_flow.drop(["endtime"], axis=1, inplace=True)

    other_intra_flow_wlabs = feather.read_feather(data_dir +"flow_ts/Imputed_other_flow.feather")
    other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index, inplace=True)
    other_intra_flow_wlabs = other_intra_flow_wlabs.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                          on="orlogid_encoded")
    other_intra_flow_wlabs = other_intra_flow_wlabs.loc[
        other_intra_flow_wlabs['endtime'] > other_intra_flow_wlabs['timepoint']]
    if FirstLastFlag == True:
        # dropping the rows that represent observations after first 45 minutes and before last 45 minutes
        other_intra_flow_wlabs['temp'] = other_intra_flow_wlabs['endtime']-45
        other_intra_flow_wlabs = other_intra_flow_wlabs.drop(other_intra_flow_wlabs[(other_intra_flow_wlabs['timepoint'] > 45) & (other_intra_flow_wlabs['timepoint'] < other_intra_flow_wlabs['temp'])].index)
        other_intra_flow_wlabs.drop(["endtime", 'temp'], axis=1, inplace=True)
    else:
        other_intra_flow_wlabs.drop(["endtime"], axis=1, inplace=True)


    # reading the med files
    all_med_data = feather.read_feather(data_dir + 'med_ts/intraop_meds_filterd.feather')
    all_med_data.drop(all_med_data[all_med_data['time'] > 511].index, inplace=True)
    all_med_data = all_med_data.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
    all_med_data = all_med_data.loc[all_med_data['endtime'] > all_med_data['time']]
    if FirstLastFlag == True:
        # dropping the rows that represent observations after forst 45 minutes anf before last 45 minutes
        all_med_data['temp'] = all_med_data['endtime']-45
        all_med_data = all_med_data.drop(all_med_data[(all_med_data['time'] > 45) & (all_med_data['time'] < all_med_data['temp'])].index)
        all_med_data.drop(["endtime", 'temp'], axis=1, inplace=True)
    else:
        all_med_data.drop(["endtime"], axis=1, inplace=True)


    ## Special med * unit comb encoding
    all_med_data['med_unit_comb'] = list(zip(all_med_data['med_integer'], all_med_data['unit_integer']))
    med_unit_coded, med_unit_unique_codes = pd.factorize(all_med_data['med_unit_comb'])
    all_med_data['med_unit_comb'] = med_unit_coded

    # a = pd.DataFrame(columns=['med_integer','unit_integer', 'med_unit_combo'])
    # a['med_integer'] = [ med_unit_unique_codes[i][0] for i in range(len(med_unit_unique_codes))]
    # a['unit_integer'] = [ med_unit_unique_codes[i][1] for i in range(len(med_unit_unique_codes))]
    # a['med_unit_combo'] = np.arange(len(med_unit_unique_codes))
    # a.sort_values(by=['med_integer','med_unit_combo'], inplace=True)
    #
    #
    # group_start = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']), 1)).nonzero().squeeze()  +1 # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.
    # group_end = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']), -1)).nonzero().squeeze() +1 # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.
    #
    # group_start = torch.cat((torch.tensor(0).reshape((1)), group_start)) # prepending 0 to make sure that it is treated as an empty slot
    # group_end = torch.cat((torch.tensor(0).reshape((1)), group_end)) # prepending 0 to make sure that it is treated as an empty slot

if TSMedAggType == 'SumOverTime':
    all_med_data['dose'] = all_med_data['dose'].astype('float')
    all_med_data_stat = all_med_data.groupby(by=['orlogid_encoded','med_integer'])['dose'].agg(['sum']).reset_index() # assuming that the the unit and route of a med for a patient was consistent
    all_med_data_stat.med_integer = all_med_data_stat['med_integer'].astype('str')
    all_med_data_stat = all_med_data_stat.pivot(index='orlogid_encoded', columns='med_integer', values=['sum']).reset_index()
    all_med_data_stat.columns = ['_'.join(col) for col in all_med_data_stat.columns]
    all_med_data_stat.fillna(0, inplace=True)
    temp_name = ['orlogid_encoded'] + [col+"Meds" for col in all_med_data_stat.columns if col not in ['orlogid_encoded_']]
    all_med_data_stat.rename(columns = dict(zip(all_med_data_stat.columns, temp_name)), inplace =True)

if TSMedAggType == 'MedUnitComboSumOverTime':
    all_med_data['dose'] = all_med_data['dose'].astype('float')
    all_med_data_stat = all_med_data.groupby(by=['orlogid_encoded','med_unit_comb'])['dose'].agg(['sum']).reset_index()  # the sum is over the time for each unique med unit combo
    all_med_data_stat.med_unit_comb = all_med_data_stat['med_unit_comb'].astype('str')
    all_med_data_stat = all_med_data_stat.pivot(index='orlogid_encoded', columns='med_unit_comb', values=['sum']).reset_index()
    all_med_data_stat.columns = ['_'.join(col) for col in all_med_data_stat.columns]
    all_med_data_stat.fillna(0, inplace=True)
    temp_name = ['orlogid_encoded'] + [col+"MedUnit" for col in all_med_data_stat.columns if col not in ['orlogid_encoded_']]
    all_med_data_stat.rename(columns = dict(zip(all_med_data_stat.columns, temp_name)), inplace =True)
    med_combo_redundant = list(all_med_data_stat.sum(axis=0).sort_values()[:23].index)  # dropping the med unti combo columns that do not have any recorded dosage
    all_med_data_stat = all_med_data_stat.drop(columns= med_combo_redundant)
    low_freq_rec = [i for i in all_med_data_stat.columns if np.count_nonzero(all_med_data_stat[i].to_numpy()) < 10] # med unit combo recorded in only a handful of patients
    all_med_data_stat = all_med_data_stat.drop(columns= low_freq_rec)

if TSMedAggType in ['SumOverTime', 'MedUnitComboSumOverTime'] :
    very_dense_flow_stat = very_dense_flow.groupby(by=['orlogid_encoded','measure_index'])['VALUE'].agg(['min','max', 'mean', 'var']).reset_index()
    very_dense_flow_stat['var'].fillna(0, inplace=True)
    very_dense_flow_stat.measure_index = very_dense_flow_stat['measure_index'].astype('str')
    very_dense_flow_stat = very_dense_flow_stat.pivot(index='orlogid_encoded', columns='measure_index', values=['min','max', 'mean', 'var']).reset_index()
    very_dense_flow_stat.columns = ['_'.join(col) for col in very_dense_flow_stat.columns]
    temp_name = ['orlogid_encoded'] + [col+"FlowD" for col in very_dense_flow_stat.columns if col not in ['orlogid_encoded_']]
    very_dense_flow_stat.rename(columns = dict(zip(very_dense_flow_stat.columns, temp_name)), inplace =True)


    other_intra_flow_wlabs_stat = other_intra_flow_wlabs.groupby(by=['orlogid_encoded','measure_index'])['VALUE'].agg(['min','max', 'mean', 'var', 'count']).reset_index()
    other_intra_flow_wlabs_stat['var'].fillna(0, inplace=True)
    other_intra_flow_wlabs_stat.measure_index = other_intra_flow_wlabs_stat['measure_index'].astype('str')
    other_intra_flow_wlabs_stat = other_intra_flow_wlabs_stat.pivot(index='orlogid_encoded', columns='measure_index', values=['min','max', 'mean', 'var', 'count']).reset_index()
    other_intra_flow_wlabs_stat.columns  = ['_'.join(col) for col in other_intra_flow_wlabs_stat.columns]
    temp_name = ['orlogid_encoded'] + [col+"FlowS" for col in other_intra_flow_wlabs_stat.columns if col not in ['orlogid_encoded_']]
    other_intra_flow_wlabs_stat.rename(columns = dict(zip(other_intra_flow_wlabs_stat.columns, temp_name)), inplace =True)



if task in regression_outcome_list:
    outcomes['survival_time'] = np.minimum(outcomes['survival_time'], 90)
    # outcomes['readmission_survival'] = np.minimum(outcomes['readmission_survival'], 30) # this is being commented because we are going to predict everything as regression and then evaluate as classification by thresholding at 30
    # outcomes['n_glucose_high'] = outcomes['n_glucose_high'].fillna(0)  # this might not be needed as already taken of by the where statement
    outcomes['n_glu_high'] = np.where(outcomes['N_glu_measured'] > 0, outcomes['n_glu_high']/outcomes['N_glu_measured'], 0)
    # outcomes['total_blood'] = outcomes['total_blood'].fillna(0) # no null observations
    # outcomes['low_sbp_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_sbp_time']/outcomes['total_t'], 0)
    outcomes['low_relmap_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_time']/outcomes['total_t'], 0)
    outcomes['low_map_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_time']/outcomes['total_t'], 0)
    # to check the following
    # outcomes['aoc_low_sbp'] = np.where(outcomes['total_t'] > 0, outcomes['aoc_low_sbp']/outcomes['total_t'], 0)
    # outcomes['low_relmap_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_aoc']/outcomes['total_t'], 0)
    # outcomes['low_map_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_aoc']/outcomes['total_t'], 0)
    # outcomes['postop_vent_duration'] = outcomes['postop_vent_duration'].fillna(0)

    outcome_df = outcomes[['orlogid_encoded', task]]
elif task in binary_outcome_list:
    if task == 'VTE':
        temp_outcome = outcomes[['orlogid_encoded']]
        temp_outcome[task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[task] == 2, task] = 1    # not the most efficient but spent more time than it needed
    elif task in  ['n_glucose_low', 'n_glu_high', 'low_sbp_time']:  # the threshold of 0 for low_sbp_time was decided as the 75th percentile (outcomes[task].describe()['75%'])
        temp_outcome = outcomes[['orlogid_encoded', task]]
        temp_outcome[task] = temp_outcome[task].fillna(0)
        temp_outcome[task] = np.where(temp_outcome[task]>0, 1, 0)
    elif task in ['worst_pain_1', 'worst_pain_0']:
        temp_outcome = outcomes[['orlogid_encoded', task]]
        temp_outcome[task] = np.where(temp_outcome[task]>=7, 1, 0)
    else:
        temp_outcome = outcomes[['orlogid_encoded', task]]
        temp_outcome.loc[temp_outcome[task] == True, task] = 1
        temp_outcome.loc[temp_outcome[task] == False, task] = 0
    temp_outcome[task] = temp_outcome[task].astype(int)
    outcome_df = temp_outcome
elif (task == 'dvt_pe'):
    outcome_df = dvt_pe_outcome
elif (task == 'icu'):
    outcome_df = icu_outcome
elif (task == 'mortality'):
    outcome_df = mortality_outcome
elif (task == 'aki1' or task == 'aki2' or task == 'aki3'):
    outcome_df = aki_outcome
elif (task == 'endofcase'):
    outcome_df = end_of_case_times[['orlogid_encoded', 'true_test']]
else:
    raise Exception("outcome not handled")


## intersect 3 mandatory data sources: preop, outcome, case end times
combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(
    set(end_of_case_times['orlogid_encoded'].values)).intersection(
    set(preops['orlogid_encoded'].values)))

if False:
    combined_case_set = np.random.choice(combined_case_set, 8000, replace=False)

outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

# outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1, inplace=True)
outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)


# checking for NA and other filters
outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)   # this df basically reindexes everything so from now onwards orlogid_encoded is an integer


preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                     axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

home_meds_ohe = new_index.merge(home_meds_ohe, on="orlogid_encoded", how="left").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)  # this is a left merge to make sure that all the patients exist in the home_meds_ohe_df
home_meds_ohe.fillna(0, inplace =True)  # setting the value for the ones that were added later

home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
home_meds_sum = new_index.merge(home_meds_sum, on="orlogid_encoded", how="left").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
home_meds_sum.fillna(0, inplace =True)

cbow_proc_text = cbow_proc_text.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

bow_cols = [col for col in cbow_proc_text.columns if 'BOW' in col]
cbow_proc_text['BOW_NA'] = np.where(np.isnan(cbow_proc_text[bow_cols[0]]), 1, 0)
cbow_proc_text.fillna(0, inplace=True)

if(BioGptProbPmh):
    pmh_emb = new_index.merge(pmh_emb, on="orlogid_encoded", how="left").drop(["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
    pmh_emb.fillna(0, inplace=True)  # setting the value for the ones that were added later

    prob_list_emb = new_index.merge(prob_list_emb, on="orlogid_encoded", how="left").drop(["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
    prob_list_emb.fillna(0, inplace =True)  # setting the value for the ones that were added later


outcome_df.drop(["orlogid_encoded"], axis=1, inplace=True)
outcome_df.reset_index(inplace=True)
outcome_df.rename({"index": "person_integer"}, axis=1, inplace=True)

if (TSMedAggType == 'SumOverTime') or (TSMedAggType == 'MedUnitComboSumOverTime'):
    all_med_data_stat = all_med_data_stat.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                                   axis=1).rename(
        {"new_person": "person_integer"}, axis=1)
    very_dense_flow_stat = very_dense_flow_stat.merge(new_index, on="orlogid_encoded", how="inner").drop(
        ["orlogid_encoded"],
        axis=1).rename(
        {"new_person": "person_integer"}, axis=1)
    other_intra_flow_wlabs_stat = other_intra_flow_wlabs_stat.merge(new_index, on="orlogid_encoded", how="inner").drop(
        ["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)

    preops = pd.concat([preops.set_index('person_integer'), all_med_data_stat.set_index('person_integer'), very_dense_flow_stat.set_index('person_integer'), other_intra_flow_wlabs_stat.set_index('person_integer')], axis=1).reset_index()


if task in ['postop_del', 'severe_present_1', 'worst_pain_1', 'worst_pain_0']:  # this number is so small because we are currently ignoring the validation set
    validation_size = 0.0005
else:
    validation_size = 0.00005

preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = pps.preprocess_train(preops, skipPreops,task,
                                                                                                           y_outcome=
                                                                                                           outcome_df[
                                                                                                               "outcome"].values,
                                                                                                           binary_outcome=binary_outcome,
                                                                                                           valid_size=validation_size)

if task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
    test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo']!=3]['plannedDispo'].index
    preops_te = preops_te.iloc[test_index]


if task == 'post_dialysis':
    test_index = preops_te[(preops_te['ESRD'] != 1) & (preops_te['Dialysis'] != 1)].index   # dropping every row in which at least one value is 1
    preops_te = preops_te[(preops_te['ESRD'] != 1) & (preops_te['Dialysis'] != 1)]


if searchType == 'exhaustive':
    # Exhaustive grid based search
    home_meds_flag = ['with_hm', 'wo_hm']
    n_estimatorslist = [100,200,500]
    max_depthlist = [2, 4, 6]
    learning_ratelist = [0.001, 0.01, 0.1,0.5,1.0]
    L1_reg = [0, .0001, 0.001, 0.005, .01 ]
    L2_reg = [0, .0001, 0.001, 0.005, .01 ]

if searchType == 'sobol':
    # SOBOL sequence based HP search
    home_meds_flag = ['with_hm', 'wo_hm']
    ## bounds are inclusive for integers
    ## third element is 1 if is an integer, 2 binary (so no value is passed)
    ## fourth element 1 if on log scale
    bounds = { 'n_estimator':[100, 500, 1, 0 ],
               'max_depthlist':[2, 8, 1, 0 ],
               'learningRate':[.0001, .01, 0 , 1],
               'L2_weight':[.0001, .01, 0 , 1],
               'L1_weight':[.0001, .01, 0 , 1]
      }
    sampler = qmc.Sobol(d =len(bounds.keys()))
    sample = sampler.random_base2(m=poweroftwo_samples)

    n_estimatorslist = [math.floor(bounds['n_estimator'][0] + samplei * (
                        bounds['n_estimator'][1] - bounds['n_estimator'][0] + bounds['n_estimator'][2]) )for samplei in sample[:,0]]
    max_depthlist = [math.floor(bounds['max_depthlist'][0] + samplei * (
                        bounds['max_depthlist'][1] - bounds['max_depthlist'][0] + bounds['max_depthlist'][2])) for samplei in sample[:,1]]
    learning_ratelist = [round(bounds['learningRate'][0] * math.exp(
                samplei * (math.log(bounds['learningRate'][1]) - math.log(bounds['learningRate'][0] + bounds['learningRate'][2]))), 4)  for samplei in sample[:,2]]
    L1_reg = [round(bounds['L1_weight'][0] * math.exp(
                samplei * (math.log(bounds['L1_weight'][1]) - math.log(bounds['L1_weight'][0] + bounds['L1_weight'][2]))), 4)  for samplei in sample[:,3]]
    L2_reg = [round(bounds['L2_weight'][0] * math.exp(
                samplei * (math.log(bounds['L2_weight'][1]) - math.log(bounds['L2_weight'][0] + bounds['L2_weight'][2]))), 4)  for samplei in sample[:,4]]

if(bestModel):   # temporary to be used when need to run the best model

    # best_hp_dict0 = {'icu':[346, 8, 0.0091, 0.0036, 0.0054], 'mortality': [479, 7, 0.0071, 0.0066, 0.0002], 'aki1': [457, 8, 0.0087, 0.0024, 0.0035],
    #                 'aki2':[492, 4, 0.0093, 0.0012, 0.0017], 'aki3':[432, 7, 0.0091, 0.0009, 0.0008], 'dvt_pe': [403, 6, 0.0082, 0.0032, 0.0011],
    #                 'AF': [338, 6, 0.0092, 0.0009, 0.0015],'cardiac':[482, 2, 0.0097, 0.0009, 0.0008], 'CVA':[322, 6, 0.0091, 0.0015, 0.0013],
    #                 'DVT':[293, 7, 0.0095, 0.0017, 0.004], 'GI': [483, 8, 0.0072, 0.0011, 0.0054], 'PNA': [408, 5, 0.0091, 0.001, 0.0001],
    #                 'UTI': [466, 5, 0.0083, 0.0005, 0.0014], 'VTE': [364, 4, 0.0088, 0.0002, 0.001], 'postop_trop_crit': [422, 7, 0.0086, 0.0015, 0.0001],
    #                 'postop_trop_high': [460, 6, 0.0062, 0.0079, 0.0001], 'post_dialysis':[403, 6, 0.0072, 0.0099, 0.0093], 'postop_del':[301, 5, 0.0092, 0.0012, 0.0017],
    #                 'severe_present_1':[499, 4, 0.0081, 0.0007, 0.0047]}
    best_hp_dict = {'icu':[346, 8, 0.0091, 0.0036, 0.0054], 'mortality': [479, 7, 0.0071, 0.0066, 0.0002], 'aki1': [457, 8, 0.0087, 0.0024, 0.0035],
                    'aki2':[492, 4, 0.0093, 0.0012, 0.0017], 'aki3':[432, 7, 0.0091, 0.0009, 0.0008], 'dvt_pe': [403, 6, 0.0082, 0.0032, 0.0011],
                    'AF': [338, 6, 0.0092, 0.0009, 0.0015],'cardiac':[409, 5, 0.0093, 0.0003, 0.0011], 'CVA':[322, 6, 0.0091, 0.0015, 0.0013],
                    'DVT':[293, 7, 0.0095, 0.0017, 0.004], 'GI': [483, 8, 0.0072, 0.0011, 0.0054], 'PNA': [405, 8, 0.0099, 0.0002, 0.0002],
                    'UTI': [498, 7, 0.0077, 0.0001, 0.0026], 'VTE': [435, 7, 0.0094, 0.003, 0.0006], 'postop_trop_crit': [339, 7, 0.0071, 0.0045, 0.0011],
                    'postop_trop_high': [494, 7, 0.0061, 0.0036, 0.0018], 'post_dialysis':[403, 6, 0.0072, 0.0099, 0.0093], 'postop_del':[265, 7, 0.0088, 0.0008, 0.0002],
                    'severe_present_1':[499, 4, 0.0081, 0.0007, 0.0047], 'PE':[486, 8, 0.0071, 0.0002, 0.002], 'worst_pain_1':[379, 7, 0.009, 0.0003, 0.0036]
, 'worst_pain_0': [468, 8, 0.0096, 0.0032, 0.0011]}
    n_estimatorslist = [best_hp_dict[task][0]] # n_estimatorss
    max_depthlist = [best_hp_dict[task][1]] # max_depth
    learning_ratelist = [best_hp_dict[task][2]] # learning_rate
    L1_reg = [best_hp_dict[task][3]] # L1_w
    L2_reg = [best_hp_dict[task][4]] # L2_w
    home_meds_flag = ['with_hm'] # including the home meds embedding or not
# models = ['gbdt', 'lr', 'xgbt', 'lightgb']

if(onlyProcedureTxt):  # this is done to make sure th loop only runs once
    home_meds_flag = ['wo_hm']


if(binary_outcome):

    if task == 'endofcase':
        true_y_train = np.repeat([1,0], len(train_index))
        true_y_test =  np.repeat([1,0], len(test_index))
    else:
        true_y_train = outcome_df.iloc[train_index]["outcome"].values
        true_y_test = outcome_df.iloc[test_index]["outcome"].values

    cbow_proc_text_train = cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).values
    cbow_proc_text_test = cbow_proc_text.iloc[test_index].drop(columns=['person_integer']).values

    feature_names_order = list(preops_tr.columns) + list(cbow_proc_text.drop(columns=['person_integer']).columns)
    print(feature_names_order)


    if task == 'endofcase':
        preops_tr = np.vstack((np.hstack((preops_tr.values, np.reshape(endtimes.iloc[train_index]["true_test"].values, (len(train_index),1)))),  np.hstack((preops_tr.values, np.reshape(endtimes.iloc[train_index]["t1"].values, (len(train_index),1))))))
        preops_te = np.vstack((np.hstack((preops_te.values, np.reshape(endtimes.iloc[test_index]["true_test"].values, (len(test_index),1)))),  np.hstack((preops_te.values, np.reshape(endtimes.iloc[test_index]["t1"].values, (len(test_index),1))))))


    # appending the word embeddings to the preops
    preops_tr = np.concatenate((preops_tr, cbow_proc_text_train), axis=1)
    preops_te = np.concatenate((preops_te, cbow_proc_text_test), axis=1)

    if (BioGptProbPmh):
        pmh_emb_train = pmh_emb.iloc[train_index].drop(columns=['person_integer']).values
        pmh_emb_test = pmh_emb.iloc[test_index].drop(columns=['person_integer']).values

        prob_list_emb_train = prob_list_emb.iloc[train_index].drop(columns=['person_integer']).values
        prob_list_emb_test = prob_list_emb.iloc[test_index].drop(columns=['person_integer']).values
        # breakpoint()
        # appending the past medical history and problem list embeddings to the preops
        if BioGptProbPmh_agg == 'Agg':
            preops_tr = np.concatenate((preops_tr, np.sum(pmh_emb_train, axis=1).reshape(len(train_index), 1),
                                        np.sum(prob_list_emb_train, axis=1).reshape(len(train_index), 1)), axis=1)
            preops_te = np.concatenate((preops_te, np.sum(pmh_emb_test, axis=1).reshape(len(test_index), 1),
                                        np.sum(prob_list_emb_test, axis=1).reshape(len(test_index), 1)), axis=1)
        else:
            preops_tr = np.concatenate((preops_tr, pmh_emb_train, prob_list_emb_train), axis=1)
            preops_te = np.concatenate((preops_te, pmh_emb_test, prob_list_emb_test), axis=1)
        # breakpoint()

    if home_medsform == 'ohe':
        home_meds_train = home_meds_ohe.iloc[train_index].drop(columns=['person_integer']).values
        home_meds_test = home_meds_ohe.iloc[test_index].drop(columns=['person_integer']).values

        feature_names_order = list(preops_val.columns) + list(
            cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).columns) + list(
            home_meds_ohe.iloc[train_index].drop(columns=['person_integer']).columns)

    elif home_medsform == 'embedding_sum':
        home_meds_train = home_meds_sum.iloc[train_index].drop(columns=['person_integer']).values
        home_meds_test = home_meds_sum.iloc[test_index].drop(columns=['person_integer']).values

        feature_names_order = list(preops_val.columns) + list(
            cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).columns) + list(
            home_meds_sum.iloc[train_index].drop(columns=['person_integer']).columns)

    import itertools

    if searchType == 'sobol':
        iterator_def = itertools.zip_longest(n_estimatorslist, max_depthlist, learning_ratelist, L1_reg, L2_reg)
    else:
        iterator_def = itertools.product(n_estimatorslist, max_depthlist, learning_ratelist, L1_reg, L2_reg)

    for (n_est, m_d, lr, L1_w, L2_w) in iterator_def:

        for hm in home_meds_flag:

            if model == 'gbdt':
                clf = ensemble.GradientBoostingClassifier(n_estimators=n_est, max_depth=m_d, learning_rate=lr, random_state=42)
            if model == 'lr':
                clf = LRCV(random_state=0, cv =5)
            if model == 'xgbt':
                clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w, reg_alpha=L1_w, random_state=42)
            if model == 'linear':
                clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w,
                                        reg_alpha=L1_w,
                                        random_state=42, booster='gblinear')
            if model =='lightgb':
                clf = lgb.LGBMClassifier(n_estimators=n_est,max_depth=m_d, learning_rate=lr,  reg_lambda=L2_w, reg_alpha=L1_w,random_state=42)

            if (onlyProcedureTxt):
                X_tr = cbow_proc_text_train
                X_te = cbow_proc_text_test
            else:
                if hm == 'with_hm':
                    X_tr = np.concatenate((preops_tr, home_meds_train), axis=1)
                    X_te = np.concatenate((preops_te, home_meds_test), axis=1)
                else:
                    X_tr = preops_tr
                    X_te = preops_te


            clf.fit(X_tr, true_y_train)
            if False:
                from sklearn.calibration import CalibratedClassifierCV

                calibrated_clf = CalibratedClassifierCV(clf, cv="prefit")
                calibrated_clf.fit(X_te[:5000,:], true_y_test[:5000])
                pred_y_test_Cal = calibrated_clf.predict_proba(X_te[5000:,:])[:,1]

            pred_y_test = clf.predict_proba(X_te)
            pred_y_test = pred_y_test[:, 1]
            # breakpoint()
            if(bestModel): # temporary to be used when need to run the best model

                # saving the feature name order and the training csv that was used to train the best model
                with open("./fitter_feature_names_" + str(task) + ".txt", 'w') as txt_to_write:
                # with open("/output_model_related/fitter_feature_names_" + str(task) + ".txt", 'w') as txt_to_write:
                    txt_to_write.write(str(feature_names_order))
                train_to_save = pd.DataFrame(X_tr, columns=feature_names_order)
                train_to_save.to_csv('./' + str(task) + "_train.csv", index=False)
                # train_to_save.to_csv('/output_model_related/'+str(task)+"_train.csv", index= False)
                test_to_save = pd.DataFrame(X_te, columns=feature_names_order)
                # test_to_save.to_csv('/output_model_related/' + str(task) + "_test.csv", index=False)
                test_to_save.to_csv('./' + str(task) + "_test.csv", index=False)
                clf.save_model('./' +
                               "BestXgBoost_model_" + task + "_" + hm + "_" + home_medsform + "-" + str(
                    BioGptProbPmh_agg) + ".json")
                # clf.save_model('/output_model_related/'+
                #     "BestXgBoost_model_" + task + "_" + hm + "_" + home_medsform +  "-" + str(BioGptProbPmh_agg) +".json")
                import shap
                explainer = shap.TreeExplainer(model=clf, data=None, model_output='raw',
                                               feature_perturbation='tree_path_dependent')
                shap_values_test = explainer.shap_values(X_te)
                np.savetxt("./Shap_on_x_valid_" + str(model) + "_" + str(task) + ".csv",
                           shap_values_test, delimiter=",")
                # np.savetxt('/output_model_related/'+"Shap_on_x_valid_" + str(model) + "_" + str(task) + ".csv",
                #            shap_values_test, delimiter=",")

                import pickle

                # filename_expl = '/output_model_related/'+ "ShapTreeExplainer" + str(model) + "_" + str(task) + ".sav"
                # pickle.dump(explainer, open(filename_expl, 'wb'))
                # load_explainer = pickle.load(open(filename_expl, 'rb'))
                # print(load_explainer)



                # for model card computation
                basic_features_surg = [i for i in feature_names_order if 'SurgService_Name' in i]
                basic_features_race = [i for i in feature_names_order if 'RACE' in i]
                basic_features_sex = [i for i in feature_names_order if 'Sex' in i]

                basic_features = basic_features_surg + basic_features_race + basic_features_sex + ['age']

                # subsetting the dataset
                x_train_few = train_to_save[basic_features]
                x_test_few = test_to_save[basic_features]
                clf_level1 = xgb.XGBClassifier(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w, reg_alpha=L1_w, random_state=42)
                clf_level1.fit(x_train_few, true_y_train)


                file_to_save_df = pd.DataFrame(columns=['new_person', 'true_y', 'pred_y', 'train id or not', 'pred_y_basic'])
                file_to_save_df['new_person'] = train_index
                file_to_save_df['true_y'] = true_y_train
                file_to_save_df['pred_y'] = clf.predict_proba(X_tr)[:, 1]
                file_to_save_df['train id or not'] = 1
                file_to_save_df['pred_y_basic'] = clf_level1.predict_proba(x_train_few.values)[:,1]

                file_to_save_df_te = pd.DataFrame(columns=['new_person', 'true_y', 'pred_y', 'train id or not', 'pred_y_basic'])  # test prediction saving
                file_to_save_df_te['new_person'] = test_index
                file_to_save_df_te['true_y'] = true_y_test
                file_to_save_df_te['pred_y'] = pred_y_test
                file_to_save_df_te['train id or not'] = 0
                file_to_save_df_te['pred_y_basic'] = clf_level1.predict_proba(x_test_few.values)[:,1]


                file_to_save_df = pd.concat([file_to_save_df, file_to_save_df_te], axis=0)
                Pred_to_save_df = file_to_save_df.merge(new_index, on=['new_person'], how='left').drop(columns=['new_person'])
                # Pred_to_save_df.to_csv('/output_model_related/' + 'Best_Pred_file_Classification_' + task + "_" + hm + "_" + home_medsform +  "-" + str(BioGptProbPmh_agg) +".csv", index=False)
                Pred_to_save_df.to_csv('./' + 'Best_Pred_file_Classification_' + task + "_" + hm + "_" + home_medsform +  "-" + str(BioGptProbPmh_agg) +".csv", index=False)

            if False:
                print("subgrouping process")
                grouped_outcome_Rate_te = pd.DataFrame(columns=['new_person', 'true_ICU_y', 'pred_ICU_y', '15%DicICU_pred' ])  # test prediction saving
                grouped_outcome_Rate_te['new_person'] = test_index
                grouped_outcome_Rate_te['true_ICU_y'] = true_y_test
                grouped_outcome_Rate_te['pred_ICU_y'] = pred_y_test

                # excluding the patients with high pred prob of icu with the argument that they are probably the cases where the assesment person didn't enter the plannned dispoosition
                grouped_outcome_Rate_te = grouped_outcome_Rate_te[grouped_outcome_Rate_te['pred_ICU_y'] <= 0.75]

                grouped_outcome_Rate_te['15%DicICU_pred'] = np.where(grouped_outcome_Rate_te['pred_ICU_y'] > grouped_outcome_Rate_te['pred_ICU_y'].describe(percentiles=[.25,.5,.75, .85])['85%'], 1, 0)
                grouped_outcome_Rate_te = grouped_outcome_Rate_te.merge(new_index, on=['new_person'], how='left').drop(columns=['new_person'])
                # adding the AKI, mortality, and los outcome
                grouped_outcome_Rate_te = grouped_outcome_Rate_te.merge(outcomes[['orlogid_encoded','death_in_30', 'postop_los', 'post_aki_status']], on=['orlogid_encoded'], how='left')
                grouped_outcome_Rate_te['aki2'] = np.where(grouped_outcome_Rate_te['post_aki_status'] >=2, 1, 0) # specific for aki2
                grouped_outcome_Rate_te.to_csv('./Outcome_Rate_subgrouped_' + task + "_" + hm + "_" + home_medsform + ".csv", index=False)

                final_grouped_outcome_Rate_sum = grouped_outcome_Rate_te.groupby(by=['15%DicICU_pred']).agg(['mean', 'std', 'count']).reset_index()


            # model performance on the test set
            test_auroc = roc_auc_score(true_y_test, pred_y_test)
            test_auprc = average_precision_score(true_y_test, pred_y_test)
            print("Test AUROC and AUPRC values are ", np.round(test_auroc, 4), np.round(test_auprc, 4))
            fpr_roc, tpr_roc, thresholds_roc = roc_curve(true_y_test, pred_y_test, drop_intermediate=False)
            precision_prc, recall_prc, thresholds_prc = precision_recall_curve(true_y_test, pred_y_test)
            # interpolation in ROC
            mean_fpr = np.linspace(0, 1, 100)
            tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
            mean_fpr = np.round(mean_fpr, decimals=2)
            print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))
            # breakpoint()
            if (onlyProcedureTxt):
                hm = 'onlyProcedureTxt'

            csvdata = {
                'hp': json.dumps(vars(args)),
                'n_estimator/max_depth/learningrate/L1_w/L2_w': str([n_est, m_d, lr, L1_w, L2_w]),
                'outcome_rate': np.round(sum(outcome_df["outcome"].values) / len(outcome_df), decimals=4),
                'AUROC': test_auroc,
                'AUPRC': test_auprc,
                'Sensitivity': tpr_inter[np.where(mean_fpr == 0.10)],
                'model': model,
                'name': hm,
                'target': args.task,
                'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S")
            }

            csvdata = pd.DataFrame(csvdata)
            outputcsv = os.path.join('/output/', args.outputcsv)
            if (os.path.exists(outputcsv)):
                csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
            else:
                csvdata.to_csv(outputcsv, header=True, index=False)


else:

    true_y_train = outcome_df.iloc[train_index]["outcome"].values
    true_y_test = outcome_df.iloc[test_index]["outcome"].values

    cbow_proc_text_train = cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).values
    cbow_proc_text_test = cbow_proc_text.iloc[test_index].drop(columns=['person_integer']).values

    # appending the word embeddings to the preops
    preops_tr = np.concatenate((preops_tr, cbow_proc_text_train), axis=1)
    preops_te = np.concatenate((preops_te, cbow_proc_text_test), axis=1)

    if home_medsform == 'ohe':
        home_meds_train = home_meds_ohe.iloc[train_index].drop(columns=['person_integer']).values
        home_meds_test = home_meds_ohe.iloc[test_index].drop(columns=['person_integer']).values

        feature_names_order = list(preops_val.columns) + list(
            cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).columns) + list(
            home_meds_ohe.iloc[train_index].drop(columns=['person_integer']).columns)

    elif home_medsform == 'embedding_sum':
        home_meds_train = home_meds_sum.iloc[train_index].drop(columns=['person_integer']).values
        home_meds_test = home_meds_sum.iloc[test_index].drop(columns=['person_integer']).values

        feature_names_order = list(preops_val.columns) + list(
            cbow_proc_text.iloc[train_index].drop(columns=['person_integer']).columns) + list(
            home_meds_sum.iloc[train_index].drop(columns=['person_integer']).columns)

    import itertools

    if searchType == 'sobol':
        iterator_def = itertools.zip_longest(n_estimatorslist, max_depthlist, learning_ratelist, L1_reg, L2_reg)
    else:
        iterator_def = itertools.product(n_estimatorslist, max_depthlist, learning_ratelist, L1_reg, L2_reg)

    for (n_est, m_d, lr, L1_w, L2_w) in iterator_def:

        for hm in home_meds_flag:

            if model == 'gbdt':
                reg = ensemble.GradientBoostingRegressor(n_estimators=n_est, max_depth=m_d, learning_rate=lr,
                                                         random_state=42)
            if model == 'xgbt':
                reg = xgb.XGBRegressor(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w, reg_alpha=L1_w, random_state=42)
            if model == 'linear':
                reg = xgb.XGBRegressor(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w,
                                        reg_alpha=L1_w,
                                        random_state=42, booster='gblinear')
            if model == 'lightgb':
                clf = lgb.LGBMRegressor(n_estimators=n_est, max_depth=m_d, learning_rate=lr, reg_lambda=L2_w,
                                         reg_alpha=L1_w, random_state=42)

            if (onlyProcedureTxt):
                X_tr = cbow_proc_text_train
                X_te = cbow_proc_text_test
            else:
                if hm == 'with_hm':
                    X_tr = np.concatenate((preops_tr, home_meds_train), axis=1)
                    X_te = np.concatenate((preops_te, home_meds_test), axis=1)
                else:
                    X_tr = preops_tr
                    X_te = preops_te

            reg.fit(X_tr, true_y_train)
            pred_y_test = reg.predict(X_te)


            corr_value = np.round(pearsonr(np.array(true_y_test), np.array(pred_y_test))[0], 3)
            cor_p_value = np.round(pearsonr(np.array(true_y_test), np.array(pred_y_test))[1], 3)
            print(str(task) + " prediction with correlation ", corr_value, ' and corr p value of ', cor_p_value)
            r2value = r2_score(np.array(true_y_test), np.array(pred_y_test))  # inbuilt function also exists for R2
            print(" Value of R2 ", r2value)
            temp_df = pd.DataFrame(columns=['true_value', 'pred_value'])
            temp_df['true_value'] = np.array(true_y_test)
            temp_df['pred_value'] = np.array(pred_y_test)
            temp_df['abs_diff'] = abs(temp_df['true_value'] - temp_df['pred_value'])
            temp_df['sqr_diff'] = (temp_df['true_value'] - temp_df['pred_value']) * (
                        temp_df['true_value'] - temp_df['pred_value'])
            mae_full = np.round(temp_df['abs_diff'].mean(), 3)
            mse_full = np.round(temp_df['sqr_diff'].mean(), 3)
            print("MAE on the test set ", mae_full)
            print("MSE on the test set ", mse_full)
            q25, q7, q9 = temp_df['true_value'].quantile([0.25, 0.7, 0.9])
            firstP_data = temp_df.query('true_value<={high}'.format(high=q25))
            secondP_data = temp_df.query('{low}<true_value<={high}'.format(low=q25, high=q7))
            thirdP_data = temp_df.query('{low}<true_value<={high}'.format(low=q7, high=q9))
            fourthP_data = temp_df.query('{low}<true_value'.format(low=q9))

            mae_dict = {'<' + str(np.round(q25, decimals=1)): firstP_data['abs_diff'].mean(),
                        str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): secondP_data['abs_diff'].mean(),
                        str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): thirdP_data['abs_diff'].mean(),
                        str(np.round(q9, decimals=1)) + "<": fourthP_data['abs_diff'].mean()}

            stratifying_point_dict = {'<' + str(np.round(q25, decimals=1)): '<' + str(np.round(q25, decimals=1)),
                                      str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): str(
                                          np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)),
                                      str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): str(
                                          np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)),
                                      str(np.round(q9, decimals=1)) + "<": str(np.round(q9, decimals=1)) + "<"}

            # evaluating as classification
            if task == 'total_morphine_equivalent_dose_day1':
                pred_y_test1 = np.where(pred_y_test > outcomes[task].describe()['75%'], 1, 0)
                true_y_test1 = np.where(true_y_test > outcomes[task].describe()['75%'], 1, 0)
            if task == 'total_blood':
                pred_y_test1 = np.where(pred_y_test > 0, 1, 0)
                true_y_test1 = np.where(true_y_test > 0, 1, 0)
            if task == 'readmission_survival':
                pred_y_test1 = np.where(pred_y_test < 30, 1, 0)
                true_y_test1 = np.where(true_y_test < 30, 1, 0)
            if task == 'postop_los':
                pred_y_test1 = np.where(pred_y_test > outcomes[task].describe()['75%'], 1, 0)
                true_y_test1 = np.where(true_y_test > outcomes[task].describe()['75%'], 1, 0)

            test_auroc = roc_auc_score(true_y_test1, pred_y_test1)
            test_auprc = average_precision_score(true_y_test1, pred_y_test1)
            print("Test AUROC and AUPRC (Regression converted ) values are ", np.round(test_auroc, 4), np.round(test_auprc, 4))

            if (onlyProcedureTxt):
                hm = 'onlyProcedureTxt'

            csvdata = {
                'hp': json.dumps(vars(args)),
                'n_estimator/max_depth/learningrate': str([n_est, m_d, lr, L1_w, L2_w]),
                # 'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
                'corr': corr_value,
                'corr_pvalue': cor_p_value,
                'R2': r2value,
                'MAE': mae_full,
                'MSE': mse_full,
                'Stratifying_points': stratifying_point_dict,
                'Stratified_MAE': mae_dict,
                'AUROC_simplified': test_auroc,
                'AUPRC_simplified': test_auprc,
                'model': model,
                'name': hm,
                'target': args.task,
                'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S")
            }

            csvdata = pd.DataFrame(csvdata)
            outputcsv = os.path.join('/output/', args.outputcsv)
            if (os.path.exists(outputcsv)):
                csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
            else:
                csvdata.to_csv(outputcsv, header=True, index=False)




