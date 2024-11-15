
# importing packages
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.stats import pearsonr
from xgboost import XGBClassifier, XGBRegressor
import sys, argparse
import json
from datetime import datetime
from pyarrow import feather  # directly writing import pyarrow didn't work

import Preops_processing as pps


from datetime import datetime
import json
import pickle

parser = argparse.ArgumentParser(description='Tabular modular XGBT model training')

## modalities to select
parser.add_argument('--preops', default=True, action='store_true',
                    help='Whether to add preops and bow to ts representation')
parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
parser.add_argument('--homemeds', action="store_true",
                    help='Whether to add homemeds to ts representation')

## for the homemeds
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum'}


## task and setup parameters
parser.add_argument("--modelType", default='XGBT')  # options {'TabNet', others later}
parser.add_argument("--task",  default="icu") #
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--bestModel",  default="False", help='True when the best HP tuned settings are used on the train+valid setup') #


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

# data_dir = '/mnt/ris/ActFastExports/v1.3.3/'
data_dir = '/input/'

# out_dir = './'
out_dir = '/output/'

preops0 = pd.read_csv(data_dir + 'mv_data/mv_preop.csv')
preops0 = preops0.drop_duplicates(subset=['orlogid_encoded'])
outcomes0 = pd.read_csv(data_dir + 'mv_data/outcomes_mv.csv')
outcomes0 = outcomes0.dropna(subset=['orlogid_encoded'])
end_of_case_times0 = feather.read_feather(data_dir + 'mv_data/end_of_case_times_wave0.feather')

end_of_case_times0 = end_of_case_times0[['orlogid_encoded', 'endtime']]

preops1 = pd.read_csv(data_dir + 'epic_preop.csv')
outcomes1 = pd.read_csv(data_dir + 'epic_outcomes.csv')
end_of_case_times1 = outcomes1[['orlogid_encoded', 'endtime']]

outcomes = pd.concat([outcomes0, outcomes1], axis=0)
end_of_case_times = pd.concat([end_of_case_times0, end_of_case_times1], axis=0)
preops = pd.concat([preops0, preops1], axis=0)


# end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration', 'n_glu_high',
                           'low_sbp_time','aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                           'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                           'opioids_count_day0', 'opioids_count_day1']
binary_outcome = args.task not in regression_outcome_list

if args.task=='icu':
    outcomes = outcomes.dropna(subset=['ICU'])
outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]
#end_of_case_times1 = end_of_case_times1.loc[end_of_case_times1['endtime'] > 30]

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
combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(set(end_of_case_times['orlogid_encoded'].values)).intersection(set(preops['orlogid_encoded'].values)))

if False:
    combined_case_set = np.random.choice(combined_case_set, 2500, replace=False)
    #combined_case_set1 = np.random.choice(combined_case_set1, 2500, replace=False)
    #combined_case_set = list(combined_case_set) + list(combined_case_set1)
    #combined_case_set = np.concatenate([combined_case_set, combined_case_set1])


outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
#preops1 = preops1.loc[preops1['orlogid_encoded'].isin(combined_case_set)]
end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)


# checking for NA and other filters
outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
outcome_df['orlogid_encoded'] = outcome_df['orlogid_encoded'].astype('str')
outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)   # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

end_of_case_times['orlogid_encoded'] = end_of_case_times['orlogid_encoded'].astype('str')
preops['orlogid_encoded'] = preops['orlogid_encoded'].astype('str')
endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],axis=1).rename({"new_person": "person_integer"},axis=1).sort_values(["person_integer"]).reset_index(drop=True)

#preop_comb = pd.concat([preops, preops1], axis=0)

preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"},axis=1).sort_values(["person_integer"]).reset_index(drop=True)


if 'preops' in modality_to_use:
    # this is being used because we will be adding the problem list and pmh as a seperate module in this file too
    # to drop the old pmh
    to_drop_old_pmh_with_others = ["MentalHistory_anxiety", "MentalHistory_bipolar",
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

    to_drop_old_pmh_with_others = list(set(preops).intersection(to_drop_old_pmh_with_others))
    preops = preops.drop(columns=to_drop_old_pmh_with_others) # "['pre_aki_status', 'preop_ICU', 'AnestStop']


    bow_input0 = pd.read_csv(data_dir + 'mv_data/cbow_proc_text_mv.csv')
    if "Unnamed: 0" in bow_input0.columns:  # because the csv file has index column
        bow_input0.drop(columns=['Unnamed: 0'], inplace=True)
    bow_input1 = pd.read_csv(data_dir + 'cbow_proc_text.csv')

    bow_input = pd.concat([bow_input0, bow_input1], axis=0)


    bow_input = bow_input.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
        ["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    bow_cols = [col for col in bow_input.columns if 'BOW' in col]
    bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
    bow_input.fillna(0, inplace=True)

if 'homemeds' in modality_to_use:
    # home meds reading and processing
    home_meds0 = pd.read_csv(data_dir + 'mv_data/home_med_cui_mv.csv', low_memory=False)
    home_meds1 = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)

    home_meds = pd.concat([home_meds0, home_meds1], axis=0)
    home_meds['orlogid_encoded'] = home_meds['orlogid_encoded'].astype('str')

    Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

    # home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
    # home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')
    home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded',
                                                  'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
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
    hm_embed_flag = 0  # not applicable case
    if args.home_medsform == 'embedding_attention':
        hm_embed_flag = 1
        col_names = [col for col in home_meds_embedded.columns if 'V' in col]
        home_meds_embedded.fillna(0, inplace=True)
        home_meds_embedded['med_pos'] = [item for idx in
                                         home_meds_embedded.groupby(by='orlogid_encoded')['rxcui'].count()
                                         for item in range(idx)]
        home_meds_embedded1 = new_index.merge(home_meds_embedded, on="orlogid_encoded", how="left").drop(
            ["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
        home_meds_embedded1.fillna(0, inplace=True)  # setting the value for the ones that were added later

    if args.home_medsform == 'embedding_attention':
        index_HM_med_ids = torch.tensor(home_meds_embedded1[['person_integer', 'med_pos']].values, dtype=int)
        value_HMmed_embed = torch.tensor(home_meds_embedded1[col_names].astype('float').values, dtype=float)
        dense_HM_embedding = torch.sparse_coo_tensor(torch.transpose(index_HM_med_ids, 0, 1), value_HMmed_embed,
                                                     dtype=torch.float32)
        hm_input_dim = len(col_names)
    else:
        hm_input_dim = len(home_meds_final.columns)

    breakpoint()

if 'pmh' in modality_to_use:
    pmh_emb_sb0 = pd.read_csv(data_dir + 'mv_data/pmh_sherbert_mv.csv')
    pmh_emb_sb1 = pd.read_csv(data_dir + 'pmh_sherbert.csv')

    pmh_emb_sb = pd.concat([pmh_emb_sb0, pmh_emb_sb1], axis=0)
    pmh_emb_sb['orlogid_encoded'] = pmh_emb_sb['orlogid_encoded'].astype('str')
    if args.pmhform == 'embedding_sum':
        pmh_emb_sb.drop(columns=['ICD_10_CODES'], inplace=True)  # although the next groupby sum is capable of removing this column, explicit removal is better
        pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)

        pmh_input_dim = len(pmh_emb_sb_final.columns)

    if args.pmhform == 'embedding_attention':
        col_names = [col for col in pmh_emb_sb.columns if 'sherbet' in col]
        pmh_emb_sb.fillna(0, inplace=True)
        pmh_emb_sb['pmh_pos'] = [item for idx in
                                         pmh_emb_sb.groupby(by='orlogid_encoded')['ICD_10_CODES'].count()
                                         for item in range(idx)]
        pmh_emb_sb1 = new_index.merge(pmh_emb_sb, on="orlogid_encoded", how="left").drop(
            ["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
        pmh_emb_sb1.fillna(0, inplace=True)  # setting the value for the ones that were added later

        index_pmh_ids = torch.tensor(pmh_emb_sb1[['person_integer', 'pmh_pos']].values, dtype=int)
        value_pmh_embed = torch.tensor(pmh_emb_sb1[col_names].astype('float').values, dtype=float)
        dense_pmh_embedding = torch.sparse_coo_tensor(torch.transpose(index_pmh_ids, 0, 1), value_pmh_embed,
                                                     dtype=torch.float32)
        pmh_input_dim = len(col_names)

best_5_random_number = []  # this will take the args when directly run otherwise it will read the number from the file namee
if eval(args.bestModel) == True:
    # path_to_dir = '/home/trips/PeriOperative_RiskPrediction/HP_output/'
    # sav_dir = '/home/trips/PeriOperative_RiskPrediction/Best_results/Preoperative/'
    path_to_dir = out_dir + 'HP_output/'
    sav_dir = out_dir + 'Best_results/Preoperative/'
    # path_to_dir = '../HP_output/'
    # sav_dir = '../Best_results/Preoperative/'
    # best_file_name= path_to_dir + 'Best_trial_resulticu_TabNet_modal__preops_cbow_pmh_problist_homemeds174_24-07-17-10:55:55.json'
    file_names = os.listdir(path_to_dir)
    import re

    best_5_names = []

    best_5_initial_name = 'Best_trial_result' + args.task + "_" + str(args.modelType) + "_modal_"
    pattern_num = r'\d+'
    pattern_text = r'[a-zA-Z]+'
    modal_name = 'DataModal'
    for i in range(len(modality_to_use)):
        best_5_initial_name = best_5_initial_name + "_" + modality_to_use[i]
        modal_name = modal_name + "_" + modality_to_use[i]

    dir_name = sav_dir + args.modelType + '/' + modal_name + "_" + str(args.task) + "/"

    for file_name in file_names:
        if best_5_initial_name in file_name:
            print(file_name)
            string = file_name.split("_")[-2]
            match_text = re.search(pattern_text, string)
            if match_text:
                text = match_text.group()
                if text in modality_to_use:
                    best_5_names.append(file_name)
                    match_num = re.search(pattern_num, string)
                    if match_num:
                        number = match_num.group()
                        best_5_random_number.append(number)
                    else:
                        print("There should be a number here, so gotta find it")
                        breakpoint()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print(f"The directory '{dir_name}' already exists.")

    best_metadata_dict = {}
else:
    best_5_random_number.append(args.randomSeed)
    hm_reading_form = args.home_medsform

outcome_with_pred_test = outcome_df.iloc[test_index]
outcome_with_pred_test = outcome_with_pred_test.rename(columns={'outcome': 'y_true'})

if binary_outcome:
    perf_metric = np.zeros((len(best_5_random_number), 2))  # 2 is for the metrics auroc and auprc
else:
    perf_metric = np.zeros((len(best_5_random_number), 5))  # 5 is for the metrics corr, corr_p, R2, MAE, MSE


