"""
This file reads an external dataset and validates the existing intraoperative models on the external dataset.
Currently, the external dataset is one from a different time period which may not have a complete overlap with the training data in terms of the features.
The tabular models here are: XGBTtsSum, lstm, MVCL

"""

# importing packages
import numpy as np
import pandas as pd
import os
import torch
from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.stats import pearsonr
from xgboost import XGBClassifier, XGBRegressor
import sys, argparse
import json
import pickle
import ast
from datetime import datetime
from End_to_end_supervised import preprocess_inference
from End_to_end_supervised import preop_flow_med_bow_model

parser = argparse.ArgumentParser(description='TS modular model validation in wave2')

## modalities to select
parser.add_argument('--preops', action='store_true',
                    help='Whether to add preops and bow to ts representation')
parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problem list ')
parser.add_argument('--homemeds', action="store_true",
                    help='Whether to add homemeds to ts representation')
parser.add_argument('--meds', action="store_true",
                    help='Whether to add meds to ts representation')
parser.add_argument('--flow', action="store_true",
                    help='Whether to add flowsheets to ts representation')


## for the homemeds and meds summary
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum'}
parser.add_argument('--TSMedAggType', default='MedUnitComboSumOverTime') # options {'Embedding', 'MedUnitComboSumOverTime'}


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

if eval('args.flow') == True:
    modality_to_use.append('flow')

if eval('args.meds') == True:
    modality_to_use.append('meds')

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

end_of_case_times = outcomes_wave2[['orlogid_encoded', 'endtime']]
# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
    sub_id = preops_wave2[preops_wave2['plannedDispo']!='ICU'].index
    preops_wave2 = preops_wave2.loc[sub_id]

if args.task == 'post_dialysis':
    sub_id = preops_wave2[(preops_wave2['ESRD'] != 1) & (preops_wave2['Dialysis'] != 1)].index   # dropping every row in which at least one value is 1
    preops_wave2 = preops_wave2[(preops_wave2['ESRD'] != 1) & (preops_wave2['Dialysis'] != 1)]

## intersect 3 mandatory data sources: preop, outcome
combined_case_set = list(set(outcome_df_wave2["orlogid_encoded"].values).intersection(
    set(end_of_case_times['orlogid_encoded'].values)).intersection(set(preops_wave2['orlogid_encoded'].values)))

if True:
    combined_case_set = np.random.choice(combined_case_set, 1000, replace=False)

end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

outcome_df_wave2 = outcome_df_wave2.loc[outcome_df_wave2['orlogid_encoded'].isin(combined_case_set)]
outcome_df_wave2.set_axis(["orlogid_encoded", "outcome"], axis=1, inplace=True)

new_index = outcome_df_wave2["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)  # this will serve as a good baseline to select cases with outcomes

endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

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

if 'flow' in modality_to_use:
    # flowsheet data
    very_dense_flow = feather.read_feather(data_dir +"flow_ts/Imputed_very_dense_flow_wave2.feather")
    very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace=True)
    very_dense_flow = very_dense_flow.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
    very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint']]
    very_dense_flow.drop(["endtime"], axis=1, inplace=True)

    other_intra_flow_wlabs = feather.read_feather(data_dir +"flow_ts/Imputed_other_flow_wave2.feather")
    other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index, inplace=True)
    other_intra_flow_wlabs = other_intra_flow_wlabs.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                          on="orlogid_encoded")
    other_intra_flow_wlabs = other_intra_flow_wlabs.loc[
        other_intra_flow_wlabs['endtime'] > other_intra_flow_wlabs['timepoint']]
    other_intra_flow_wlabs.drop(["endtime"], axis=1, inplace=True)

if 'meds' in modality_to_use:
    # reading the med files
    all_med_data = feather.read_feather(data_dir + 'med_ts/intraop_meds_filterd_wave2.feather')
    all_med_data.drop(all_med_data[all_med_data['time'] > 511].index, inplace=True)
    all_med_data = all_med_data.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
    all_med_data = all_med_data.loc[all_med_data['endtime'] > all_med_data['time']]
    all_med_data.drop(["endtime"], axis=1, inplace=True)

# model_list = ['XGBTtsSum', 'lstm', 'MVCL']
model_list = ['lstm']

sav_dir = out_dir + 'Best_results/Intraoperative/'
file_names = os.listdir(sav_dir)

output_file_name = sav_dir + 'preops_metadata_' + str(args.task) + '.json'

md_f = open(output_file_name)
metadata = json.load(md_f)

processed_preops_wave2 = preprocess_inference(preops_wave2.copy(), metadata)

# Reading the models now
for m_name in model_list:
    modal_name = 'DataModal'
    for i in range(len(modality_to_use)):
        modal_name = modal_name + "_" + modality_to_use[i]
    dir_name = sav_dir + m_name + '/' + modal_name + "_" + str(args.task) +"/"

    if (m_name=='lstm') or (m_name=='MVCL'):
        if 'flow' in modality_to_use:
            very_dense_flow = very_dense_flow.copy().merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

            other_intra_flow_wlabs = other_intra_flow_wlabs.copy().merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

            breakpoint()
            """ TS flowsheet proprocessing """
            # need to convert the type of orlogid_encoded from object to int
            other_intra_flow_wlabs['new_person'] = other_intra_flow_wlabs['new_person'].astype('int')
            very_dense_flow['new_person'] = very_dense_flow['new_person'].astype('int')

            index_med_other_flow = torch.tensor(
                other_intra_flow_wlabs[['new_person', 'timepoint', 'measure_index']].values, dtype=int)
            value_med_other_flow = torch.tensor(other_intra_flow_wlabs['VALUE'].values)
            flowsheet_other_flow = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                                           value_med_other_flow, dtype=torch.float32)

            index_med_very_dense = torch.tensor(
                very_dense_flow[['new_person', 'timepoint', 'measure_index']].values, dtype=int)
            value_med_very_dense = torch.tensor(very_dense_flow['VALUE'].values)
            flowsheet_very_dense_sparse_form = torch.sparse_coo_tensor(torch.transpose(index_med_very_dense, 0, 1),
                                                                       value_med_very_dense,
                                                                       dtype=torch.float32)  ## this is memory heavy and could be skipped, only because it is making a copy not really because it is harder to store
            flowsheet_very_dense = flowsheet_very_dense_sparse_form.to_dense()
            flowsheet_very_dense = torch.cumsum(flowsheet_very_dense, dim=1)

        if 'meds' in modality_to_use:
            ## Special med * unit comb encoding
            all_med_data['med_unit_comb'] = list(zip(all_med_data['med_integer'], all_med_data['unit_integer']))
            med_unit_coded, med_unit_unique_codes = pd.factorize(all_med_data['med_unit_comb'])
            all_med_data['med_unit_comb'] = med_unit_coded

            a = pd.DataFrame(columns=['med_integer', 'unit_integer', 'med_unit_combo'])
            a['med_integer'] = [med_unit_unique_codes[i][0] for i in range(len(med_unit_unique_codes))]
            a['unit_integer'] = [med_unit_unique_codes[i][1] for i in range(len(med_unit_unique_codes))]
            a['med_unit_combo'] = np.arange(len(med_unit_unique_codes))
            a.sort_values(by=['med_integer', 'med_unit_combo'], inplace=True)

            group_start = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']),
                                                                        1)).nonzero().squeeze() + 1  # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.
            group_end = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']),
                                                                      -1)).nonzero().squeeze() + 1  # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.

            group_start = torch.cat((torch.tensor(0).reshape((1)),
                                     group_start))  # prepending 0 to make sure that it is treated as an empty slot
            group_end = torch.cat((torch.tensor(0).reshape((1)),
                                   group_end))  # prepending 0 to make sure that it is treated as an empty slot

            drug_med_ids = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_integer']]

            drug_med_id_map = feather.read_feather(data_dir + 'med_ts/med_id_map.feather')
            drug_words = None
            word_id_map = None

            # drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'unit_integer',
            #                           'dose']]
            drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_unit_comb',
                                      'dose']]  # replacing the unit_integer column by med_unit_comb column

            unit_id_map = feather.read_feather(data_dir + 'med_ts/unit_id_map.feather')
            # vocab_len_units = len(unit_id_map)
            # vocab_len_units = len(med_unit_unique_codes)  # replacing  len(unit_id_map) by len(med_unit_unique_codes)

            drug_dose = drug_dose.merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)


            if drug_words is not None:
                drug_words = drug_words.merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

            if drug_med_ids is not None:
                drug_med_ids = drug_med_ids.merge(new_index, on="orlogid_encoded", how="inner").reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

            breakpoint()
            ## I suppose these could have sorted differently
            ## TODO apparently, torch.from_numpy shares the memory buffer and inherits type
            index_med_ids = torch.tensor(drug_med_ids[['new_person', 'time', 'drug_position']].values, dtype=int)
            index_med_dose = torch.tensor(drug_dose[['new_person', 'time', 'drug_position']].values, dtype=int)
            value_med_dose = torch.tensor(drug_dose['dose'].astype('float').values, dtype=float)
            value_med_unit = torch.tensor(drug_dose['med_unit_comb'].values, dtype=int)

            add_unit = 0 in value_med_unit.unique()
            dense_med_units = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_unit + add_unit,
                                                      dtype=torch.int32)
            dense_med_dose = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_dose,
                                                     dtype=torch.float32)

            value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
            add_med = 0 in value_med_ids.unique()
            dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0, 1), value_med_ids + add_med,
                                                    dtype=torch.int32)
    breakpoint()
    if m_name=='MVCL':
        print("TO be filled")

    metadata_best_run_file = dir_name + '/Best_runs_metadata.pickle'
    with open(metadata_best_run_file, 'rb') as file:
        existing_data = pickle.load(file)

    best_5_random_number = [int(num.split("_")[-1]) for num in list(existing_data.keys())]

    if binary_outcome:
        perf_metric = np.zeros((len(best_5_random_number), 2))  # 2 is for the metrics auroc and auprc
    else:
        perf_metric = np.zeros((len(best_5_random_number), 5))  # 5 is for the metrics corr, corr_p, R2, MAE, MSE

    if m_name=='XGBTtsSum':
        if 'flow' in modality_to_use:
            very_dense_flow_stat = very_dense_flow.groupby(by=['orlogid_encoded', 'measure_index'])['VALUE'].agg(
                ['mean', 'std']).reset_index()
            very_dense_flow_stat = pd.concat([very_dense_flow_stat, pd.concat([very_dense_flow.groupby(
                by=['orlogid_encoded', 'measure_index'])['VALUE'].quantile(0.2).reset_index().rename(
                columns={'VALUE': 'VALUE_20perc'}), very_dense_flow.groupby(
                by=['orlogid_encoded', 'measure_index'])['VALUE'].quantile(0.8).reset_index().drop(
                columns=['orlogid_encoded', 'measure_index']).rename(columns={'VALUE': 'VALUE_80perc'})],
                axis=1).drop(
                columns=['orlogid_encoded', 'measure_index'])], axis=1)
            very_dense_flow_stat['std'].fillna(0, inplace=True)
            very_dense_flow_stat.measure_index = very_dense_flow_stat['measure_index'].astype('str')
            very_dense_flow_stat = very_dense_flow_stat.pivot(index='orlogid_encoded', columns='measure_index',
                                                              values=['mean', 'std', 'VALUE_20perc',
                                                                      'VALUE_80perc']).reset_index()
            very_dense_flow_stat.columns = ['_'.join(col) for col in very_dense_flow_stat.columns]
            temp_name = ['orlogid_encoded'] + [col + "FlowD" for col in very_dense_flow_stat.columns if
                                               col not in ['orlogid_encoded_']]
            very_dense_flow_stat.rename(columns=dict(zip(very_dense_flow_stat.columns, temp_name)), inplace=True)

            other_intra_flow_wlabs_stat = other_intra_flow_wlabs.groupby(by=['orlogid_encoded', 'measure_index'])[
                'VALUE'].agg(['mean', 'std']).reset_index()
            other_intra_flow_wlabs_stat = pd.concat([other_intra_flow_wlabs_stat, pd.concat([
                other_intra_flow_wlabs.groupby(
                    by=[
                        'orlogid_encoded',
                        'measure_index'])[
                    'VALUE'].quantile(
                    0.2).reset_index().rename(
                    columns={
                        'VALUE': 'VALUE_20perc'}),
                other_intra_flow_wlabs.groupby(
                    by=[
                        'orlogid_encoded',
                        'measure_index'])[
                    'VALUE'].quantile(
                    0.8).reset_index().drop(
                    columns=[
                        'orlogid_encoded',
                        'measure_index']).rename(
                    columns={
                        'VALUE': 'VALUE_80perc'})],
                axis=1).drop(
                columns=['orlogid_encoded', 'measure_index'])], axis=1)
            other_intra_flow_wlabs_stat['std'].fillna(0, inplace=True)
            other_intra_flow_wlabs_stat.measure_index = other_intra_flow_wlabs_stat['measure_index'].astype('str')
            other_intra_flow_wlabs_stat = other_intra_flow_wlabs_stat.pivot(index='orlogid_encoded',
                                                                            columns='measure_index',
                                                                            values=['mean', 'std', 'VALUE_20perc',
                                                                                    'VALUE_80perc']).reset_index()
            other_intra_flow_wlabs_stat.columns = ['_'.join(col) for col in other_intra_flow_wlabs_stat.columns]
            temp_name = ['orlogid_encoded'] + [col + "FlowS" for col in other_intra_flow_wlabs_stat.columns if
                                               col not in ['orlogid_encoded_']]
            other_intra_flow_wlabs_stat.rename(columns=dict(zip(other_intra_flow_wlabs_stat.columns, temp_name)),
                                               inplace=True)

            very_dense_flow_stat1 = very_dense_flow_stat.copy().merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

            other_intra_flow_wlabs_stat1 = other_intra_flow_wlabs_stat.copy().merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

        if 'meds' in modality_to_use:
            ## Special med * unit comb encoding
            all_med_data['med_unit_comb'] = list(zip(all_med_data['med_integer'], all_med_data['unit_integer']))
            med_unit_coded, med_unit_unique_codes = pd.factorize(all_med_data['med_unit_comb'])
            all_med_data['med_unit_comb'] = med_unit_coded
            all_med_data['dose'] = all_med_data['dose'].astype('float')
            all_med_data_stat = all_med_data.groupby(by=['orlogid_encoded', 'med_unit_comb'])['dose'].agg(
                ['sum']).reset_index()  # the sum is over the time for each unique med unit combo
            all_med_data_stat.med_unit_comb = all_med_data_stat['med_unit_comb'].astype('str')
            all_med_data_stat = all_med_data_stat.pivot(index='orlogid_encoded', columns='med_unit_comb',
                                                        values=['sum']).reset_index()
            all_med_data_stat.columns = ['_'.join(col) for col in all_med_data_stat.columns]
            all_med_data_stat.fillna(0, inplace=True)
            temp_name = ['orlogid_encoded'] + [col + "MedUnit" for col in all_med_data_stat.columns if
                                               col not in ['orlogid_encoded_']]
            all_med_data_stat.rename(columns=dict(zip(all_med_data_stat.columns, temp_name)), inplace=True)
            med_combo_redundant = list(all_med_data_stat.sum(axis=0).sort_values()[all_med_data_stat.sum(
                axis=0).sort_values() == 0].index)  # dropping the med unit combo columns that do not have any recorded dosage
            all_med_data_stat = all_med_data_stat.drop(columns=med_combo_redundant)
            low_freq_rec = [i for i in all_med_data_stat.columns if np.count_nonzero(
                all_med_data_stat[i].to_numpy()) < 10]  # med unit combo recorded in only a handful of patients
            all_med_data_stat = all_med_data_stat.drop(columns=low_freq_rec)

            all_med_data_stat1 = all_med_data_stat.copy().merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(new_index.set_index('new_person').index,fill_value=0).reset_index(drop=True).drop(columns=['orlogid_encoded'], axis=1)

    for runNum in range(len(best_5_random_number)):

        test_set = []
        if m_name=='lstm':
            data_te = {}
            data_te['outcomes'] = torch.tensor(outcome_df_wave2["outcome"].values)
            data_te['endtimes'] = torch.tensor(endtimes["endtime"].values, dtype=int)
        if 'preops' in modality_to_use:
            test_set.append(processed_preops_wave2)
            test_set.append(cbow_wave2)
            if m_name=='lstm':
                data_te['preops'] = torch.tensor(processed_preops_wave2.to_numpy(), dtype=torch.float32)
                data_te['cbow'] = torch.tensor(cbow_wave2.to_numpy(), dtype=torch.float32)

        if 'homemeds' in modality_to_use:
            hm_reading_form = existing_data['run_randomSeed_'+str(int(best_5_random_number[runNum]))]['hm_form']
            if hm_reading_form == 'ohe':
                home_meds_final = home_meds_ohe
            if hm_reading_form == 'embedding_sum':
                home_meds_final = home_meds_sum.copy().drop(["rxcui"], axis=1)
            if m_name=='lstm':
                hm_reading_form='embedding_sum'  # because this choice was used for hp tuning lstms
                data_te['homemeds'] = torch.tensor(home_meds_final.to_numpy(), dtype=torch.float32)

            test_set.append(home_meds_final)

        if 'pmh' in modality_to_use:
            new_name_pmh = ['pmh_sherbet'+str(num) for num in range(len(pmh_emb_sb.columns))]
            dict_name = dict(zip(pmh_emb_sb.columns,new_name_pmh ))
            if m_name=='lstm':
                data_te['pmh']=torch.tensor(pmh_emb_sb.copy().rename(columns=dict_name).to_numpy(), dtype=torch.float32)
            test_set.append(pmh_emb_sb.copy().rename(columns=dict_name))

        if 'problist' in modality_to_use:
            new_name_prbl = ['prbl_sherbet'+str(num) for num in range(len(prob_list_emb_sb.columns))]
            dict_name = dict(zip(prob_list_emb_sb.columns,new_name_prbl ))
            if m_name=='lstm':
                data_te['problist']=torch.tensor(prob_list_emb_sb.copy().rename(columns=dict_name).to_numpy(), dtype=torch.float32)
            test_set.append(prob_list_emb_sb.copy().rename(columns=dict_name))

        if 'flow' in modality_to_use:
            if m_name=='XGBTtsSum':
                test_set.append(very_dense_flow_stat1)
                test_set.append(other_intra_flow_wlabs_stat1)
            if m_name=='lstm':
                data_te['flow'] = [flowsheet_very_dense,flowsheet_other_flow.coalesce()]

        if 'meds' in modality_to_use:
            if m_name=='XGBTtsSum':
                test_set.append(all_med_data_stat1)
            if m_name == 'lstm':
                data_te['meds'] = [dense_med_ids.coalesce(), dense_med_dose, dense_med_units]

        y_test = outcome_df_wave2["outcome"].values

        if m_name=='XGBTtsSum':
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
            saving_path_name = dir_name + 'XGBTtsSum_BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".json"
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

        if m_name=='lstm':
            device = torch.device('cuda')
            config = existing_data['run_randomSeed_' + str(int(best_5_random_number[runNum]))]['model_params']
            breakpoint()
            model = preop_flow_med_bow_model.TS_lstm_Med_index(**config).to(device)

            saving_path_name = existing_data['run_randomSeed_' + str(int(best_5_random_number[runNum]))]['model_file_path']
            # saving_path_name = dir_name + 'BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + ".pkl"
            state_dict = torch.load(saving_path_name, map_location=device)
            model.load_state_dict(state_dict)

            model.eval()
            true_y_test = []
            pred_y_test = []
            breakpoint()
            batchsize = 32
            # nbatch = data_te['outcomes'].shape[0] // batchsize
            nbatch, remain_batch = divmod(data_te['outcomes'].shape[0], batchsize)
            if remain_batch > 0:
                nbatch = nbatch + 1  # this is being done to make sure all the test data is being used when the test set size is not a multiple of batchsize
            for i in range(nbatch):

                if (remain_batch > 0) and (i == nbatch - 1):
                    these_index = torch.tensor(list(range(i * batchsize, (i * batchsize) + remain_batch)), dtype=int)
                else:
                    these_index = torch.tensor(list(range(i * batchsize, (i + 1) * batchsize)), dtype=int)
                local_data = {}
                for k in data_te.keys():
                    if type(data_te[k]) != list:
                        local_data[k] = torch.index_select(data_te[k], 0, these_index)
                    else:
                        local_data[k] = [torch.index_select(x, 0, these_index) for x in data_te[k]]

                if args.task == 'endofcase':
                    local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                        len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
                    local_data[0][:, -1] = local_data[
                        1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
                    local_data[-1] = torch.from_numpy(
                        np.repeat([1, 0], [int(batchsize / 2), batchsize - int(batchsize / 2)]))
                if (args.includeMissingnessMasks):  # appending the missingness masks in test data
                    if 'preops' in modality_to_use:
                        local_data['preops'] = [local_data['preops'],
                                                torch.tensor(preops_te_mask.iloc[these_index].to_numpy(),
                                                             dtype=torch.float32)]
                    if 'flow' in modality_to_use:
                        local_data['flow'].append(very_dense_te_mask[these_index, :, :])
                        sparse_mask = torch.sparse_coo_tensor(local_data['flow'][1]._indices(),
                                                              np.ones(len(local_data['flow'][1]._values())),
                                                              local_data['flow'][1].size())
                        local_data['flow'].append(sparse_mask)

                data_valid, mod_order_dict = preop_flow_med_bow_model.collate_time_series(local_data, device)

                y_pred, reg_loss = model(data_valid[0])
                # values from the last epoch; it will get overwritten
                # using test data only instead of validation data for evaluation currently because the validation will be done on a seperate data
                true_y_test.append(data_valid[1].float().detach().numpy())
                pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())

            y_test = np.concatenate(true_y_test)
            pred_y_test = np.concatenate(pred_y_test)

        if binary_outcome:
            test_auroc = roc_auc_score(y_test, pred_y_test)
            test_auprc = average_precision_score(y_test, pred_y_test)

            perf_metric[runNum, 0] = test_auroc
            perf_metric[runNum, 1] = test_auprc
        else:
            corr_value = np.round(pearsonr(np.array(y_test.reshape(-1, 1)), np.array(pred_y_test))[0], 3)
            cor_p_value = np.round(pearsonr(np.array(y_test.reshape(-1, 1)), np.array(pred_y_test))[1], 3)
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

            perf_metric[runNum, 0] = corr_value[0]
            perf_metric[runNum, 1] = cor_p_value[0]
            perf_metric[runNum, 2] = r2value
            perf_metric[runNum, 3] = mae_full
            perf_metric[runNum, 4] = mse_full

        print(perf_metric)

    print("Final performance metric", perf_metric)
    breakpoint()
    print(" Model type :", m_name, " Modal name: ", modal_name, "  Finished for wave 2" )