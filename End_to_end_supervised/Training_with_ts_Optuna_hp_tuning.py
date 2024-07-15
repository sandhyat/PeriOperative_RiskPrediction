


import optuna
from optuna.trial import TrialState

import json
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys, argparse
import hashlib
import numpy as np
import pandas as pd
import math

from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import Preops_processing as pps
import ts_model_class_Optuna_hp_tune


def objective(trial, args):
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

    print("MODALITY TO USE ", modality_to_use)

    config = dict(
        linear_out=1,
        p_final=trial.suggest_float("p_final", 0.01, 0.5, log=False),
        finalBN=trial.suggest_categorical("finalBN", [True, False]),
        hidden_units_final=trial.suggest_int('hidden_units_final', 5,16),
        # this is being added apriori because we are projecting the final representation to this dimension
        weightInt=trial.suggest_categorical("weightInt", [True, False]),
    )

    # reproducibility settings
    # random_seed = 1 # or any of your favorite number
    torch.manual_seed(args.randomSeed)
    torch.cuda.manual_seed(args.randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.randomSeed)

    # data_dir = '/mnt/ris/ActFastExports/v1.3.2/'
    data_dir = '/input/'
    out_dir = '/output/'

    # reading the preop and outcome feather files
    # preops = feather.read_feather(data_dir + 'preops_reduced_for_training.feather')
    # preops = preops.drop(['MRN_encoded'], axis =1)

    preops = pd.read_csv(data_dir + 'epic_preop.csv')
    outcomes = pd.read_csv(data_dir + 'epic_outcomes.csv')
    end_of_case_times = outcomes[['orlogid_encoded', 'endtime']]

    # end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
    regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood',
                               'postop_Vent_duration', 'n_glu_high',
                               'low_sbp_time', 'aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                               'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1']
    binary_outcome = args.task not in regression_outcome_list
    config['binary'] = binary_outcome

    outcomes = outcomes.dropna(subset=['ICU'])
    outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

    # exclude very short cases (this also excludes some invalid negative times)
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

    if args.task == 'endofcase':
        # updating the end_of_case_times targets for bigger distribution;
        """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
        end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 60]  ## cases that are too short
        end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] < 25 + 511]  ## cases that are too long
        end_of_case_times['true_test'] = end_of_case_times['endtime'] - 10
        end_of_case_times['t1'] = end_of_case_times['true_test'] - 30
        end_of_case_times['t2'] = end_of_case_times[
                                      'true_test'] - 35  # temporary just to make sure nothing breaks; not being used
        end_of_case_times['t3'] = end_of_case_times[
                                      'true_test'] - 40  # temporary just to make sure nothing breaks; not being used
        overSampling = False  # TODO: there could be a better way to handle this.
    else:
        end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'], 511)
        # end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 90)

    binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac',
                           'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']

    if args.task in regression_outcome_list:
        outcomes['survival_time'] = np.minimum(outcomes['survival_time'], 90)
        outcomes['readmission_survival'] = np.minimum(outcomes['readmission_survival'], 30)
        # outcomes['n_glucose_high'] = outcomes['n_glucose_high'].fillna(0)  # this might not be needed as already taken of by the where statement
        outcomes['n_glu_high'] = np.where(outcomes['N_glu_measured'] > 0,
                                          outcomes['n_glu_high'] / outcomes['N_glu_measured'], 0)
        outcomes['total_blood'] = outcomes['total_blood'].fillna(0)
        outcomes['low_sbp_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_sbp_time'] / outcomes['total_t'], 0)
        outcomes['low_relmap_time'] = np.where(outcomes['total_t'] > 0,
                                               outcomes['low_relmap_time'] / outcomes['total_t'], 0)
        outcomes['low_map_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_time'] / outcomes['total_t'], 0)
        outcomes['aoc_low_sbp'] = np.where(outcomes['total_t'] > 0, outcomes['aoc_low_sbp'], 0)
        outcomes['low_relmap_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_aoc'], 0)
        outcomes['low_map_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_aoc'], 0)
        outcomes['postop_vent_duration'] = outcomes['postop_vent_duration'].fillna(0)
        outcomes['timew_pain_avg_0'] = outcomes['timew_pain_avg_0'] / (
                    outcomes['timew_pain_avg_0'].max() - outcomes['timew_pain_avg_0'].min())
        outcomes['median_pain_0'] = outcomes['median_pain_0'] / (
                    outcomes['median_pain_0'].max() - outcomes['median_pain_0'].min())
        outcomes['worst_pain_0'] = outcomes['worst_pain_0'] / (
                    outcomes['worst_pain_0'].max() - outcomes['worst_pain_0'].min())
        outcomes['worst_pain_1'] = outcomes['worst_pain_1'] / (
                    outcomes['worst_pain_1'].max() - outcomes['worst_pain_1'].min())

        outcome_df = outcomes[['orlogid_encoded', args.task]]
    elif args.task in binary_outcome_list:
        if args.task == 'VTE':
            temp_outcome = outcomes[['orlogid_encoded']]
            temp_outcome[args.task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
            temp_outcome.loc[temp_outcome[args.task] == 2, args.task] = 1
        elif args.task == 'n_glucose_low':
            temp_outcome = outcomes[['orlogid_encoded', args.task]]
            temp_outcome[args.task] = temp_outcome[args.task].fillna(0)
            temp_outcome[args.task] = np.where(temp_outcome[args.task] > 0, 1, 0)
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

    if True:
        combined_case_set = np.random.choice(combined_case_set, 2500, replace=False)

    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
    preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
    end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

    outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

    # checking for NA and other filters
    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
    outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
    new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"},
                                                                          axis=1)  # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

    endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                          axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

    preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

    if 'preops' not in modality_to_use:
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
            assert outcome_df.iloc[valid.index]["outcome"].values.mean()>0, " THE VALIDATION SET MUST HAVE ATLEAST ONE POSITIVE EXAMPLE "

        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=args.randomSeed)

        train_index = train.index
        valid_index = valid.index
        test_index = test.index

        if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 'ICU']['plannedDispo'].index
            valid_index = preops.iloc[valid_index][preops.iloc[valid_index]['plannedDispo'] != 'ICU']['plannedDispo'].index


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
        bow_input = pd.read_csv(data_dir + 'cbow_proc_text.csv')

        bow_input = bow_input.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
            list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
            ["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)
        bow_cols = [col for col in bow_input.columns if 'BOW' in col]
        bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
        bow_input.fillna(0, inplace=True)

        y_outcome = outcome_df["outcome"].values
        # currently sacrificing 5 data points in the valid set and using the test set to finally compute the auroc etc
        preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = pps.preprocess_train(
            preops,args.task, y_outcome=y_outcome, binary_outcome=binary_outcome, valid_size=0.05, random_state=args.randomSeed, input_dr=data_dir, output_dr=out_dir)  # change back to 0.00005

        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            assert outcome_df.iloc[valid_index][
                   "outcome"].values.mean() > 0, " THE VALIDATION SET MUST HAVE ATLEAST ONE POSITIVE EXAMPLE "

        if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set (value of plannedDispo are numeric after processing; the df has also been changed )
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 3]['plannedDispo'].index
            preops_te = preops_te.iloc[test_index]
            valid_index = preops.iloc[valid_index][preops.iloc[valid_index]['plannedDispo'] != 3]['plannedDispo'].index
            preops_val = preops_val.loc[valid_index]  # pay attention to indexing here

        preop_mask_counter = 0
        num_preop_features = len(preops_tr.columns)
        if (args.includeMissingnessMasks):
            """  Masks for preops based on train test"""
            if args.task == 'endofcase':
                preops_tr_mask = pd.concat([preops_mask.iloc[train_index], preops_mask.iloc[train_index]])
                preops_te_mask = pd.concat([preops_mask.iloc[test_index], preops_mask.iloc[test_index]])
            else:
                preops_tr_mask = preops_mask.iloc[train_index]
                preops_te_mask = preops_mask.iloc[test_index]
            preop_mask_counter = 1

        config['hidden_units'] = trial.suggest_int("hidden_units",10,20)
        config['hidden_depth'] = trial.suggest_int("hidden_depth",2,6)
        config['weight_decay_preopsL2'] = trial.suggest_float("weight_decay_preopsL2", 1e-5, 1e-2, log=True)
        config['weight_decay_preopsL1'] = trial.suggest_float("weight_decay_preopsL1", 1e-5, 1e-2, log=True)
        config['input_shape'] = num_preop_features + preop_mask_counter * num_preop_features,  # this is done so that I dont have to write a seperate condition for endofcase where the current time is being appended to preops

        config['hidden_units_bow'] = trial.suggest_int("hidden_units_bow",50,100)
        config['hidden_units_final_bow'] = trial.suggest_int("hidden_units_final_bow",5,16)
        config['hidden_depth_bow'] = trial.suggest_int("hidden_depth_bow",3,5)
        config['weight_decay_bowL2'] = trial.suggest_float("weight_decay_bowL2", 1e-5, 1e-2, log=True)
        config['weight_decay_bowL1'] = trial.suggest_float("weight_decay_bowL1", 1e-5, 1e-2, log=True)
        config['input_shape_bow'] = len(bow_input.columns)

    if 'homemeds' in modality_to_use:
        # home meds reading and processing
        home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
        Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

        # home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
        # home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')
        home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded',
                                                      'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
        home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left',
                                                                           on='rxcui')
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

        home_meds_ohe = home_meds_ohe.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(
            list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
            ["orlogid_encoded"],
            axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)
        home_meds_ohe.fillna(0, inplace=True)  # setting the value for the ones that were added later

        home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
        home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(
            list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
            ["orlogid_encoded"],
            axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)
        home_meds_sum.fillna(0, inplace=True)  # setting the value for the ones that were added later

        if args.home_medsform == 'ohe':
            home_meds_final = home_meds_ohe
        if args.home_medsform == 'embedding_sum':
            # TODO: remove the rxcui number from the home_meds_sum dataframe
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
            hm_tr = torch.index_select(dense_HM_embedding, 0, torch.tensor(train_index)).coalesce()
            hm_val = torch.index_select(dense_HM_embedding, 0, torch.tensor(valid_index)).coalesce()
            hm_te = torch.index_select(dense_HM_embedding, 0, torch.tensor(test_index)).coalesce()
            hm_input_dim = len(col_names)
        else:
            hm_tr = torch.tensor(home_meds_final.iloc[train_index].to_numpy(), dtype=torch.float32)
            hm_te = torch.tensor(home_meds_final.iloc[test_index].to_numpy(), dtype=torch.float32)
            hm_val = torch.tensor(home_meds_final.iloc[valid_index].to_numpy(), dtype=torch.float32)
            hm_input_dim = len(home_meds_final.columns)

        config['hidden_units_hm'] = trial.suggest_int('hidden_units_hm', 5,16)
        config['hidden_units_final_hm'] = trial.suggest_int('hidden_units_final_hm', 5,16)
        config['hidden_depth_hm'] = trial.suggest_int('hidden_depth_hm', 3,5)
        config['weight_decay_hmL2'] = trial.suggest_float("weight_decay_hmL2", 1e-4, 1e-2, log=False)
        config['weight_decay_hmL1'] = trial.suggest_float("weight_decay_hmL1", 1e-4, 1e-2, log=False)
        config['input_shape_hm'] = hm_input_dim
        config['Att_HM_Agg'] = args.AttentionhomeMedsAgg
        if config['Att_HM_Agg'] == True:
            temp_flag = 0
            while temp_flag==0:
                config['Att_HM_agg_Heads'] = trial.suggest_int('Att_HM_agg_Heads', 2,8)
                if config['input_shape_hm'] % config['Att_HM_agg_Heads'] ==0:
                    temp_flag=1

    if 'pmh' in modality_to_use:

        pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert.csv')

        if args.pmhform == 'embedding_sum':
            pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
            pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
                'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                      fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
                {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
                ["person_integer"], axis=1)

            pmh_tr = torch.tensor(pmh_emb_sb_final.iloc[train_index].to_numpy(), dtype=torch.float32)
            pmh_te = torch.tensor(pmh_emb_sb_final.iloc[test_index].to_numpy(), dtype=torch.float32)
            pmh_val = torch.tensor(pmh_emb_sb_final.iloc[valid_index].to_numpy(), dtype=torch.float32)
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
            pmh_tr = torch.index_select(dense_pmh_embedding, 0, torch.tensor(train_index)).coalesce()
            pmh_val = torch.index_select(dense_pmh_embedding, 0, torch.tensor(valid_index)).coalesce()
            pmh_te = torch.index_select(dense_pmh_embedding, 0, torch.tensor(test_index)).coalesce()
            pmh_input_dim = len(col_names)

        config['hidden_units_pmh'] = trial.suggest_int('hidden_units_pmh', 5,16)
        config['hidden_units_final_pmh'] = trial.suggest_int('hidden_units_final_pmh', 5,16)
        config['hidden_depth_pmh'] = trial.suggest_int('hidden_depth_pmh', 3,5)
        config['weight_decay_pmhL2'] = trial.suggest_float("weight_decay_pmhL2", 1e-4, 1e-2, log=False)
        config['weight_decay_pmhL1'] = trial.suggest_float("weight_decay_pmhL1", 1e-4, 1e-2, log=False)
        config['input_shape_pmh'] = pmh_input_dim
        config['Att_pmh_Agg'] = args.AttentionPmhAgg
        if config['Att_pmh_Agg'] == True:
            temp_flag = 0
            while temp_flag==0:
                config['AttPmhAgg_Heads'] = trial.suggest_int('AttPmhAgg_Heads', 2,8)
                if config['input_shape_pmh'] % config['AttPmhAgg_Heads'] ==0:
                    temp_flag=1

    if 'problist' in modality_to_use:
        prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')

        if args.problistform == 'embedding_sum':
            prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
            prob_list_emb_sb_final = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
                'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                      fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
                {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
                ["person_integer"], axis=1)

            problist_tr = torch.tensor(prob_list_emb_sb_final.iloc[train_index].to_numpy(), dtype=torch.float32)
            problist_te = torch.tensor(prob_list_emb_sb_final.iloc[test_index].to_numpy(), dtype=torch.float32)
            problist_val = torch.tensor(prob_list_emb_sb_final.iloc[valid_index].to_numpy(), dtype=torch.float32)
            problist_input_dim = len(prob_list_emb_sb_final.columns)

        if args.problistform == 'embedding_attention':
            col_names = [col for col in prob_list_emb_sb.columns if 'sherbet' in col]
            prob_list_emb_sb.fillna(0, inplace=True)
            prob_list_emb_sb['pmh_pos'] = [item for idx in
                                           prob_list_emb_sb.groupby(by='orlogid_encoded')['ICD_10_CODES'].count()
                                           for item in range(idx)]
            prob_list_emb_sb1 = new_index.merge(pmh_emb_sb, on="orlogid_encoded", how="left").drop(
                ["orlogid_encoded"], axis=1).rename(
                {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
            prob_list_emb_sb1.fillna(0, inplace=True)  # setting the value for the ones that were added later

            index_problist_ids = torch.tensor(prob_list_emb_sb1[['person_integer', 'pmh_pos']].values, dtype=int)
            value_problist_embed = torch.tensor(prob_list_emb_sb1[col_names].astype('float').values, dtype=float)
            dense_problist_embedding = torch.sparse_coo_tensor(torch.transpose(index_problist_ids, 0, 1),
                                                               value_problist_embed,
                                                               dtype=torch.float32)
            problist_tr = torch.index_select(dense_problist_embedding, 0, torch.tensor(train_index)).coalesce()
            problist_val = torch.index_select(dense_problist_embedding, 0, torch.tensor(valid_index)).coalesce()
            problist_te = torch.index_select(dense_problist_embedding, 0, torch.tensor(test_index)).coalesce()
            problist_input_dim = len(col_names)

        config['hidden_units_problist'] = trial.suggest_int('hidden_units_problist', 5,16)
        config['hidden_units_final_problist'] = trial.suggest_int('hidden_units_final_problist', 5,16)
        config['hidden_depth_problist'] = trial.suggest_int('hidden_depth_problist', 3,5)
        config['weight_decay_problistL2'] = trial.suggest_float("weight_decay_problistL2", 1e-4, 1e-2, log=False)
        config['weight_decay_problistL1'] = trial.suggest_float("weight_decay_problistL1", 1e-4, 1e-2, log=False)
        config['input_shape_problist'] = problist_input_dim
        config['Att_problist_Agg'] = args.AttentionProblistAgg

    if 'flow' in modality_to_use:
        # flowsheet data
        very_dense_flow = feather.read_feather(data_dir + "flow_ts/Imputed_very_dense_flow.feather")
        very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace=True)
        very_dense_flow = very_dense_flow.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
        very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint']]
        very_dense_flow.drop(["endtime"], axis=1, inplace=True)

        other_intra_flow_wlabs = feather.read_feather(data_dir + "flow_ts/Imputed_other_flow.feather")
        other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index,
                                    inplace=True)
        other_intra_flow_wlabs = other_intra_flow_wlabs.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                              on="orlogid_encoded")
        other_intra_flow_wlabs = other_intra_flow_wlabs.loc[
            other_intra_flow_wlabs['endtime'] > other_intra_flow_wlabs['timepoint']]
        other_intra_flow_wlabs.drop(["endtime"], axis=1, inplace=True)

        very_dense_flow = very_dense_flow.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                                   axis=1).rename(
            {"new_person": "person_integer"}, axis=1)
        other_intra_flow_wlabs = other_intra_flow_wlabs.merge(new_index, on="orlogid_encoded", how="inner").drop(
            ["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)

        """ TS flowsheet proprocessing """
        # need to convert the type of orlogid_encoded from object to int
        other_intra_flow_wlabs['person_integer'] = other_intra_flow_wlabs['person_integer'].astype('int')
        very_dense_flow['person_integer'] = very_dense_flow['person_integer'].astype('int')

        index_med_other_flow = torch.tensor(
            other_intra_flow_wlabs[['person_integer', 'timepoint', 'measure_index']].values,
            dtype=int)
        value_med_other_flow = torch.tensor(other_intra_flow_wlabs['VALUE'].values)
        flowsheet_other_flow = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                                       value_med_other_flow, dtype=torch.float32)

        index_med_very_dense = torch.tensor(very_dense_flow[['person_integer', 'timepoint', 'measure_index']].values,
                                            dtype=int)
        value_med_very_dense = torch.tensor(very_dense_flow['VALUE'].values)
        flowsheet_very_dense_sparse_form = torch.sparse_coo_tensor(torch.transpose(index_med_very_dense, 0, 1),
                                                                   value_med_very_dense,
                                                                   dtype=torch.float32)  ## this is memory heavy and could be skipped, only because it is making a copy not really because it is harder to store
        flowsheet_very_dense = flowsheet_very_dense_sparse_form.to_dense()
        flowsheet_very_dense = torch.cumsum(flowsheet_very_dense, dim=1)

        # trying to concatenate the two types of flowsheet tensors at the measure_index dimension
        # flowsheet_dense_comb = torch.cat((flowsheet_very_dense, flowsheet_other_flow), dim=2)
        total_flowsheet_measures = other_intra_flow_wlabs['measure_index'].unique().max() + 1 + very_dense_flow[
            'measure_index'].unique().max() + 1  # plus 1 because of the python indexing from 0

        if (args.includeMissingnessMasks):
            """  Masks very dense flowsheet seperation based on train test; will double the dimension of flowsheet """

            # mask for very dense
            mask_flowsheet_very_dense = torch.sparse_coo_tensor(flowsheet_very_dense_sparse_form._indices(),
                                                                np.ones(
                                                                    len(flowsheet_very_dense_sparse_form._values())),
                                                                flowsheet_very_dense_sparse_form.size()).to_dense()

            total_flowsheet_measures = 2 * total_flowsheet_measures
            if args.task == 'endofcase':
                very_dense_tr_mask = torch.vstack([mask_flowsheet_very_dense[train_index, :, :]] * 2)
                very_dense_te_mask = torch.vstack([mask_flowsheet_very_dense[test_index, :, :]] * 2)
            else:
                very_dense_tr_mask = mask_flowsheet_very_dense[train_index, :, :]
                very_dense_te_mask = mask_flowsheet_very_dense[test_index, :, :]


        config['preops_init_flow'] = trial.suggest_categorical("preops_init_flow", [True, False])
        config['lstm_flow_hid'] = trial.suggest_int('lstm_flow_hid', 10,40)
        config['lstm_flow_num_layers'] = trial.suggest_int('lstm_flow_num_layers', 1,4)
        config['bilstm_flow'] = trial.suggest_categorical("bilstm_flow", [True, False])
        config['p_flow'] = trial.suggest_float("p_flow", 0.01, 0.25, log=False)
        config['weight_decay_LSTMflowL2'] = trial.suggest_float("weight_decay_LSTMflowL2", 1e-4, 1e-2, log=False)
        config['num_flowsheet_feat'] = total_flowsheet_measures

    if 'meds' in modality_to_use:
        # reading the med files
        all_med_data = feather.read_feather(data_dir + 'med_ts/intraop_meds_filterd.feather')
        all_med_data.drop(all_med_data[all_med_data['time'] > 511].index, inplace=True)
        all_med_data = all_med_data.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
        all_med_data = all_med_data.loc[all_med_data['endtime'] > all_med_data['time']]
        all_med_data.drop(["endtime"], axis=1, inplace=True)

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
        group_end = torch.cat(
            (torch.tensor(0).reshape((1)), group_end))  # prepending 0 to make sure that it is treated as an empty slot

        drug_med_ids = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_integer']]

        if args.drugNamesNo == True:
            drug_med_id_map = feather.read_feather(data_dir + 'med_ts/med_id_map.feather')
            drug_words = None
            word_id_map = None
        else:
            drug_words = feather.read_feather(data_dir + 'med_ts/drug_words.feather')
            drug_words.drop(drug_words[drug_words['timepoint'] > 511].index, inplace=True)
            word_id_map = feather.read_feather(data_dir + 'med_ts/word_id_map.feather')
            drug_med_id_map = None

        # drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'unit_integer',
        #                           'dose']]
        drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_unit_comb',
                                  'dose']]  # replacing the unit_integer column by med_unit_comb column

        unit_id_map = feather.read_feather(data_dir + 'med_ts/unit_id_map.feather')
        # vocab_len_units = len(unit_id_map)
        vocab_len_units = len(med_unit_unique_codes)  # replacing  len(unit_id_map) by len(med_unit_unique_codes)

        if args.drugNamesNo == False:
            vocab_len_words = len(word_id_map)
        else:
            vocab_len_med_ids = len(drug_med_id_map)

        drug_dose = drug_dose.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                       axis=1).rename(
            {"new_person": "person_integer"}, axis=1)

        if drug_words is not None:
            drug_words = drug_words.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                             axis=1).rename(
                {"new_person": "person_integer"}, axis=1)

        if drug_med_ids is not None:
            drug_med_ids = drug_med_ids.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                                 axis=1).rename(
                {"new_person": "person_integer"}, axis=1)

        ## I suppose these could have sorted differently
        ## TODO apparently, torch.from_numpy shares the memory buffer and inherits type
        index_med_ids = torch.tensor(drug_med_ids[['person_integer', 'time', 'drug_position']].values, dtype=int)
        index_med_dose = torch.tensor(drug_dose[['person_integer', 'time', 'drug_position']].values, dtype=int)
        value_med_dose = torch.tensor(drug_dose['dose'].astype('float').values, dtype=float)
        value_med_unit = torch.tensor(drug_dose['med_unit_comb'].values, dtype=int)

        add_unit = 0 in value_med_unit.unique()
        dense_med_units = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_unit + add_unit,
                                                  dtype=torch.int32)
        dense_med_dose = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_dose,
                                                 dtype=torch.float32)

        if args.drugNamesNo == True:
            value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
            add_med = 0 in value_med_ids.unique()
            dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0, 1), value_med_ids + add_med,
                                                    dtype=torch.int32)
        else:  ## not considered
            drug_words.dropna(axis=0, inplace=True)
            # convert name and unit+dose data seperately into the required format
            drug_words['time'] = drug_words['time'].astype('int64')
            drug_words['person_integer'] = drug_words['person_integer'].astype('int')
            index_med_names = torch.tensor(
                drug_words[['person_integer', 'time', 'drug_position', 'word_position']].values,
                dtype=int)
            value_med_name = torch.tensor(drug_words['word_integer'].values, dtype=int)
            add_name = 0 in value_med_name.unique()
            dense_med_names = torch.sparse_coo_tensor(torch.transpose(index_med_names, 0, 1),
                                                      value_med_name + add_name, dtype=torch.int32).to_dense()

        config['v_units'] = vocab_len_units
        config['v_med_ids'] = vocab_len_med_ids
        config['e_dim_med_ids'] = trial.suggest_int('e_dim_med_ids', 10,20)
        config['e_dim_units'] = False  # hardcoded for now
        config['preops_init_med'] = trial.suggest_categorical("preops_init_med", [True, False])
        config['lstm_hid'] = trial.suggest_int('lstm_hid', 10,40)
        config['lstm_num_layers'] = trial.suggest_int('lstm_num_layers', 1,4)
        config['bilstm_med'] = trial.suggest_categorical("bilstm_med", [True, False])
        config['p_idx_med_ids'] = 0  # putting these 0 because the to dense sets everything not available as 0
        config['p_idx_units'] = 0
        config['p_time'] = trial.suggest_float("p_time", 0.01, 0.25, log=False)
        config['p_rows'] = trial.suggest_float("p_rows", 0.01, 0.25, log=False)
        config['weight_decay_LSTMmedL2'] = trial.suggest_float("weight_decay_LSTMmedL2", 1e-5, 1e-2, log=False)
        config['group_start_list'] = group_start
        config['group_end_list'] = group_end
        config['Att_MedAgg'] = args.AttentionMedAgg
        if config['Att_MedAgg'] == True:
            temp_flag = 0
            while temp_flag==0:
                config['AttMedAgg_Heads'] = trial.suggest_int('AttMedAgg_Heads', 2,8)
                if config['e_dim_med_ids'] % config['AttMedAgg_Heads'] ==0:
                    temp_flag=1

    outcome_df.drop(["orlogid_encoded"], axis=1, inplace=True)
    outcome_df.reset_index(inplace=True)
    outcome_df.rename({"index": "person_integer"}, axis=1, inplace=True)

    print("Passed all the data processing stage")

    if args.task == 'endofcase':  ##I only included the first two timepoints; doing the rest requires either excluding cases so that all 4 are defined or more complex indexing
        data_tr = [torch.hstack([torch.tensor(preops_tr.to_numpy(), dtype=torch.float32),
                                 torch.tensor(endtimes.iloc[train_index]["true_test"].values,
                                              dtype=int).reshape(len(preops_tr), 1)]),
                   torch.tensor(endtimes.iloc[train_index]["true_test"].values, dtype=int),
                   torch.tensor(bow_input.iloc[train_index].to_numpy(), dtype=torch.float32),
                   hm_tr,
                   torch.index_select(dense_med_ids, 0, torch.tensor(train_index)).coalesce(),
                   torch.index_select(dense_med_dose, 0, torch.tensor(train_index)).coalesce(),
                   torch.index_select(dense_med_units, 0, torch.tensor(train_index)).coalesce(),
                   flowsheet_very_dense[train_index, :, :],
                   torch.index_select(flowsheet_other_flow, 0, torch.tensor(train_index)).coalesce(),
                   torch.tensor(endtimes.iloc[train_index]["t1"].values, dtype=int)
                   ]
        data_te = [torch.hstack([torch.tensor(preops_te.to_numpy(), dtype=torch.float32),
                                 torch.tensor(endtimes.iloc[test_index]["true_test"].values,
                                              dtype=int).reshape(len(preops_te), 1)]),
                   torch.tensor(endtimes.iloc[test_index]["true_test"].values, dtype=int),
                   torch.tensor(bow_input.iloc[test_index].to_numpy(), dtype=torch.float32),
                   hm_te,
                   torch.index_select(dense_med_ids, 0, torch.tensor(test_index)).coalesce(),
                   torch.index_select(dense_med_dose, 0, torch.tensor(test_index)).coalesce(),
                   torch.index_select(dense_med_units, 0, torch.tensor(test_index)).coalesce(),
                   flowsheet_very_dense[test_index, :, :],
                   torch.index_select(flowsheet_other_flow, 0, torch.tensor(test_index)).coalesce(),
                   torch.tensor(endtimes.iloc[test_index]["t1"].values, dtype=int)
                   ]
        data_va = [torch.hstack([torch.tensor(preops_val.to_numpy(), dtype=torch.float32),
                                 torch.tensor(endtimes.iloc[valid_index]["true_test"].values,
                                              dtype=int).reshape(len(preops_val), 1)]),
                   torch.tensor(endtimes.iloc[valid_index]["true_test"].values, dtype=int),
                   torch.tensor(bow_input.iloc[valid_index].to_numpy(), dtype=torch.float32),
                   hm_val,
                   torch.index_select(dense_med_ids, 0, torch.tensor(valid_index)).coalesce(),
                   torch.index_select(dense_med_dose, 0, torch.tensor(valid_index)).coalesce(),
                   torch.index_select(dense_med_units, 0, torch.tensor(valid_index)).coalesce(),
                   flowsheet_very_dense[valid_index, :, :],
                   torch.index_select(flowsheet_other_flow, 0, torch.tensor(valid_index)).coalesce(),
                   torch.tensor(endtimes.iloc[valid_index]["t1"].values, dtype=int)
                   ]
    else:
        data_tr = {}
        data_tr['outcomes'] = torch.tensor(outcome_df.iloc[train_index]["outcome"].values)
        data_tr['endtimes'] = torch.tensor(endtimes.iloc[train_index]["endtime"].values, dtype=int)
        data_val = {}
        data_val['outcomes'] = torch.tensor(outcome_df.iloc[valid_index]["outcome"].values)
        data_val['endtimes'] = torch.tensor(endtimes.iloc[valid_index]["endtime"].values, dtype=int)
        data_te = {}
        data_te['outcomes'] = torch.tensor(outcome_df.iloc[test_index]["outcome"].values)
        data_te['endtimes'] = torch.tensor(endtimes.iloc[test_index]["endtime"].values, dtype=int)
        if 'preops' in modality_to_use:
            data_tr['preops'] = torch.tensor(preops_tr.to_numpy(), dtype=torch.float32)
            data_tr['cbow'] = torch.tensor(bow_input.iloc[train_index].to_numpy(), dtype=torch.float32)
            data_val['preops'] = torch.tensor(preops_val.to_numpy(), dtype=torch.float32)
            data_val['cbow'] = torch.tensor(bow_input.iloc[valid_index].to_numpy(), dtype=torch.float32)
            data_te['preops'] = torch.tensor(preops_te.to_numpy(), dtype=torch.float32)
            data_te['cbow'] = torch.tensor(bow_input.iloc[test_index].to_numpy(), dtype=torch.float32)

        if 'homemeds' in modality_to_use:
            data_tr['homemeds'] = hm_tr
            data_val['homemeds'] = hm_val
            data_te['homemeds'] = hm_te

        if 'pmh' in modality_to_use:
            data_tr['pmh'] = pmh_tr
            data_val['pmh'] = pmh_val
            data_te['pmh'] = pmh_te

        if 'problist' in modality_to_use:
            data_tr['problist'] = problist_tr
            data_val['problist'] = problist_val
            data_te['problist'] = problist_te

        if 'flow' in modality_to_use:
            data_tr['flow'] = [flowsheet_very_dense[train_index, :, :],
                               torch.index_select(flowsheet_other_flow, 0, torch.tensor(train_index)).coalesce()]
            data_val['flow'] = [flowsheet_very_dense[valid_index, :, :],
                                torch.index_select(flowsheet_other_flow, 0, torch.tensor(valid_index)).coalesce()]
            data_te['flow'] = [flowsheet_very_dense[test_index, :, :],
                               torch.index_select(flowsheet_other_flow, 0, torch.tensor(test_index)).coalesce()]

        if 'meds' in modality_to_use:
            data_tr['meds'] = [torch.index_select(dense_med_ids, 0, torch.tensor(train_index)).coalesce(),
                               torch.index_select(dense_med_dose, 0, torch.tensor(train_index)).coalesce(),
                               torch.index_select(dense_med_units, 0, torch.tensor(train_index)).coalesce()]
            data_val['meds'] = [torch.index_select(dense_med_ids, 0, torch.tensor(valid_index)).coalesce(),
                                torch.index_select(dense_med_dose, 0, torch.tensor(valid_index)).coalesce(),
                                torch.index_select(dense_med_units, 0, torch.tensor(valid_index)).coalesce()]
            data_te['meds'] = [torch.index_select(dense_med_ids, 0, torch.tensor(test_index)).coalesce(),
                               torch.index_select(dense_med_dose, 0, torch.tensor(test_index)).coalesce(),
                               torch.index_select(dense_med_units, 0, torch.tensor(test_index)).coalesce()]

    config['modality_used'] = modality_to_use
    device = torch.device('cuda')

    if args.modelType == 'transformer':
        config['e_dim_flow'] =  trial.suggest_int('e_dim_flow', 10,20)  # this is needed because when combining the meds and flowsheets for attention, meds have been emmbedded but flowsheets are raw
        config['AttTS_depth'] = trial.suggest_int('AttTS_depth', 4,6)
        config['cnn_before_Att'] = trial.suggest_categorical("cnn_before_Att", [True, False])
        if config['cnn_before_Att'] == True:
            config['kernel_size_conv'] = trial.suggest_int('kernel_size_conv', 2,10)
            config['stride_conv'] = trial.suggest_int('stride_conv', 1,3)
        config['ats_dropout'] = trial.suggest_float("ats_dropout", 0.01, 0.3, log=False)

        # breakpoint()
        # for getting a compatible attention head value
        temp_flag = 0
        config['AttTS_Heads'] = trial.suggest_int('AttTS_Heads', 2, 10)
        while temp_flag == 0:
            if 'meds' in modality_to_use and 'flow' in modality_to_use:
                if config['e_dim_med_ids'] + config['e_dim_flow'] % config['AttTS_Heads'] == 0:
                    temp_flag = 1
            elif 'flow' in modality_to_use:
                if config['e_dim_flow'] % config['AttTS_Heads'] == 0:
                    temp_flag = 1
            elif 'meds' in modality_to_use:
                if config['e_dim_med_ids'] % config['AttTS_Heads'] == 0:
                    temp_flag = 1
            config['AttTS_Heads'] = trial.suggest_int('AttTS_Heads', 2+config['AttTS_Heads'], 10+config['AttTS_Heads'])


        # breakpoint()
        model = ts_model_class_Optuna_hp_tune.TS_Transformer_Med_index(**config).to(device)
    else:
        model = ts_model_class_Optuna_hp_tune.TS_lstm_Med_index(**config).to(device)

    batchSize = trial.suggest_int("batchSize",110,128)

    learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

    # lr scheduler
    LRPatience = trial.suggest_int("LRPatience",2,6)
    learningRateFactor =trial.suggest_float("learningRateFactor", 0.1, 0.5, log=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LRPatience, verbose=True,
                                                           factor=learningRateFactor)

    # initializing the loss function
    if not binary_outcome:
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.BCELoss()

    total_train_loss = []
    total_test_loss = []

    updating_lr = learningRate
    best_metric = 1000  # some large number
    lr_schedular_epoch_dict = {}
    lr_schedular_epoch_dict[0] = updating_lr

    num_epochs = 50

    for epoch in range(num_epochs):  # setting a max of 100 on the number of batches
        loss_tr = 0
        loss_tr_cls = 0
        model.train()
        ## the default __getitem__ is like 2 orders of magnitude slower
        shuffle_index = torch.randperm(n=data_tr['outcomes'].shape[0])
        if (args.overSampling == True) and (args.task != 'endofcase'):
            pos_idx = (data_tr['outcomes'] == 1).nonzero()
            neg_idx = (data_tr['outcomes'] == 0).nonzero()
            if batchSize % 2 == 0:  # this is done because it was creating a problem when the batchsize was an odd number
                nbatch = neg_idx.shape[0] // int(batchSize / 2)
            else:
                nbatch = neg_idx.shape[0] // math.ceil(batchSize / 2)
        else:
            nbatch = data_tr['outcomes'].shape[0] // batchSize
        for i in range(nbatch):
            if (args.overSampling == True) and (args.task != 'endofcase'):
                if batchSize % 2 == 0:
                    neg_indexbatch = neg_idx[range(i * int(batchSize / 2), (i + 1) * int(batchSize / 2))]
                else:
                    neg_indexbatch = neg_idx[
                        range(i * math.ceil(batchSize / 2), (i + 1) * math.ceil(batchSize / 2))]
                p = torch.from_numpy(np.repeat([1 / len(pos_idx)], len(pos_idx)))
                pos_indexbatch = pos_idx[p.multinomial(num_samples=int(batchSize / 2),
                                                       replacement=True)]  # this is sort of an equivalent of numpy.random.choice
                these_index = torch.vstack([neg_indexbatch, pos_indexbatch]).reshape([batchSize])
            else:
                these_index = shuffle_index[range(i * batchSize, (i + 1) * batchSize)]

            ## this collate method is pretty inefficent for this task but works with the generic DataLoader method

            local_data = {}
            for k in data_tr.keys():
                if type(data_tr[k]) != list:
                    local_data[k] = torch.index_select(data_tr[k], 0, these_index)
                else:
                    local_data[k] = [torch.index_select(x, 0, these_index) for x in data_tr[k]]

            if args.task == 'endofcase':
                local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                    len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
                local_data[0][:, -1] = local_data[
                    1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
                local_data[-1] = torch.from_numpy(
                    np.repeat([1, 0], [int(batchSize / 2), batchSize - int(batchSize / 2)]))
            if (args.includeMissingnessMasks):  # appending the missingness masks in training data
                if 'preops' in modality_to_use:
                    local_data['preops'] = [local_data['preops'],
                                            torch.tensor(preops_tr_mask.iloc[these_index].to_numpy(),
                                                         dtype=torch.float32)]
                if 'flow' in modality_to_use:
                    local_data['flow'].append(very_dense_tr_mask[these_index, :, :])
                    sparse_mask = torch.sparse_coo_tensor(local_data['flow'][1]._indices(),
                                                          np.ones(len(local_data['flow'][1]._values())),
                                                          local_data['flow'][1].size())
                    local_data['flow'].append(sparse_mask)

            data_train, mod_order_dict = ts_model_class_Optuna_hp_tune.collate_time_series(local_data, device)

            # reset the gradients back to zero as PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            y_pred, reg_loss = model(data_train[0])
            cls_loss_tr = criterion(y_pred.squeeze(-1), data_train[1].float().to(device)).float()
            train_loss = cls_loss_tr + reg_loss
            train_loss.backward()
            optimizer.step()
            loss_tr += train_loss.item()
            loss_tr_cls += cls_loss_tr.item()
        loss_tr = loss_tr / data_tr['outcomes'].shape[0]
        loss_tr_cls = loss_tr_cls / data_tr['outcomes'].shape[0]

        loss_te = 0
        loss_te_cls = 0
        with torch.no_grad():
            model.eval()
            true_y_test = []
            pred_y_test = []
            nbatch = data_val['outcomes'].shape[0] // batchSize
            if nbatch > 0:  # this is to make sure that some data points are being used and the code doesn't break. This will only happen during the development time because the val set could be smaller
                batchSize1 = batchSize
            else:
                batchSize1 = data_val['outcomes'].shape[0]
                nbatch = 1
            for i in range(nbatch):
                these_index = torch.tensor(list(range(i * batchSize1, (i + 1) * batchSize1)), dtype=int)
                local_data = {}
                for k in data_val.keys():
                    if type(data_val[k]) != list:
                        local_data[k] = torch.index_select(data_val[k], 0, these_index)
                    else:
                        local_data[k] = [torch.index_select(x, 0, these_index) for x in data_val[k]]

                if args.task == 'endofcase':
                    local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                        len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
                    local_data[0][:, -1] = local_data[
                        1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
                    local_data[-1] = torch.from_numpy(
                        np.repeat([1, 0], [int(batchSize1 / 2), batchSize1 - int(batchSize1 / 2)]))
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

                data_valid, mod_order_dict = ts_model_class_Optuna_hp_tune.collate_time_series(local_data, device)

                y_pred, reg_loss = model(data_valid[0])
                cls_loss_te = criterion(y_pred.squeeze(-1), data_valid[1].float().to(device)).float()
                test_loss = cls_loss_te + reg_loss
                loss_te += test_loss.item()
                loss_te_cls += cls_loss_te.item()

                # values from the last epoch; it will get overwritten
                true_y_test.append(data_valid[1].float().detach().numpy())
                pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())

            loss_te = loss_te / data_val['outcomes'].shape[0]
            loss_te_cls = loss_te_cls / data_val['outcomes'].shape[0]

            if best_metric > loss_te_cls:
                best_metric = loss_te_cls
                # torch.save(model.state_dict(), PATH)
                pred_y_test_best = pred_y_test

            # display the epoch training and test loss
            print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, num_epochs,
                                                                                           loss_tr_cls, loss_te_cls))
            total_train_loss.append(loss_tr)
            total_test_loss.append(loss_te)

        scheduler.step(loss_te_cls)

        if optimizer.param_groups[0]['lr'] != updating_lr:
            updating_lr = optimizer.param_groups[0]['lr']
            lr_schedular_epoch_dict[epoch] = updating_lr

        try:
            # this is needed because in every batch the labels are saved as a list and need concatenation for accesing the whol valid set labels
            true_y_test = np.concatenate(true_y_test)
            pred_y_test = np.concatenate(pred_y_test)
        except(ValueError):
            print("---Debug---")
            breakpoint()
        try:
            val_auroc = roc_auc_score(true_y_test, pred_y_test)
            val_auprc = average_precision_score(true_y_test, pred_y_test)
        except(ValueError):
            val_auroc=0
            print("THE VALIDATION SET DIDN'T HAVE ANY POSITIVE EXAMPLES")
            break
        trial.report(val_auroc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # print("current lr ", optimizer.param_groups[0]['lr'])
        # 1e-8 is obtained by multiplying 1e-3 by (0.25)^5 so to make it general we can have initial_learning_rate * (learningRateFactor)^5
        if optimizer.param_groups[0]['lr'] <= learningRate * np.power(learningRateFactor,
                                                                           5):  # hardcoding for now because our schedule is such that 10**-3 * (1, 1/4, 1/16, 1/64, 1/256, 1/1024, 0) with an initial rate of 10**-3 an learning rate factor of 0.25
            print("inside the early stopping loop")
            print("best validation loss ", best_metric)
            break

        print('AUROC and AUPRC for the validation set', val_auroc, val_auprc)
    return val_auroc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TS modular different model training')

    ## modalities to select
    parser.add_argument('--preops', action="store_true",
                        help='Whether to add preops and bow to ts representation')
    parser.add_argument('--pmhProblist', action="store_true",
                        help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
    parser.add_argument('--homemeds', action="store_true",
                        help='Whether to add homemeds to ts representation')
    parser.add_argument('--meds', action="store_true",
                        help='Whether to add meds to ts representation')
    parser.add_argument('--flow', action="store_true",
                        help='Whether to add flowsheets to ts representation')

    # these are some of the choices that do not need hp tuner but would be decided by the user
    parser.add_argument("--home_medsform",
                        default='embedding_sum')  # options {'ohe', 'embedding_sum', 'embedding_attention'}
    parser.add_argument("--AttentionhomeMedsAgg", default=False,
                        action='store_true')  # this needs to be true when embedding_attention is active in the above line

    ## for the past medical history before concat to ts output
    parser.add_argument("--pmhform", default='embedding_sum')  # options { 'embedding_sum', 'embedding_attention'}
    parser.add_argument("--AttentionPmhAgg", default=False,
                        action='store_true')  # this needs to be true when embedding_attention is active in the above line

    ## for the problem list before concat to ts output
    parser.add_argument("--problistform", default='embedding_sum')  # options {'embedding_sum', 'embedding_attention'}
    parser.add_argument("--AttentionProblistAgg", default=False,
                        action='store_true')  # this needs to be true when embedding_attention is active in the above line

    ## for processing medication IDs (or the post-embedding words)
    parser.add_argument("--AttentionMedAgg", default=False, action='store_true')

    ## for model type for time series
    parser.add_argument("--modelType", default='lstm')  # options {'lstm', 'transformer'}

    parser.add_argument("--drugNamesNo", default=True, action='store_true')  # whether to use the med id or name
    parser.add_argument("--includeMissingnessMasks", default=False, action='store_true')
    parser.add_argument("--overSampling", default=True, action='store_true')  # keep it as False when task is endofcase

    parser.add_argument("--randomSeed", default=100, type=int)
    parser.add_argument("--task", default="icu")
    parser.add_argument("--numtrialsHP", default=25, type=int)

    args_input = parser.parse_args()

    # this is sort of repeated here but for now its fine
    modality_to_use = []
    if eval('args_input.preops') == True:
        modality_to_use.append('preops')
        modality_to_use.append('cbow')

    if eval('args_input.pmhProblist') == True:
        modality_to_use.append('pmh')
        modality_to_use.append('problist')

    if eval('args_input.homemeds') == True:
        modality_to_use.append('homemeds')

    if eval('args_input.flow') == True:
        modality_to_use.append('flow')

    if eval('args_input.meds') == True:
        modality_to_use.append('meds')

    modalities_to_add = '_modal_'
    for i in range(len(modality_to_use)):
        modalities_to_add = modalities_to_add + "_" + modality_to_use[i]

    std_name = str(args_input.task)+"_"+str(args_input.modelType)+modalities_to_add + "_"+str(args_input.randomSeed) +"_"
    study = optuna.create_study(direction="maximize", study_name=std_name)
    study.set_metric_names(["Validation_set_auroc"])

    study.optimize(lambda trial: objective(trial, args_input), n_trials=args_input.numtrialsHP, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    trial_summary_df = study.trials_dataframe()  # this

    hpcsv = os.path.join('/output/', std_name+ datetime.now().strftime("%y-%m-%d-%H:%M:%S") + "_HP_df.csv")
    trial_summary_df.to_csv(hpcsv, header=True, index=False)

    best_Trial_metadata={}
    best_Trial_metadata['params'] =study.best_params
    # best_Trial_metadata['trial'] = study.best_trial
    best_Trial_metadata['value'] = study.best_value

    # best_trial_file_name = '/output/Best_trial_result' + std_name + datetime.now().strftime("%y-%m-%d-%H:%M:%S")+'.txt' # frozentrial is not serializable so can't save it. The ideal way would be to use optuna storage but for not using this.
    best_trial_file_name = '/output/Best_trial_result' + std_name + datetime.now().strftime("%y-%m-%d-%H:%M:%S")+'.json' # frozentrial is not serializable so can't save it.


    # with open(best_trial_file_name, 'w') as f: print(best_Trial_metadata, file=f)
    with open(best_trial_file_name, 'w') as outfile: json.dump(best_Trial_metadata, outfile)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))