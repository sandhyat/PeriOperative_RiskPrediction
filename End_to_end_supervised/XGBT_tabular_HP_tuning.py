
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
import optuna
from optuna.trial import TrialState
import Preops_processing as pps


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

    print("MODALITY TO USE ", modality_to_use)

    # reproducibility settings
    # random_seed = 1 # or any of your favorite number
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
                               'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                               'opioids_count_day0', 'opioids_count_day1']
    binary_outcome = args.task not in regression_outcome_list

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

    if False:
        combined_case_set = np.random.choice(combined_case_set, 5000, replace=False)

    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
    preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]

    outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

    # checking for NA and other filters
    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
    outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
    new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"},
                                                                          axis=1)  # this df basically reindexes everything so from now onwards orlogid_encoded is an integer


    preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

    train_set = []
    valid_set = []

    features = []
    if 'preops' not in modality_to_use:
        test_size = 0.2
        valid_size = 0.05  # change back to 0.00005 for the full dataset
        y_outcome = outcome_df["outcome"].values
        preops.reset_index(drop=True, inplace=True)
        upto_test_idx = int(test_size * len(preops))
        test = preops.iloc[:upto_test_idx]
        train0 = preops.iloc[upto_test_idx:]
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=args.randomSeed,
                                            stratify=y_outcome[train0.index])
        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=args.randomSeed)

        train_index = train.index
        valid_index = valid.index
        test_index = test.index

        if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
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

        # currently sacrificing 5 data points in the valid set and using the test set to finally compute the auroc etc
        preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = pps.preprocess_train(preops,
                                                                                                               args.task,
                                                                                                               y_outcome=
                                                                                                               outcome_df[
                                                                                                                   "outcome"].values,
                                                                                                               binary_outcome=binary_outcome,
                                                                                                               valid_size=0.05,
                                                                                                               random_state=args.randomSeed,
                                                                                                               input_dr=data_dir,
                                                                                                               output_dr=out_dir)  # change back to 0.00005

        if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set (value of plannedDispo are numeric after processing; the df has also been changed )
            valid_index = preops.iloc[valid_index][preops.iloc[valid_index]['plannedDispo'] != 3]['plannedDispo'].index
            preops_val = preops_val.loc[valid_index]  # pay attention to indexing here

        # this is local to this file because later on we nee the min and max of each column to not be the same:
        if preops_tr.columns[preops_tr.min(axis=0) == preops_tr.max(axis=0)].all() != None:
            list_col = preops_tr.columns[preops_tr.min(axis=0) == preops_tr.max(axis=0)]
            preops_tr = preops_tr.drop(list_col, axis=1)
            preops_val = preops_val.drop(list_col, axis=1)

        bow_tr = bow_input.iloc[train_index]
        bow_val = bow_input.iloc[valid_index]

        train_set.append(preops_tr)
        train_set.append(bow_tr)
        valid_set.append(preops_val)
        valid_set.append(bow_val)

        features = features + list(preops_tr.columns) + list(bow_input.columns)

    if 'homemeds' in modality_to_use:
        home_medsform = trial.suggest_categorical("home_medsform", ["ohe", "embedding_sum"])
        # home meds reading and processing
        home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
        Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')


        home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded',
                                                      'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now

        if home_medsform == 'ohe':
            # home meds basic processing
            rxcui_freq = home_meds["rxcui"].value_counts().reset_index()
            # rxcui_freq = rxcui_freq.rename({'count':'rxcui_freq', 'rxcui':'rxcui'}, axis =1)
            rxcui_freq = rxcui_freq.rename({'rxcui': 'rxcui_freq', 'index': 'rxcui'}, axis=1)
            home_meds_small = home_meds[
                home_meds['rxcui'].isin(list(rxcui_freq[rxcui_freq['rxcui_freq'] > 100]['rxcui']))]
            home_meds_small['temp_const'] = 1
            home_meds_ohe = home_meds_small[['orlogid_encoded', 'rxcui', 'temp_const']].pivot_table(
                index='orlogid_encoded',
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
            home_meds_final = home_meds_ohe
        if home_medsform == 'embedding_sum':
            home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left',
                                                                               on='rxcui')
            home_meds_embedded.drop(columns=['code', 'description', 'source'], inplace=True)
            home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
            home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index(
                'new_person').reindex(
                list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
                ["orlogid_encoded"],
                axis=1).rename(
                {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
                ["person_integer"], axis=1)
            home_meds_sum.fillna(0, inplace=True)  # setting the value for the ones that were added later
            home_meds_sum = home_meds_sum.drop(["rxcui"], axis=1)
            home_meds_final = home_meds_sum

        hm_tr = home_meds_final.iloc[train_index]
        hm_val = home_meds_final.iloc[valid_index]
        hm_input_dim = len(home_meds_final.columns)

        train_set.append(hm_tr)
        valid_set.append(hm_val)

        features = features + list(home_meds_final.columns)

    if 'pmh' in modality_to_use:
        pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert.csv')

        pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                  fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)

        pmh_tr = pmh_emb_sb_final.iloc[train_index]
        pmh_val = pmh_emb_sb_final.iloc[valid_index]
        pmh_input_dim = len(pmh_emb_sb_final.columns)

        train_set.append(pmh_tr)
        valid_set.append(pmh_val)

        features = features + list(pmh_emb_sb_final.columns)

    if 'problist' in modality_to_use:
        prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')

        prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        prob_list_emb_sb_final = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                  fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)

        problist_tr = prob_list_emb_sb_final.iloc[train_index]
        problist_val = prob_list_emb_sb_final.iloc[valid_index]
        problist_input_dim = len(prob_list_emb_sb_final.columns)

        train_set.append(problist_tr)
        valid_set.append(problist_val)

        features = features + list(prob_list_emb_sb_final.columns)

    outcome_df.drop(["orlogid_encoded"], axis=1, inplace=True)
    outcome_df.reset_index(inplace=True)
    outcome_df.rename({"index": "person_integer"}, axis=1, inplace=True)

    train_data = np.concatenate(train_set, axis=1)
    valid_data = np.concatenate(valid_set, axis=1)

    y_train = outcome_df.iloc[train_index]["outcome"].values
    y_valid = outcome_df.iloc[valid_index]["outcome"].values

    # xgbt tuning parameter
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 9, step=2)
    learning_rate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 250)

    if torch.cuda.is_available():
        device_available = torch.device("cuda")
    else:
        device_available = torch.device("cpu")

    # the usage of device here not an efficient solution. Ideally I should be using a cupy library to move the numpy to cuda device so that both the data and classifier are on the same device.
    # However, XGB internally does this part which could be slow. TODO: for future implementation
    if binary_outcome:
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha, learning_rate=learning_rate, random_state=args.randomSeed, device=device_available)
        clf.fit(train_data, y_train)

        preds_valid = clf.predict_proba(valid_data)
        valid_auroc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)
        valid_auprc = average_precision_score(y_score=preds_valid[:, 1], y_true=y_valid)
        valid_metric = valid_auroc

        print('AUROC and AUPRC for the validation set', valid_auroc, valid_auprc)
    else:
        clf = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha, learning_rate=learning_rate, random_state=args.randomSeed, device=device_available)
        clf.fit(train_data, y_train)

        preds_valid = clf.predict(valid_data)
        r2value = r2_score(np.array(y_valid), np.array(preds_valid))  # inbuilt function also exists for R2
        valid_metric = r2value

    return valid_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tabular modular XGBT HP training')

    ## modalities to select
    parser.add_argument('--preops', default=True, action='store_true',
                        help='Whether to add preops and bow to ts representation')
    parser.add_argument('--pmhProblist', action="store_true",
                        help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
    parser.add_argument('--homemeds', action="store_true",
                        help='Whether to add homemeds to ts representation')

    parser.add_argument("--modelType", default='XGBT')  # options {'TabNet', others later}

    parser.add_argument("--randomSeed", default=100, type=int)
    parser.add_argument("--task", default="icu")
    parser.add_argument("--numtrialsHP", default=25, type=int)

    args_input = parser.parse_args()

    modality_to_use = []
    if eval('args_input.preops') == True:
        modality_to_use.append('preops')
        modality_to_use.append('cbow')

    if eval('args_input.pmhProblist') == True:
        modality_to_use.append('pmh')
        modality_to_use.append('problist')

    if eval('args_input.homemeds') == True:
        modality_to_use.append('homemeds')

    modalities_to_add = '_modal_'
    for i in range(len(modality_to_use)):
        modalities_to_add = modalities_to_add + "_" + modality_to_use[i]

    std_name = str(args_input.task)+"_"+str(args_input.modelType)+modalities_to_add +str(args_input.randomSeed) +"_"
    study = optuna.create_study(direction="maximize", study_name=std_name)
    study.set_metric_names(["Validation_set_aurocOrR2"])

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