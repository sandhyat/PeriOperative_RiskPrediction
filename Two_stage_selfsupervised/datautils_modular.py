import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
import json
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.profiler import profile, record_function, ProfilerActivity
from pyarrow import feather  # directly writing import pyarrow didn't work

import sys
sys.path.append("..")
from End_to_end_supervised import preprocess_train

class Med_embedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]

        self.embed_layer_med_ids = nn.Embedding(
            num_embeddings=self.v_med_ids + 1,
            embedding_dim=self.e_dim_med_ids,
            padding_idx=self.p_idx_med_ids
        )  # there is NA already in the dictionary that can be used a padding token

        ## NOTE: for the multiply structure, need to be conformable
        if self.e_dim_units is True:
            self.e_dim_units = self.e_dim_med_ids
        else:
            self.e_dim_units = 1
        self.embed_layer_units = nn.Embedding(
            num_embeddings=self.v_units + 1,
            embedding_dim=self.e_dim_units,
            padding_idx=self.p_idx_units
        )

        self.linear = nn.Linear(in_features=self.e_dim_med_ids, out_features=1)

    def forward(self, medication_ids, dose, units, test=0):

        # if test==1:
        #     print("debug")
        #     breakpoint()
        # breakpoint()
        # Get a list of all non-garbage collected objects and their sizes.
        # non_gc_objects_with_size_After = get_non_gc_objects_with_size()
        #
        # # Print the number of non-garbage collected objects and their total size.
        # print("before forward ",len(non_gc_objects_with_size_After), sum([size for obj, size in non_gc_objects_with_size_After]))
        #
        # print("input min max")
        # print(medication_ids.min(), medication_ids.max())
        # print(units.min(), units.max())
        #
        # if units.min()<0 or units.max() > 219:
        #     print("debug")
        #     breakpoint()

        units_embedding = self.embed_layer_units(units.long())
        med_ids_temp_embedding = self.embed_layer_med_ids(medication_ids.long())
        med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        med_combined_embed = torch.sum(med_combined_embed, 2)

        outcome_pred = self.linear(torch.sum(med_combined_embed,1))
        outcome_pred = torch.sigmoid(outcome_pred)


        # # Get a list of all non-garbage collected objects and their sizes.
        # non_gc_objects_with_size_After = get_non_gc_objects_with_size()
        #
        # # Print the number of non-garbage collected objects and their total size.
        # print("after forward ", len(non_gc_objects_with_size_After), sum([size for obj, size in non_gc_objects_with_size_After]))
        # print("-----\n")

        return med_combined_embed, outcome_pred


class customdataset(Dataset):
    def __init__(self, data, outcome, transform = None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------
        data: multidimensional torch tensor
        """
        self.n = data[0].shape[0]
        self.y = outcome
        self.X_1 = data[0]
        self.X_2 = data[1]
        self.X_3 = data[2]

        self.transform = transform

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx],[self.X_1[idx], self.X_2[idx], self.X_3[idx]]]


import gc

def get_object_size(obj):
    """Returns the size of the given object in bytes."""
    import sys
    return sys.getsizeof(obj)

def get_non_gc_objects_with_size():
    """Returns a list of all non-garbage collected objects and their sizes."""
    objects = []
    for obj in gc.get_objects():
        if not gc.is_tracked(obj):
            objects.append((obj, get_object_size(obj)))
    return objects

def load_epic_mv(outcome, modality_to_uselist, randomSeed, data_dir, out_dir):
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
    regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood',
                               'postop_Vent_duration', 'n_glu_high',
                               'low_sbp_time', 'aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                               'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                               'opioids_count_day0', 'opioids_count_day1']
    binary_outcome = outcome not in regression_outcome_list

    config = dict(
        linear_out=1
    )
    config['binary'] = binary_outcome

    if outcome == 'icu':
        outcomes = outcomes.dropna(subset=['ICU'])
    outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

    # exclude very short cases (this also excludes some invalid negative times)
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]
    # end_of_case_times1 = end_of_case_times1.loc[end_of_case_times1['endtime'] > 30]

    if outcome == 'endofcase':
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

    if outcome in regression_outcome_list:
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

        outcome_df = outcomes[['orlogid_encoded', outcome]]
    elif outcome  in binary_outcome_list:
        if outcome  == 'VTE':
            temp_outcome = outcomes[['orlogid_encoded']]
            temp_outcome[outcome] =  np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
            temp_outcome.loc[temp_outcome[outcome] == 2, outcomes] = 1
        elif outcome == 'n_glucose_low':
            temp_outcome = outcomes[['orlogid_encoded', outcome]]
            temp_outcome[outcome] = temp_outcome[outcome].fillna(0)
            temp_outcome[outcome] = np.where(temp_outcome[outcome] > 0, 1, 0)
        else:
            temp_outcome = outcomes[['orlogid_encoded', outcome]]
            temp_outcome.loc[temp_outcome[outcome] == True, outcome] = 1
            temp_outcome.loc[temp_outcome[outcome] == False, outcome] = 0
        temp_outcome[outcome] = temp_outcome[outcome].astype(int)
        outcome_df = temp_outcome
    elif (outcome == 'dvt_pe'):
        dvt_pe_outcome = outcomes[['orlogid_encoded', 'DVT_PE']]
        outcome_df = dvt_pe_outcome
    elif (outcome == 'icu'):
        icu_outcome = outcomes[['orlogid_encoded', 'ICU']]
        icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
        icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
        icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)
        outcome_df = icu_outcome
    elif (outcome == 'mortality'):
        mortality_outcome = outcomes[['orlogid_encoded', 'death_in_30']]
        mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
        mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
        mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)
        outcome_df = mortality_outcome

    elif (outcome == 'aki1' or outcome == 'aki2' or outcome == 'aki3'):
        aki_outcome = outcomes[['orlogid_encoded', 'post_aki_status']]
        aki_outcome = aki_outcome.dropna(subset=[
            'post_aki_status'])  # this is droping the patients with baseline kidney failure as they are now post_aki_status = NA_integer_
        if outcome == 'aki1':
            aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
            aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
        if outcome == 'aki2':
            aki_outcome.loc[aki_outcome[
                                'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
            aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
        if outcome == 'aki3':
            aki_outcome.loc[aki_outcome[
                                'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will become zero :(; there is a one liner too that can be used
            aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
        aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)
        outcome_df = aki_outcome
    elif (outcome == 'endofcase'):
        outcome_df = end_of_case_times[['orlogid_encoded', 'true_test']]
    else:
        raise Exception("outcome not handled")

    ## intersect 3 mandatory data sources: preop, outcome, case end times
    combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(
        set(end_of_case_times['orlogid_encoded'].values)).intersection(set(preops['orlogid_encoded'].values)))

    # combined_case_set1 = list(set(outcome_df1["orlogid_encoded"].values).intersection(set(end_of_case_times1['orlogid_encoded'].values)).intersection(set(preops1['orlogid_encoded'].values)))
    # combined_case_set = combined_case_set + combined_case_set1
    # outcome_df = pd.concat([outcome_df, outcome_df1], axis=0)
    # end_of_case_times = pd.concat([end_of_case_times, end_of_case_times1], axis=0)

    if False:
        combined_case_set = np.random.choice(combined_case_set, 5000, replace=False)
        # combined_case_set1 = np.random.choice(combined_case_set1, 2500, replace=False)
        # combined_case_set = list(combined_case_set) + list(combined_case_set1)
        # combined_case_set = np.concatenate([combined_case_set, combined_case_set1])

    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
    preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
    # preops1 = preops1.loc[preops1['orlogid_encoded'].isin(combined_case_set)]
    end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

    outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

    # checking for NA and other filters
    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
    outcome_df['orlogid_encoded'] = outcome_df['orlogid_encoded'].astype('str')
    outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
    new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"},
                                                                          axis=1)  # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

    end_of_case_times['orlogid_encoded'] = end_of_case_times['orlogid_encoded'].astype('str')
    preops['orlogid_encoded'] = preops['orlogid_encoded'].astype('str')
    endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                          axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

    # preop_comb = pd.concat([preops, preops1], axis=0)

    preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

    if 'preops_o' not in modality_to_uselist:
        test_size = 0.2
        valid_size = 0.05  # change back to 0.00005 for the full dataset
        y_outcome = outcome_df["outcome"].values
        preops.reset_index(drop=True, inplace=True)
        upto_test_idx = int(test_size * len(preops))
        test = preops.iloc[:upto_test_idx]
        train0 = preops.iloc[upto_test_idx:]
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=randomSeed,
                                            stratify=y_outcome[train0.index])
        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=randomSeed)

        train_index = train.index
        valid_index = valid.index
        test_index = test.index

        if outcome == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 'ICU']['plannedDispo'].index

    if ('preops_o' in modality_to_uselist) or ('preops_l' in modality_to_uselist) or ('cbow' in modality_to_uselist):
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
        preops = preops.drop(columns=to_drop_old_pmh_with_others)  # "['pre_aki_status', 'preop_ICU', 'AnestStop']

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

        # currently sacrificing 5 data points in the valid set and using the test set to finally compute the auroc etc
        preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = preprocess_train(
            preops,
            outcome,
            y_outcome=
            outcome_df[
                "outcome"].values,
            binary_outcome=binary_outcome,
            valid_size=0.00005, random_state=randomSeed, input_dr=data_dir,
            output_dr=out_dir)  # change back to 0.00005

        if outcome == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set (value of plannedDispo are numeric after processing; the df has also been changed )
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 3]['plannedDispo'].index
            preops_te = preops_te.iloc[test_index]

    train_y = outcome_df.iloc[train_index]["outcome"]
    test_y = outcome_df.iloc[test_index]["outcome"]

    if train_y.dtype == 'O' or train_y.dtype == 'bool':
        train_y = train_y.replace([True, False], [1, 0])
        test_y = test_y.replace([True, False], [1, 0])

    # removing nans;this is mainly needed for pain and glucose and bp outcomes
    nan_idx_train = np.argwhere(np.isnan(train_y.values))
    nan_idx_test = np.argwhere(np.isnan(test_y.values))

    if nan_idx_train.size != 0:
        train_y = np.delete(train_y.values, nan_idx_train, axis=0)

    if nan_idx_test.size != 0:
        test_y = np.delete(test_y.values, nan_idx_test, axis=0)

    if binary_outcome:
        labels = np.unique(train_y)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)
    else:
        train_y = np.array(train_y)
        test_y = np.array(test_y)

    output_to_return_train = {}
    output_to_return_test = {}

    if 'cbow' in modality_to_uselist:
        cbow_train = bow_input.iloc[train_index].to_numpy()
        cbow_test = bow_input.iloc[test_index].to_numpy()

        scaler_cbow = StandardScaler()
        scaler_cbow.fit(cbow_train)
        train_X_cbow = scaler_cbow.transform(cbow_train)
        test_X_cbow = scaler_cbow.transform(cbow_test)

        if nan_idx_train.size != 0:
            train_X_cbow = np.delete(train_X_cbow, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_cbow = np.delete(test_X_cbow, nan_idx_test, axis=0)

        output_to_return_train['cbow'] = train_X_cbow
        output_to_return_test['cbow'] = test_X_cbow

        del train_X_cbow, test_X_cbow

    if ('preops_o' in modality_to_uselist) or ('preops_l' in modality_to_uselist):
        # read the metadata file, seperate out the indices of labs and encoded labs from it and then use them to seperate in the laoded processed preops files above
        md_filename = out_dir + 'preops_metadata_' + str(outcome) + "_" + datetime.now().strftime("%y-%m-%d") + '.json'
        if os.path.exists(md_filename):
            md_f1 = open(md_filename)
        else:
            print(" Need a valid meta data file name")
            exit()
        metadata_icu = json.load(md_f1)
        all_column_names = metadata_icu['column_all_names']
        all_column_names.remove('person_integer')

        # lab names
        f =open(data_dir + 'mapping_info/used_labs.txt')
        preoplabnames = f.read()
        f.close()
        preoplabnames_f = preoplabnames.split('\n')[:-1]

        # labs_to_sep = [i for i in all_column_names: if i in preoplabnames_f elif i.split("_")[:-1] in preoplabnames_f]
        labs_to_sep = []
        for i in all_column_names:
            if i in preoplabnames_f:
                labs_to_sep.append(i)
            else:
                try:
                    if i.split("_")[:-1][0] in preoplabnames_f:
                        labs_to_sep.append(i)
                except IndexError:
                    pass

        lab_indices_Sep = [all_column_names.index(i) for i in labs_to_sep]
        # preop_indices = [i for i in range(len(all_column_names)) if i not in lab_indices_Sep]

        # this is when the pmh and problist modalities are being used
        if 'pmh' in modality_to_uselist or 'problist' in modality_to_uselist:
            # dropping the pmh and problist columns from the preop list
            to_drop_old_pmh_problist = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                        "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                        "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose",
                                        'pre_aki_status', 'preop_ICU', 'preop_los']

            preop_indices = [all_column_names.index(i) for i in all_column_names if i not in (labs_to_sep + to_drop_old_pmh_problist)]
        else:
            preop_indices = [all_column_names.index(i) for i in all_column_names if i not in (labs_to_sep)]

        preops_train_true = preops_tr.values[:, preop_indices]
        preops_test_true = preops_te.values[:, preop_indices]

        preops_train_labs = preops_tr.values[:, lab_indices_Sep]
        preops_test_labs = preops_te.values[:, lab_indices_Sep]

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_true)
        train_X_pr_o = scaler.transform(preops_train_true)  # o here means only the non labs
        test_X_pr_o = scaler.transform(preops_test_true)

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_labs)
        train_X_pr_l = scaler.transform(preops_train_labs)  # l here means preop labs
        test_X_pr_l = scaler.transform(preops_test_labs)

        if nan_idx_train.size != 0:
            train_X_pr_o = np.delete(train_X_pr_o, nan_idx_train, axis=0)
            train_X_pr_l = np.delete(train_X_pr_l, nan_idx_train, axis=0)

        if nan_idx_test.size != 0:
            test_X_pr_o = np.delete(test_X_pr_o, nan_idx_test, axis=0)
            test_X_pr_l = np.delete(test_X_pr_l, nan_idx_test, axis=0)

        output_to_return_train['preops_o'] = train_X_pr_o
        output_to_return_test['preops_o'] = test_X_pr_o

        output_to_return_train['preops_l'] = train_X_pr_l
        output_to_return_test['preops_l'] = test_X_pr_l

        del train_X_pr_o, train_X_pr_l, test_X_pr_o, test_X_pr_l

    if 'homemeds' in modality_to_uselist:
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
        home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left',
                                                                           on='rxcui')
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


        home_meds_sum = home_meds_sum.drop(["rxcui"], axis=1)
        home_meds_final = home_meds_sum

        hm_tr = home_meds_final.iloc[train_index].to_numpy()
        hm_te = home_meds_final.iloc[test_index].to_numpy()

        # scaling only the embeded homemed version
        scaler_hm = StandardScaler()
        scaler_hm.fit(hm_tr)
        train_X_hm = scaler_hm.transform(hm_tr)
        test_X_hm = scaler_hm.transform(hm_te)

        if nan_idx_train.size != 0:
            train_X_hm = np.delete(train_X_hm, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_hm = np.delete(test_X_hm, nan_idx_test, axis=0)

        output_to_return_train['homemeds'] = train_X_hm
        output_to_return_test['homemeds'] = test_X_hm

        del train_X_hm, test_X_hm

    if 'pmh' in modality_to_uselist:

        pmh_emb_sb0 = pd.read_csv(data_dir + 'mv_data/pmh_sherbert_mv.csv')
        pmh_emb_sb1 = pd.read_csv(data_dir + 'pmh_sherbert.csv')

        pmh_emb_sb = pd.concat([pmh_emb_sb0, pmh_emb_sb1], axis=0)
        pmh_emb_sb['orlogid_encoded'] = pmh_emb_sb['orlogid_encoded'].astype('str')

        pmh_emb_sb.drop(columns=['ICD_10_CODES'],
                        inplace=True)  # although the next groupby sum is capable of removing this column, explicit removal is better
        pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                  fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)

        pmh_tr = pmh_emb_sb_final.iloc[train_index].to_numpy()
        pmh_te = pmh_emb_sb_final.iloc[test_index].to_numpy()

        # scaling the pmh
        scaler_pmh = StandardScaler()
        scaler_pmh.fit(pmh_tr)
        train_X_pmh = scaler_pmh.transform(pmh_tr)
        test_X_pmh = scaler_pmh.transform(pmh_te)

        if nan_idx_train.size != 0:
            train_X_pmh = np.delete(train_X_pmh, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_pmh = np.delete(test_X_pmh, nan_idx_test, axis=0)

        output_to_return_train['pmh'] = train_X_pmh
        output_to_return_test['pmh'] = test_X_pmh

        del train_X_pmh, test_X_pmh

    if 'flow' in modality_to_uselist:
        # flowsheet data
        very_dense_flow0 = feather.read_feather(data_dir + "mv_data/flow_ts/Imputed_very_dense_flow0.feather")
        very_dense_flow0.drop(very_dense_flow0[very_dense_flow0['timepoint'] > 511].index, inplace=True)
        very_dense_flow0 = very_dense_flow0.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                  on="orlogid_encoded")
        very_dense_flow0 = very_dense_flow0.loc[very_dense_flow0['endtime'] > very_dense_flow0['timepoint']]
        very_dense_flow0.drop(["endtime"], axis=1, inplace=True)

        other_intra_flow_wlabs0 = feather.read_feather(data_dir + "mv_data/flow_ts/Imputed_other_flow0.feather")
        other_intra_flow_wlabs0.drop(other_intra_flow_wlabs0[other_intra_flow_wlabs0['timepoint'] > 511].index,
                                     inplace=True)
        other_intra_flow_wlabs0 = other_intra_flow_wlabs0.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                                on="orlogid_encoded")
        other_intra_flow_wlabs0 = other_intra_flow_wlabs0.loc[
            other_intra_flow_wlabs0['endtime'] > other_intra_flow_wlabs0['timepoint']]
        other_intra_flow_wlabs0.drop(["endtime"], axis=1, inplace=True)

        very_dense_flow1 = feather.read_feather(data_dir + "flow_ts/Imputed_very_dense_flow.feather")
        very_dense_flow1.drop(very_dense_flow1[very_dense_flow1['timepoint'] > 511].index, inplace=True)
        very_dense_flow1['orlogid_encoded'] = very_dense_flow1['orlogid_encoded'].astype('str')
        very_dense_flow1 = very_dense_flow1.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                  on="orlogid_encoded")
        very_dense_flow1 = very_dense_flow1.loc[very_dense_flow1['endtime'] > very_dense_flow1['timepoint']]
        very_dense_flow1.drop(["endtime"], axis=1, inplace=True)

        other_intra_flow_wlabs1 = feather.read_feather(data_dir + "flow_ts/Imputed_other_flow.feather")
        other_intra_flow_wlabs1.drop(other_intra_flow_wlabs1[other_intra_flow_wlabs1['timepoint'] > 511].index,
                                     inplace=True)
        other_intra_flow_wlabs1['orlogid_encoded'] = other_intra_flow_wlabs1['orlogid_encoded'].astype('str')
        other_intra_flow_wlabs1 = other_intra_flow_wlabs1.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                                on="orlogid_encoded")
        other_intra_flow_wlabs1 = other_intra_flow_wlabs1.loc[
            other_intra_flow_wlabs1['endtime'] > other_intra_flow_wlabs1['timepoint']]
        other_intra_flow_wlabs1.drop(["endtime"], axis=1, inplace=True)

        very_dense_flow_comb = pd.concat([very_dense_flow0, very_dense_flow1], axis=0)
        other_intra_flow_wlabs_comb = pd.concat([other_intra_flow_wlabs0, other_intra_flow_wlabs1], axis=0)

        # merging on the right might have been better here because there were some cases that didn't have flowsheet data in the epic era. But that would introduce nans in the measure index column. However, the processing after this makes that redundant.
        very_dense_flow = very_dense_flow_comb.merge(new_index, on="orlogid_encoded", how="inner").drop(
            ["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)
        other_intra_flow_wlabs = other_intra_flow_wlabs_comb.merge(new_index, on="orlogid_encoded", how="inner").drop(
            ["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)

        """ TS flowsheet proprocessing """
        # need to convert the type of orlogid_encoded from object to int
        other_intra_flow_wlabs['person_integer'] = other_intra_flow_wlabs['person_integer'].astype('int')
        very_dense_flow['person_integer'] = very_dense_flow['person_integer'].astype('int')

        index_med_other_flow = torch.tensor(
            other_intra_flow_wlabs[['person_integer', 'timepoint', 'measure_index']].values, dtype=int)
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

        train_Xflow = np.concatenate((flowsheet_very_dense[train_index, :, :], torch.cumsum(torch.index_select(flowsheet_other_flow, 0, torch.tensor(train_index)).coalesce().to_dense(),axis=1)), axis=2)
        test_Xflow = np.concatenate((flowsheet_very_dense[test_index, :, :], torch.cumsum(torch.index_select(flowsheet_other_flow, 0, torch.tensor(test_index)).coalesce().to_dense(),axis=1)), axis=2)

        scaler = StandardScaler()
        scaler.fit(train_Xflow.reshape(-1, train_Xflow.shape[-1]))
        train_Xflow = scaler.transform(train_Xflow.reshape(-1, train_Xflow.shape[-1])).reshape(train_Xflow.shape)
        test_Xflow = scaler.transform(test_Xflow.reshape(-1, test_Xflow.shape[-1])).reshape(test_Xflow.shape)

        if nan_idx_train.size != 0:
            train_Xflow = np.delete(train_Xflow, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_Xflow = np.delete(test_Xflow, nan_idx_test, axis=0)

        output_to_return_train['flow'] = train_Xflow
        output_to_return_test['flow'] = test_Xflow

        del train_Xflow, test_Xflow

        breakpoint()
    if 'meds' in modality_to_uselist:
        # reading the med files
        all_med_data0 = feather.read_feather(data_dir + 'mv_data/med_ts/intraop_meds_filterd_wave0.feather')
        all_med_data0.drop(all_med_data0[all_med_data0['time'] > 511].index, inplace=True)
        all_med_data0.drop(all_med_data0[all_med_data0['time'] < 0].index,
                           inplace=True)  # there are some negative time points  ## TODO: i think it had some meaning; check this
        all_med_data0 = all_med_data0.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
        all_med_data0 = all_med_data0.loc[all_med_data0['endtime'] > all_med_data0['time']]
        all_med_data0.drop(["endtime"], axis=1, inplace=True)

        all_med_data1 = feather.read_feather(data_dir + 'med_ts/intraop_meds_filterd.feather')
        all_med_data1.drop(all_med_data1[all_med_data1['time'] > 511].index, inplace=True)
        all_med_data1.drop(all_med_data1[all_med_data1['time'] < 0].index,
                           inplace=True)  # there are some negative time points  ## TODO: i think it had some meaning; check this
        all_med_data1['orlogid_encoded'] = all_med_data1['orlogid_encoded'].astype('str')
        all_med_data1 = all_med_data1.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
        all_med_data1 = all_med_data1.loc[all_med_data1['endtime'] > all_med_data1['time']]
        all_med_data1.drop(["endtime"], axis=1, inplace=True)

        all_med_data = pd.concat([all_med_data0, all_med_data1], axis=0)

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
                                                                    1)).nonzero().squeeze() + 1  # this one is needed becasue otherwise there was some incompatibbility while the embedding for the combination are being created.
        group_end = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']),
                                                                  -1)).nonzero().squeeze() + 1  # this one is needed becasue otherwise there was some incompatibbility while the embedding for the combination are being created.

        group_start = torch.cat((torch.tensor(0).reshape((1)),
                                 group_start))  # prepending 0 to make sure that it is treated as an empty slot
        group_end = torch.cat(
            (torch.tensor(0).reshape((1)), group_end))  # prepending 0 to make sure that it is treated as an empty slot

        drug_med_ids = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_integer']]

        drug_med_id_map = pd.read_csv(data_dir + 'mv_data/med_ts/mv_drug_names.csv') ## since there are more medications in the MV era, I am using the the med mapping from mv era
        drug_words = None
        word_id_map = None

        drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_unit_comb',
                                  'dose']]  # replacing the unit_integer column by med_unit_comb column

        vocab_len_units = len(med_unit_unique_codes)  # replacing  len(unit_id_map) by len(med_unit_unique_codes)


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

        value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
        add_med = 0 in value_med_ids.unique()
        dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0, 1), value_med_ids + add_med, dtype=torch.int32)


        train_X_med = torch.cat([torch.index_select(dense_med_ids, 0, torch.tensor(train_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_dose, 0, torch.tensor(train_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_units, 0, torch.tensor(train_index)).coalesce().to_dense()],dim=2).cpu().numpy()
        test_X_med = torch.cat([torch.index_select(dense_med_ids, 0, torch.tensor(test_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_dose, 0, torch.tensor(test_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_units, 0, torch.tensor(test_index)).coalesce().to_dense()], dim=2).cpu().numpy()
        if nan_idx_train.size != 0:
            train_X_med = np.delete(train_X_med, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_med = np.delete(test_X_med, nan_idx_test, axis=0)

        output_to_return_train['meds'] = train_X_med
        output_to_return_test['meds'] = test_X_med

        del train_X_med, test_X_med

    if 'postopcomp' in modality_to_uselist:
        # also dropping the unit column from the outcomesn
        outcomes = outcomes.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
            list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
            ["orlogid_encoded", 'unit'], axis=1).rename({"new_person": "person_integer"}, axis=1).sort_values(
            ["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
        outcomes_train = outcomes.iloc[train_index]
        outcomes_test = outcomes.iloc[test_index]

        # admit_day outcome is 'the days after surgery of the ICU admission'
        for col in outcomes_train.columns:
            if outcomes_train[col].isna().any() == True:
                outcomes_train[col + "_mask"] = outcomes_train[col].notnull().astype('int')
                outcomes_test[col + "_mask"] = outcomes_test[col].notnull().astype('int')
                outcomes_train[col].fillna(0, inplace=True)
                outcomes_test[col].fillna(0, inplace=True)
            if (outcomes_train[col].dtype == bool) or (outcomes_train[col].dtype == 'O'):
                outcomes_train.loc[outcomes_train[col] == True, col] = 1
                outcomes_train.loc[outcomes_train[col] == False, col] = 0
                outcomes_test.loc[outcomes_test[col] == True, col] = 1
                outcomes_test.loc[outcomes_test[col] == False, col] = 0
                outcomes_train[col] = outcomes_train[col].astype(int)
                outcomes_test[col] = outcomes_test[col].astype(int)

        cont_outcomes = [i for i in outcomes_train.columns if outcomes_train[i].dtype == 'float64']
        binary_outcomes = [i for i in outcomes_train.columns if outcomes_train[i].dtype == 'int64']

        outcomes_train = outcomes_train.reindex(columns=cont_outcomes + binary_outcomes)
        outcomes_test = outcomes_test.reindex(columns=cont_outcomes + binary_outcomes)

        # Note: don't care about outcomes test being as the same size as that of others in icu because we are not going to have access to outcomes during the classifier training  so not really helpful

        if nan_idx_train.size != 0:
            outcomes_train = outcomes_train.drop(nan_idx_train[:, 0])
        if nan_idx_test.size != 0:
            outcomes_test = outcomes_test.reset_index().drop(nan_idx_test[:, 0]).drop(columns=['index'])

        output_to_return_train['postopcomp'] = outcomes_train
        output_to_return_test['postopcomp'] = outcomes_test

        del outcomes_train, outcomes_test
    return output_to_return_train, output_to_return_test, train_y, test_y, outcome_df, [train_index, test_index]


def load_epic(outcome, modality_to_uselist, randomSeed, data_dir, out_dir):  #dataset is whether it is flowsheets or meds, outcome is the postoperative outcome, list has the name of all modalities that will be used

    preops = pd.read_csv(data_dir + 'epic_preop.csv')
    outcomes = pd.read_csv(data_dir + 'epic_outcomes.csv')
    end_of_case_times = outcomes[['orlogid_encoded', 'endtime']]

    # end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
    regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood',
                               'postop_Vent_duration', 'n_glu_high',
                               'low_sbp_time', 'aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                               'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                               'opioids_count_day0', 'opioids_count_day1']
    binary_outcome = outcome not in regression_outcome_list

    outcomes = outcomes.dropna(subset=['ICU'])
    outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

    # exclude very short cases (this also excludes some invalid negative times)
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

    end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'], 511)


    binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac',
                           'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']

    if outcome in regression_outcome_list:
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

        outcome_df = outcomes[['orlogid_encoded', outcome]]
    elif outcome in binary_outcome_list:
        if outcome == 'VTE':
            temp_outcome = outcomes[['orlogid_encoded']]
            temp_outcome[outcome] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
            temp_outcome.loc[temp_outcome[outcome] == 2, outcome] = 1
        elif outcome == 'n_glucose_low':
            temp_outcome = outcomes[['orlogid_encoded', outcome]]
            temp_outcome[outcome] = temp_outcome[outcome].fillna(0)
            temp_outcome[outcome] = np.where(temp_outcome[outcome] > 0, 1, 0)
        else:
            temp_outcome = outcomes[['orlogid_encoded', outcome]]
            temp_outcome.loc[temp_outcome[outcome] == True, outcome] = 1
            temp_outcome.loc[temp_outcome[outcome] == False, outcome] = 0
        temp_outcome[outcome] = temp_outcome[outcome].astype(int)
        outcome_df = temp_outcome
    elif (outcome == 'dvt_pe'):
        dvt_pe_outcome = outcomes[['orlogid_encoded', 'DVT_PE']]
        outcome_df = dvt_pe_outcome
    elif (outcome == 'icu'):
        icu_outcome = outcomes[['orlogid_encoded', 'ICU']]
        icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
        icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
        icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)
        outcome_df = icu_outcome
    elif (outcome == 'mortality'):
        mortality_outcome = outcomes[['orlogid_encoded', 'death_in_30']]
        mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
        mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
        mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)
        outcome_df = mortality_outcome
    elif (outcome == 'aki1' or outcome == 'aki2' or outcome == 'aki3'):
        aki_outcome = outcomes[['orlogid_encoded', 'post_aki_status']]
        aki_outcome = aki_outcome.dropna(subset=[
            'post_aki_status'])  # this is droping the patients with baseline kidney failure as they are now post_aki_status = NA_integer_
        if outcome == 'aki1':
            aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
            aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
        if outcome == 'aki2':
            aki_outcome.loc[aki_outcome[
                                'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
            aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
        if outcome == 'aki3':
            aki_outcome.loc[aki_outcome[
                                'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will become zero :(; there is a one liner too that can be used
            aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
        aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)
        outcome_df = aki_outcome
    elif (outcome == 'endofcase'):
        outcome_df = end_of_case_times[['orlogid_encoded', 'true_test']]
    else:
        raise Exception("outcome not handled")

    ## intersect 3 mandatory data sources: preop, outcome, case end times
    combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(
        set(end_of_case_times['orlogid_encoded'].values)).intersection(
        set(preops['orlogid_encoded'].values)))
    if False:
        combined_case_set = combined_case_set[:1000]
        # combined_case_set = np.random.choice(combined_case_set, 10000, replace=False)

    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
    preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
    end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

    outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

    # checking for NA and other filters
    outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
    outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
    new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"},
                                                                          axis=1)  # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

    preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)


    if 'preops_o' not in modality_to_uselist:
        test_size = 0.2
        valid_size = 0.00005  # change back to 0.00005 for the full dataset
        y_outcome = outcome_df["outcome"].values
        preops.reset_index(drop=True, inplace=True)
        upto_test_idx = int(test_size * len(preops))
        test = preops.iloc[:upto_test_idx]
        train0 = preops.iloc[upto_test_idx:]
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=randomSeed,
                                            stratify=y_outcome[train0.index])
        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=randomSeed)

        train_index = train.index
        valid_index = valid.index
        test_index = test.index

        if outcome == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 'ICU']['plannedDispo'].index


    if ('preops_o' in modality_to_uselist) or ('preops_l' in modality_to_uselist) or ('cbow' in modality_to_uselist):
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
        preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = preprocess_train(
            preops,
            outcome,
            y_outcome=
            outcome_df[
                "outcome"].values,
            binary_outcome=binary_outcome,
            valid_size=0.00005, random_state=randomSeed, input_dr=data_dir,
            output_dr=out_dir)  # change back to 0.00005

        if outcome == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set (value of plannedDispo are numeric after processing; the df has also been changed )
            test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 3]['plannedDispo'].index
            preops_te = preops_te.iloc[test_index]


    train_y = outcome_df.iloc[train_index]["outcome"]
    test_y = outcome_df.iloc[test_index]["outcome"]

    if train_y.dtype == 'O' or train_y.dtype == 'bool':
        train_y = train_y.replace([True, False], [1, 0])
        test_y = test_y.replace([True, False], [1, 0])

    # removing nans;this is mainly needed for pain and glucose and bp outcomes
    nan_idx_train = np.argwhere(np.isnan(train_y.values))
    nan_idx_test = np.argwhere(np.isnan(test_y.values))

    if nan_idx_train.size != 0:
        train_y = np.delete(train_y.values, nan_idx_train, axis=0)

    if nan_idx_test.size != 0:
        test_y = np.delete(test_y.values, nan_idx_test, axis=0)

    if binary_outcome:
        labels = np.unique(train_y)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)
    else:
        train_y = np.array(train_y)
        test_y = np.array(test_y)

    output_to_return_train = {}
    output_to_return_test = {}

    if 'cbow' in modality_to_uselist:
        cbow_train = bow_input.iloc[train_index].to_numpy()
        cbow_test = bow_input.iloc[test_index].to_numpy()

        scaler_cbow = StandardScaler()
        scaler_cbow.fit(cbow_train)
        train_X_cbow = scaler_cbow.transform(cbow_train)
        test_X_cbow = scaler_cbow.transform(cbow_test)

        if nan_idx_train.size != 0:
            train_X_cbow = np.delete(train_X_cbow, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_cbow = np.delete(test_X_cbow, nan_idx_test, axis=0)

        output_to_return_train['cbow'] = train_X_cbow
        output_to_return_test['cbow'] = test_X_cbow

        del train_X_cbow, test_X_cbow

    if ('preops_o' in modality_to_uselist) or ('preops_l' in modality_to_uselist):
        # read the metadata file, seperate out the indices of labs and encoded labs from it and then use them to seperate in the laoded processed preops files above
        md_filename = out_dir + 'preops_metadata_' + str(outcome) + "_" + datetime.now().strftime("%y-%m-%d") + '.json'
        if os.path.exists(md_filename):
            md_f1 = open(md_filename)
        else:
            print(" Need a valid meta data file name")
            exit()
        metadata_icu = json.load(md_f1)
        all_column_names = metadata_icu['column_all_names']
        all_column_names.remove('person_integer')

        # lab names
        f =open(data_dir + 'mapping_info/used_labs.txt')
        preoplabnames = f.read()
        f.close()
        preoplabnames_f = preoplabnames.split('\n')[:-1]

        # labs_to_sep = [i for i in all_column_names: if i in preoplabnames_f elif i.split("_")[:-1] in preoplabnames_f]
        labs_to_sep = []
        for i in all_column_names:
            if i in preoplabnames_f:
                labs_to_sep.append(i)
            else:
                try:
                    if i.split("_")[:-1][0] in preoplabnames_f:
                        labs_to_sep.append(i)
                except IndexError:
                    pass

        lab_indices_Sep = [all_column_names.index(i) for i in labs_to_sep]
        # preop_indices = [i for i in range(len(all_column_names)) if i not in lab_indices_Sep]

        # this is when the pmh and problist modalities are being used
        if 'pmh' in modality_to_uselist or 'problist' in modality_to_uselist:
            # dropping the pmh and problist columns from the preop list
            to_drop_old_pmh_problist = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                        "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                        "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose",
                                        'pre_aki_status', 'preop_ICU', 'preop_los']

            preop_indices = [all_column_names.index(i) for i in all_column_names if i not in (labs_to_sep + to_drop_old_pmh_problist)]
        else:
            preop_indices = [all_column_names.index(i) for i in all_column_names if i not in (labs_to_sep)]

        preops_train_true = preops_tr.values[:, preop_indices]
        preops_test_true = preops_te.values[:, preop_indices]

        preops_train_labs = preops_tr.values[:, lab_indices_Sep]
        preops_test_labs = preops_te.values[:, lab_indices_Sep]

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_true)
        train_X_pr_o = scaler.transform(preops_train_true)  # o here means only the non labs
        test_X_pr_o = scaler.transform(preops_test_true)

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_labs)
        train_X_pr_l = scaler.transform(preops_train_labs)  # l here means preop labs
        test_X_pr_l = scaler.transform(preops_test_labs)

        if nan_idx_train.size != 0:
            train_X_pr_o = np.delete(train_X_pr_o, nan_idx_train, axis=0)
            train_X_pr_l = np.delete(train_X_pr_l, nan_idx_train, axis=0)

        if nan_idx_test.size != 0:
            test_X_pr_o = np.delete(test_X_pr_o, nan_idx_test, axis=0)
            test_X_pr_l = np.delete(test_X_pr_l, nan_idx_test, axis=0)

        output_to_return_train['preops_o'] = train_X_pr_o
        output_to_return_test['preops_o'] = test_X_pr_o

        output_to_return_train['preops_l'] = train_X_pr_l
        output_to_return_test['preops_l'] = test_X_pr_l

        del train_X_pr_o, train_X_pr_l, test_X_pr_o, test_X_pr_l

    if 'homemeds' in modality_to_uselist:
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

        hm_tr = home_meds_final.iloc[train_index].to_numpy()
        hm_te = home_meds_final.iloc[test_index].to_numpy()

        # scaling only the embeded homemed version
        scaler_hm = StandardScaler()
        scaler_hm.fit(hm_tr)
        train_X_hm = scaler_hm.transform(hm_tr)
        test_X_hm = scaler_hm.transform(hm_te)

        if nan_idx_train.size != 0:
            train_X_hm = np.delete(train_X_hm, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_hm = np.delete(test_X_hm, nan_idx_test, axis=0)

        output_to_return_train['homemeds'] = train_X_hm
        output_to_return_test['homemeds'] = test_X_hm

        del train_X_hm, test_X_hm

    if 'pmh' in modality_to_uselist:

        pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert.csv')

        pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                  fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)

        pmh_tr = pmh_emb_sb_final.iloc[train_index].to_numpy()
        pmh_te = pmh_emb_sb_final.iloc[test_index].to_numpy()


        # scaling the pmh
        scaler_pmh = StandardScaler()
        scaler_pmh.fit(pmh_tr)
        train_X_pmh = scaler_pmh.transform(pmh_tr)
        test_X_pmh = scaler_pmh.transform(pmh_te)

        if nan_idx_train.size != 0:
            train_X_pmh = np.delete(train_X_pmh, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_pmh = np.delete(test_X_pmh, nan_idx_test, axis=0)

        output_to_return_train['pmh'] = train_X_pmh
        output_to_return_test['pmh'] = test_X_pmh

        del train_X_pmh, test_X_pmh

    if 'problist' in modality_to_uselist:
        prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')

        prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        prob_list_emb_sb_final = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index(
            'new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)),
                                  fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
            ["person_integer"], axis=1)

        problist_tr = prob_list_emb_sb_final.iloc[train_index].to_numpy()
        problist_te = prob_list_emb_sb_final.iloc[test_index].to_numpy()

        # scaling the prob_list
        scaler_problist = StandardScaler()
        scaler_problist.fit(problist_tr)
        train_X_problist = scaler_problist.transform(problist_tr)
        test_X_problist = scaler_problist.transform(problist_te)

        if nan_idx_train.size != 0:
            train_X_problist = np.delete(train_X_problist, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_problist = np.delete(test_X_problist, nan_idx_test, axis=0)

        output_to_return_train['problist'] = train_X_problist
        output_to_return_test['problist'] = test_X_problist

        del train_X_problist, test_X_problist

    if 'flow' in modality_to_uselist:
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

        train_Xflow = np.concatenate((flowsheet_very_dense[train_index, :, :], torch.cumsum(torch.index_select(flowsheet_other_flow, 0, torch.tensor(train_index)).coalesce().to_dense(),axis=1)), axis=2)
        test_Xflow = np.concatenate((flowsheet_very_dense[test_index, :, :], torch.cumsum(torch.index_select(flowsheet_other_flow, 0, torch.tensor(test_index)).coalesce().to_dense(),axis=1)), axis=2)

        scaler = StandardScaler()
        scaler.fit(train_Xflow.reshape(-1, train_Xflow.shape[-1]))
        train_Xflow = scaler.transform(train_Xflow.reshape(-1, train_Xflow.shape[-1])).reshape(train_Xflow.shape)
        test_Xflow = scaler.transform(test_Xflow.reshape(-1, test_Xflow.shape[-1])).reshape(test_Xflow.shape)

        if nan_idx_train.size != 0:
            train_Xflow = np.delete(train_Xflow, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_Xflow = np.delete(test_Xflow, nan_idx_test, axis=0)

        output_to_return_train['flow'] = train_Xflow
        output_to_return_test['flow'] = test_Xflow

        del train_Xflow, test_Xflow

    if 'meds' in modality_to_uselist:
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

        drug_med_id_map = feather.read_feather(data_dir + 'med_ts/med_id_map.feather')
        drug_words = None
        word_id_map = None

        # drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'unit_integer',
        #                           'dose']]
        drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_unit_comb',
                                  'dose']]  # replacing the unit_integer column by med_unit_comb column

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

        value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
        add_med = 0 in value_med_ids.unique()
        dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0, 1), value_med_ids + add_med,
                                                dtype=torch.int32)


        train_X_med = torch.cat([torch.index_select(dense_med_ids, 0, torch.tensor(train_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_dose, 0, torch.tensor(train_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_units, 0, torch.tensor(train_index)).coalesce().to_dense()],dim=2).cpu().numpy()
        test_X_med = torch.cat([torch.index_select(dense_med_ids, 0, torch.tensor(test_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_dose, 0, torch.tensor(test_index)).coalesce().to_dense(),
                           torch.index_select(dense_med_units, 0, torch.tensor(test_index)).coalesce().to_dense()], dim=2).cpu().numpy()
        if nan_idx_train.size != 0:
            train_X_med = np.delete(train_X_med, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_med = np.delete(test_X_med, nan_idx_test, axis=0)

        output_to_return_train['meds'] = train_X_med
        output_to_return_test['meds'] = test_X_med

        del train_X_med, test_X_med

    if 'alerts' in modality_to_uselist:
        """
        In wave 2, following alerts were not in the dictionary: {'antibioticVancom', 'vancomycin', 'antibioticRedose', 'tempHxMH', 'antibioticVancomycin', 'lungCompliance'}
        Out of these 6, the ones that were observed in a couple of hundred patients are 'lungCompliance', 'antibioticRedose', 'antibioticVancomycin'. So added these to the original dictionary.
        This action resolves the issue with wave 1 where 'lungCompliance' was not in the data but not in the dictionary.
        """

        alerts = pd.read_csv(data_dir +'epic_alerts.csv')
        if False:
            alerts = alerts.merge(new_index, on="orlogid_encoded", how="inner")
            alerts_outcome = alerts.merge(outcome_df, on="orlogid_encoded", how="inner")
            print(alerts_outcome.groupby(['id','contact'], dropna=False)['outcome'].agg(['mean','count']).head(50))
            print(alerts_outcome.groupby(['contact'], dropna=False)['outcome'].agg(['mean','count']).head(50))
            breakpoint()



        if False:
            # alerts_dict = pd.ExcelFile(data_dir+'AW_Alerts_Dictionary.xlsx')
            alerts_dict = pd.read_excel(data_dir+'AW_Alerts_Dictionary.xlsx')
            alerts_dict = alerts_dict[['atom','id']].dropna().reset_index(drop=True)
            alerts_dict['comb_id'] = np.arange(len(alerts_dict))+1
            alerts_dict.at[len(alerts_dict), 'comb_id'] = len(alerts_dict)+1  # this +1 is to account for the fact that
            alerts_dict_map = dict(zip(alerts_dict['id'], alerts_dict['comb_id']))

            output_file_name = out_dir + '/alertsCombID_map.json'
            with open(output_file_name, 'w') as outfile: json.dump(alerts_dict_map, outfile)

        alerts.drop(alerts[alerts['time'] > 511].index, inplace=True)
        alerts = alerts.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"},
            axis=1)  # not taking the left because its creating unnecessary nans that can be possibly handled by the sparse thing
        alerts_df = alerts.drop_duplicates()[['time', 'atom', 'overall_relevant', 'overall_interv', 'person_integer', 'id']]
        alerts_df = alerts_df.drop_duplicates()

        # adding another column to incorporate the fact that there could be more than one alerts at the same time
        alerts_df['alert_postn'] = alerts_df.groupby(['person_integer', 'time']).cumcount()

        # dropping the rows where the alert type was any of {'antibioticVancom', 'vancomycin', 'tempHxMH'} because of their low prevalence and not being in the dictionary. Also, because each alert is creating a column here.
        atomstoDrop = ['antibioticVancom', 'vancomycin', 'tempHxMH']
        alerts_df = alerts_df.drop(alerts_df[alerts_df.atom.isin(atomstoDrop)].index)

        # creating a combined id for the combination of alert type and the id
        alerts_df['comb_id'] = alerts_df['id']

        # loading an already saved json file
        output_file_name = data_dir + 'alertsCombID_map.json'
        with open(output_file_name) as outfile:  alerts_dict_map = json.load(outfile)

        alerts_df = alerts_df.replace({'comb_id': alerts_dict_map})


        alerts_df = alerts_df.drop(columns=['atom', 'id'])
        # need to change the encoding to account for the values which are not present (spike nature kind); in case of one hot encoded its fine because 0 represents not present
        alerts_df.loc[alerts_df['overall_relevant'] == True, 'overall_relevant'] = 1
        alerts_df.loc[alerts_df['overall_relevant'] == False, 'overall_relevant'] = -1
        alerts_df.loc[alerts_df['overall_interv'] == 0, 'overall_interv'] = -1

        alert_columns = [i for i in alerts_df if i not in ['person_integer', 'time', 'alert_postn']]
        for i in alert_columns:  alerts_df[i] = alerts_df[i].astype(int)

        index_alerts = torch.tensor(alerts_df[['person_integer', 'time', 'alert_postn']].values, dtype=int)
        value_alerts = torch.tensor(alerts_df[alert_columns].values, dtype=int)
        alerts_final_sparse = torch.sparse_coo_tensor(torch.transpose(index_alerts, 0, 1), value_alerts, size=[len(combined_case_set), 511, alerts_df.alert_postn.max() + 1, len(alert_columns)], dtype=torch.int64)

        if False:
            # this part is elaborate with each alert as a column. Down the line it would need an embedding layer for each
            # breakpoint()

            alerts_dict['id_map'] = alerts_dict[['id', 'atom']].groupby('atom').cumcount()
            alerts_dict = alerts_dict[['id', 'atom', 'id_map']]
            alerts_dict['nan_val'] = 0
            for i in range(len(alerts_dict)):
                if not (pd.isna(alerts_dict.iloc[i]['atom'])):
                    alerts_dict.at[i, 'nan_val'] = alerts_dict.groupby('atom')['id_map'].max()[alerts_dict.iloc[i][
                        'atom']] + 2  # assigning nan; .at is to assign single cell value for a row/column label pair
                else:
                    alerts_dict.at[i, 'nan_val'] = np.nan
            alerts_dict_map = dict(zip(alerts_dict['id'], alerts_dict['id_map']))

            unique_atoms = list(alerts_df.atom.unique())
            for atom_val in unique_atoms:
                alerts_df[atom_val] = alerts_df[alerts_df['atom']==atom_val]['id']
                alerts_df.replace({atom_val: alerts_dict_map}, inplace=True)
                alerts_df[atom_val].fillna(alerts_dict[alerts_dict['atom'] == atom_val]['nan_val'].values[0], inplace=True)
                alerts_dict_map[atom_val + str('.nan')] = alerts_dict[alerts_dict['atom'] == atom_val]['nan_val'].values[0]  # this is for later use to identify the map

            output_file_name = out_dir + 'alertsID_map_' + datetime.now().strftime("%y-%m-%d") + '.json'
            with open(output_file_name, 'w') as outfile:
                json.dump(alerts_dict_map, outfile)

            alerts_df = alerts_df.drop(columns=['atom','id'])
            # need to change the encoding to account for the values which are not present (spike nature kind); in case of one hot encoded its fine because 0 represents not present
            alerts_df.loc[alerts_df['overall_relevant'] == True, 'overall_relevant'] = 1
            alerts_df.loc[alerts_df['overall_relevant'] == False, 'overall_relevant'] = -1
            alerts_df.loc[alerts_df['overall_interv'] == 0, 'overall_interv'] = -1

            alert_columns = [i for i in alerts_df if i not in ['person_integer', 'time','alert_postn']]
            for i in alert_columns:  alerts_df[i] = alerts_df[i].astype(int)

            # breakpoint()

            index_alerts = torch.tensor(alerts_df[['person_integer', 'time', 'alert_postn']].values, dtype=int)
            value_alerts = torch.tensor(alerts_df[alert_columns].values, dtype=int)
            alerts_final_sparse = torch.sparse_coo_tensor(torch.transpose(index_alerts, 0, 1), value_alerts,size=[len(combined_case_set), 511, alerts_df.alert_postn.max() +1, len(alert_columns)], dtype=torch.int64)

        train_Xalert = torch.index_select(alerts_final_sparse, 0, torch.tensor(train_index)).to_dense().cpu().numpy() # this is
        test_Xalert = torch.index_select(alerts_final_sparse, 0, torch.tensor(test_index)).to_dense().cpu().numpy()

        if nan_idx_train.size != 0: train_Xalert = np.delete(train_Xalert, nan_idx_train, axis=0)
        if nan_idx_test.size != 0: test_Xalert = np.delete(test_Xalert, nan_idx_test, axis=0)

        output_to_return_train['alerts'] = train_Xalert
        output_to_return_test['alerts'] = test_Xalert
        del train_Xalert, test_Xalert

    if 'postopcomp' in modality_to_uselist:
        # also dropping the unit column from the outcomesn
        outcomes = outcomes.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded", 'unit'], axis=1).rename({"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
        outcomes_train = outcomes.iloc[train_index]
        outcomes_test = outcomes.iloc[test_index]


        # admit_day outcome is 'the days after surgery of the ICU admission'
        for col in outcomes_train.columns:
            if outcomes_train[col].isna().any() == True:
                outcomes_train[col + "_mask"] = outcomes_train[col].notnull().astype('int')
                outcomes_test[col + "_mask"] = outcomes_test[col].notnull().astype('int')
                outcomes_train[col].fillna(0, inplace=True)
                outcomes_test[col].fillna(0, inplace=True)
            if (outcomes_train[col].dtype == bool) or (outcomes_train[col].dtype == 'O'):
                outcomes_train.loc[outcomes_train[col] == True, col] = 1
                outcomes_train.loc[outcomes_train[col] == False, col] = 0
                outcomes_test.loc[outcomes_test[col] == True, col] = 1
                outcomes_test.loc[outcomes_test[col] == False, col] = 0
                outcomes_train[col] = outcomes_train[col].astype(int)
                outcomes_test[col] = outcomes_test[col].astype(int)

        cont_outcomes = [i for i in outcomes_train.columns if outcomes_train[i].dtype == 'float64']
        binary_outcomes = [i for i in outcomes_train.columns if outcomes_train[i].dtype == 'int64']

        outcomes_train = outcomes_train.reindex(columns=cont_outcomes + binary_outcomes)
        outcomes_test = outcomes_test.reindex(columns=cont_outcomes + binary_outcomes)

        # Note: don't care about outcomes test being as the same size as that of others in icu because we are not going to have access to outcomes during the classifier training  so not really helpful

        if nan_idx_train.size != 0:
            outcomes_train = outcomes_train.drop(nan_idx_train[:, 0])
        if nan_idx_test.size != 0:
            outcomes_test = outcomes_test.reset_index().drop(nan_idx_test[:, 0]).drop(columns=['index'])

        output_to_return_train['postopcomp'] = outcomes_train
        output_to_return_test['postopcomp'] = outcomes_test

        del outcomes_train, outcomes_test

    return output_to_return_train, output_to_return_test, train_y, test_y, outcome_df, [train_index,test_index]


