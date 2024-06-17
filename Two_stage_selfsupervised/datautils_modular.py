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
import torch.optim as optim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.profiler import profile, record_function, ProfilerActivity

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
    
def load_epic(dataset, outcome, modality_to_uselist, data_dir):  #dataset is whether it is flowsheets or meds, outcome is the postoperative outcome, list has the name of all modalities that will be used

    # to use for the modalities which were not taken in the partitioned form and getting the y values
    train_idx = pd.read_csv(data_dir + "train_test_id_orlogid_map.csv")
    all_outcomes = pd.read_csv(data_dir + "all_outcomes_with_orlogid.csv")
    train_id_withoutcomes = train_idx.merge(all_outcomes.drop(columns=['orlogid_encoded']), on=['new_person'], how='left')

    # processing the labels
    if outcome=='mortality':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['death_in_30']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['death_in_30']
    elif outcome=='PE':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['PE']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['PE']
    elif outcome=='pulm':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['pulm']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['pulm']
    elif outcome=='severe_present_1':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['severe_present_1']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['severe_present_1']
    elif outcome=='cardiac':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['cardiac']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['cardiac']
    elif outcome=='postop_del':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['postop_del']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['postop_del']
    elif outcome=='DVT':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['DVT']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['DVT']
    elif outcome in ['low_sbp_time', 'n_glu_high' ]:
        train_id_withoutcomes[outcome] = train_id_withoutcomes[outcome].fillna(0)
        train_id_withoutcomes[outcome] = np.where(train_id_withoutcomes[outcome]>0, 1, 0)
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1][outcome]
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0][outcome]
    elif outcome=='aki1':
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] >= 1, 'post_aki_status'] = 1
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] < 1, 'post_aki_status'] = 0
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status']
    elif outcome=='aki2':
        train_id_withoutcomes.loc[train_id_withoutcomes[
                            'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] >= 2, 'post_aki_status'] = 1
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status']
    elif outcome=='aki3':
        train_id_withoutcomes.loc[train_id_withoutcomes[
                            'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] == 3, 'post_aki_status'] = 1
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status']
    elif outcome=='worst_pain_1':
        train_id_withoutcomes.loc[train_id_withoutcomes['worst_pain_1'] < 7, 'worst_pain_1'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        train_id_withoutcomes.loc[train_id_withoutcomes['worst_pain_1'] >= 7, 'worst_pain_1'] = 1
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['worst_pain_1']
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['worst_pain_1']
    elif outcome == 'icu':
        preops_raw = pd.read_csv(data_dir + "Raw_preops_used_in_ICU.csv")
        test_index_orig = train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values

        # this approach is being used to be consistent with all other outcomes # crosschecked already
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['ICU']
        true_pi_test_int = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3]['person_integer'].values
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['ICU'].iloc[true_pi_test_int]

        test_index = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3]['plannedDispo'].index


        del preops_raw, true_pi_test_int

    if train_y.dtype == 'O' or train_y.dtype == 'bool':
        train_y = train_y.replace([True, False], [1, 0])
        test_y = test_y.replace([True, False], [1, 0])

    # removing nans;this is mainly needed for pain and glucose and bp outcomes
    nan_idx_train = np.argwhere(np.isnan(train_y.values))
    nan_idx_test = np.argwhere(np.isnan(test_y.values))

    train_idx_df = train_idx[train_idx['train_id_or_not']==1]
    if outcome == 'icu':  # evaluation only on the non preplanned ICU cases
       test_idx_df = train_idx[train_idx['new_person'].isin(list(test_index))]
    else:
       test_idx_df = train_idx[train_idx['train_id_or_not']==0]

    if nan_idx_train.size != 0:
        train_y = np.delete(train_y.values, nan_idx_train, axis=0)
        train_idx_df = train_idx_df.drop(nan_idx_train[:,0])
        train_idx_df['true_y'] = train_y

    if nan_idx_test.size != 0:
        test_y = np.delete(test_y.values, nan_idx_test, axis=0)
        test_idx_df = test_idx_df.reset_index().drop(nan_idx_test[:,0]).drop(columns = ['index'])
        test_idx_df['true_y'] = test_y

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    # creating modality dictionary
    output_to_return_train = {}
    output_to_return_test = {}
    if 'flow' in modality_to_uselist:
        dense_flowsheet_train = np.load(data_dir + "flowsheet_very_dense_proc_tr.npy")
        dense_flowsheet_test = np.load(data_dir + "flowsheet_very_dense_proc_te.npy")
        dense_flowsheet_valid = np.load(data_dir + "flowsheet_very_dense_proc_val.npy")

        other_flowsheet_train = np.load(data_dir + "other_flow_dense_proc_train.npy")
        other_flowsheet_test = np.load(data_dir + "other_flow_dense_proc_test.npy")
        other_flowsheet_valid = np.load(data_dir + "other_flow_dense_proc_valid.npy")

        train_Xflow = np.concatenate((dense_flowsheet_train, other_flowsheet_train), axis=2)
        test_Xflow = np.concatenate((dense_flowsheet_test, other_flowsheet_test), axis=2)

        if outcome == 'icu':
            test_Xflow = test_Xflow[test_index]

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
        dense_med_doses_train = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_train.npy"))
        dense_med_doses_test = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_test.npy"))
        dense_med_doses_valid = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_valid.npy"))

        dense_med_id_train = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_train.npy"))
        dense_med_id_test = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_test.npy"))
        dense_med_id_valid = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_valid.npy"))

        dense_med_units_train = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_train.npy"))
        dense_med_units_test = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_test.npy"))
        dense_med_units_valid = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_valid.npy"))

        # breakpoint()
        train_X_med = torch.cat([dense_med_id_train, dense_med_doses_train, dense_med_units_train],dim=2).cpu().numpy()
        test_X_med = torch.cat([dense_med_id_test, dense_med_doses_test, dense_med_units_test], dim=2).cpu().numpy()

        if outcome == 'icu':
            test_X_med = test_X_med[test_index]

        if nan_idx_train.size != 0:
            train_X_med = np.delete(train_X_med, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_med = np.delete(test_X_med, nan_idx_test, axis=0)

        output_to_return_train['meds'] = train_X_med
        output_to_return_test['meds'] = test_X_med

        del train_X_med, test_X_med

    if 'alerts' in modality_to_uselist:
        alerts = torch.from_numpy(np.load(
            data_dir + "alerts_single_final_sparse_tensor.npy"))  # train_idx[train_idx['train_id_or_not']==1]['new_person']
        alerts_train = alerts[train_idx[train_idx['train_id_or_not'] == 1]['new_person'].values, :, :]
        alerts_test = alerts[train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values, :, :]
        train_Xalert = alerts_train.cpu().numpy()
        test_Xalert = alerts_test.cpu().numpy()

        if outcome == 'icu':
            test_Xalert = test_Xalert[test_index]

        if nan_idx_train.size != 0:
            train_Xalert = np.delete(train_Xalert, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_Xalert = np.delete(test_Xalert, nan_idx_test, axis=0)

        output_to_return_train['alerts'] = train_Xalert
        output_to_return_test['alerts'] = test_Xalert

        del train_Xalert, test_Xalert

    if 'postopcomp' in modality_to_uselist:
        # since in the current setup the missingness is distributed equally between the train and test set so we can drop the non outcomes as follows and then add missingness masks for only those outcomes
        outcomes_train = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not'] == 1].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not', 'unit'])
        outcomes_test = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not'] == 0].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not', 'unit'])

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

    if 'homemeds' in modality_to_uselist:
        ## reading home meds
        home_meds_embedsum = pd.read_csv(data_dir+'Home_meds_embeddedSum_indexed.csv')
        home_meds_embedsum_tr_te = train_idx.merge(home_meds_embedsum.drop(columns= ['orlogid_encoded']), on=['new_person'], how='left')
        home_meds_embedsum_train = home_meds_embedsum_tr_te[home_meds_embedsum_tr_te['train_id_or_not']==1].drop(columns=['new_person','orlogid_encoded', 'train_id_or_not', 'rxcui'])
        home_meds_embedsum_test = home_meds_embedsum_tr_te[home_meds_embedsum_tr_te['train_id_or_not']==0].drop(columns=['new_person','orlogid_encoded', 'train_id_or_not', 'rxcui'])

        if outcome == 'icu':
            home_meds_embedsum_test = home_meds_embedsum_test.iloc[test_index]

        if False:
            home_meds_ohe = pd.read_csv(data_dir + 'Home_meds_ohe_indexed.csv')
            home_meds_ohe_tr_te = train_idx.merge(home_meds_ohe.drop(columns=['orlogid_encoded']), on=['new_person'],
                                                  how='left')
            home_meds_ohe_train = home_meds_ohe_tr_te[home_meds_ohe_tr_te['train_id_or_not']==1].drop(columns=['new_person','orlogid_encoded', 'train_id_or_not'])
            home_meds_ohe_test = home_meds_ohe_tr_te[home_meds_ohe_tr_te['train_id_or_not']==0].drop(columns=['new_person','orlogid_encoded', 'train_id_or_not'])

        # scaling only the embeded homemed version
        scaler_hm = StandardScaler()
        scaler_hm.fit(home_meds_embedsum_train)
        train_X_hm = scaler_hm.transform(home_meds_embedsum_train)
        test_X_hm = scaler_hm.transform(home_meds_embedsum_test)

        if nan_idx_train.size != 0:
            train_X_hm = np.delete(train_X_hm, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_hm = np.delete(test_X_hm, nan_idx_test, axis=0)

        output_to_return_train['homemeds'] = train_X_hm
        output_to_return_test['homemeds'] = test_X_hm

        del train_X_hm, test_X_hm

    if 'pmh' in modality_to_uselist:
        pmh_embeded = pd.read_csv(data_dir + 'pmh_emb_sb_indexed.csv')
        pmh_embeded_tr_te = train_idx.merge(pmh_embeded.drop(columns=['orlogid_encoded']), on=['new_person'],
                                            how='left')
        pmh_embeded_train = pmh_embeded_tr_te[pmh_embeded_tr_te['train_id_or_not'] == 1].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not'])
        pmh_embeded_test = pmh_embeded_tr_te[pmh_embeded_tr_te['train_id_or_not'] == 0].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not'])

        if outcome == 'icu':
            pmh_embeded_test = pmh_embeded_test.iloc[test_index]

        # scaling the pmh
        scaler_pmh = StandardScaler()
        scaler_pmh.fit(pmh_embeded_train)
        train_X_pmh = scaler_pmh.transform(pmh_embeded_train)
        test_X_pmh = scaler_pmh.transform(pmh_embeded_test)

        if nan_idx_train.size != 0:
            train_X_pmh = np.delete(train_X_pmh, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_pmh = np.delete(test_X_pmh, nan_idx_test, axis=0)

        output_to_return_train['pmh'] = train_X_pmh
        output_to_return_test['pmh'] = test_X_pmh

        del train_X_pmh, test_X_pmh

    if 'problist' in modality_to_uselist:
        prob_list_embeded = pd.read_csv(data_dir + 'prob_list_emb_sb_indexed.csv')
        prob_list_embeded_tr_te = train_idx.merge(prob_list_embeded.drop(columns=['orlogid_encoded']),
                                                  on=['new_person'], how='left')
        prob_list_embeded_train = prob_list_embeded_tr_te[prob_list_embeded_tr_te['train_id_or_not'] == 1].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not'])
        prob_list_embeded_test = prob_list_embeded_tr_te[prob_list_embeded_tr_te['train_id_or_not'] == 0].drop(
            columns=['new_person', 'orlogid_encoded', 'train_id_or_not'])

        if outcome == 'icu':
            prob_list_embeded_test = prob_list_embeded_test.iloc[test_index]

        # scaling the prob_list
        scaler_problist = StandardScaler()
        scaler_problist.fit(prob_list_embeded_train)
        train_X_problist = scaler_problist.transform(prob_list_embeded_train)
        test_X_problist = scaler_problist.transform(prob_list_embeded_test)

        if nan_idx_train.size != 0:
            train_X_problist = np.delete(train_X_problist, nan_idx_train, axis=0)
        if nan_idx_test.size != 0:
            test_X_problist = np.delete(test_X_problist, nan_idx_test, axis=0)

        output_to_return_train['problist'] = train_X_problist
        output_to_return_test['problist'] = test_X_problist

        del train_X_problist, test_X_problist

    if 'cbow' in modality_to_uselist:
        cbow_train = np.load(data_dir + "cbow_proc_train.npy")
        cbow_test = np.load(data_dir + "cbow_proc_test.npy")

        if outcome == 'icu':
            cbow_test = cbow_test[test_index]

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

        preops_train = np.load(data_dir + "preops_proc_train.npy")
        preops_test = np.load(data_dir + "preops_proc_test.npy")
        preops_valid = np.load(data_dir + "preops_proc_test.npy")

        if outcome=='icu':
            preops_test = preops_test[test_index]

        # read the metadata file, seperate out the indices of labs and encoded labs from it and then use them to seperate in the laoded processed preops files above
        md_f1 = open(data_dir + 'preops_metadataicu.json')
        metadata_icu = json.load(md_f1)
        all_column_names = metadata_icu['column_all_names']
        all_column_names.remove('person_integer')

        # lab names
        f =open(data_dir + 'used_labs.txt')
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


        preops_train_true = preops_train[:, preop_indices]
        preops_test_true = preops_test[:, preop_indices]

        preops_train_labs = preops_train[:, lab_indices_Sep]
        preops_test_labs = preops_test[:, lab_indices_Sep]

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

    # breakpoint()
    return output_to_return_train, output_to_return_test, train_y, test_y, train_idx_df, test_idx_df


