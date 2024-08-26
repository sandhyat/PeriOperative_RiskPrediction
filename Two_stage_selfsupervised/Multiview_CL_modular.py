import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import Adam
import numpy as np
from models import TSEncoder, TSEncoder_f, TSEncoder_m, TSEncoder_m_alt, TSEncoder_a
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tasks import scarf_model as preop_model
from tasks import SubTab_model as outcome_model
from tasks import loss
import gc

from tqdm.auto import tqdm

class customdataset(Dataset):
    def __init__(self, data, outcome=None, transform=None):
        """
        Characterizes a Dataset for PyTorch and returns the index too. Also returns another index from the same batch to be used with preops

        Parameters
        ----------
        data: multidimensional torch tensor
        outcome: output torch tensor
        """
        self.n = data.shape[0]
        self.data = data

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        random_idx = np.random.randint(0, len(self)) ## for scarf: a random index one from the dataset that will be used to corrupt the anchor
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, index, random_idx

    def __len__(self):
        return len(self.data)

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MVCL_f_m_sep:
    '''The Mutiview CL model'''

    def __init__(
            self,
            input_dims_f=1, # giving them default values otherwise can't call the function
            input_dims_m=1,
            med_dim=1,
            preops_input_dims_o=1,
            preops_input_dims_l=1,
            cbow_input_dims=1,
            hm_input_dims=1,
            pmh_input_dims=1,
            prob_list_input_dims=1,
            outcome_dim=1,
            alert_dim=1,
            output_dims_f=320,
            output_dims_m=320,
            output_dims_a=320,
            preops_output_dims_o=80,
            preops_output_dims_l=84,
            cbow_output_dims=100,
            homemeds_rep_dims=256,
            pmh_rep_dims = 85,
            prob_list_rep_dims = 85,
            outcome_rep_dims=50,
            hidden_dims=64,
            head_depth=2,
            depth=10,
            proj_dim=100,
            device='cuda',
            medid_embed_dim=5,
            w_pr=1000,
            w_out=1000,
            w_ts_pr=1000,
            w_std= 1000,
            w_cov= 1000,
            w_mse = 1000,
            w_ts_cross = 1000,
            lr=0.001,
            batch_size=16,
            max_train_length=None,
            temporal_unit=0,
            after_iter_callback=None,
            after_epoch_callback=None,
            save_dir=None,
            seed_used=0,
            alert_Ids=None,
            alertID_embed_dim=1,
    ):
        ''' Initialize a MVCL model.

        Args:
            input_dims_f (int): The input dimension for flowsheets. For a univariate time series, this should be set to 1.
            input_dims_m (int): The input dimension for medications. For a univariate time series, this should be set to 1.
            med_dim (int): Total number of medications. Used for creating the number of columns for med data modality.
            preops_input_dims_o (int): The input dimension for only preops modality (without labs). (Mostly 101)
            preops_input_dims_l (int): The input dimension for preops lab modality. (Mostly 110)
            cbow_input_dims (int): The input dimension for cbow (101 currently)
            hm_input_dims (int): The input dimension for embedded homemeds (500 currently)
            pmh_input_dims (int): The input dimension for embedded pmh (1024 currently)
            prob_list__input_dims (int): The input dimension for embedded problist (1024 currently)
            output_dims_f (int): The representation dimension flowsheets.
            output_dims_m (int): The representation dimension medications.
            preops_output_dims_o (int): The representation dimension for only preops modality (without labs).
            preops_output_dims_l (int): The representation dimension for preops lab modality.
            cbow_output_dims (int): The representation dimension for cbow (procedure text). Default 100 because it is possibly in the best extracted form.
            homemeds_rep_dims (int): The representation dimension for embedded home meds. default 256 (from 500).
            pmh_rep_dims (int): The representation dimension for embedded pmh. default 85 (from 128).
            prob_list_rep_dims (int): The representation dimension for embedded problist. default 85 (from 128).
            outcome_dim (int): Outcome view dimension
            outcome_rep_dim (int): outcome learnt rep dimension
            alert_dim (int) : alert view dimension
            output_dims_a (int): The representation dimension alerts.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            head_depth (int): The number of layers in the projection head that will be used to bring the different views in common space. Defaults to 2.
            proj_dim (int): Projection dimension for all the modalitites. Defaults to 100.
            w_pr (float) : weight with which the preop contrastive loss (SCARF based) would be weighed
            w_out (float) : weight with which the outcome contrastive loss (SubTab based) would be weighed
            w_ts_pr (float): weight multiplier for the inter view (ts vs preops) loss
            w_std (float): weight multiplier for the intra variance term of embeddings
            w_cov (float): weight multiplier for the intra covariance term of embeddings
            w_mse (float): weight multiplier for the between modality mse loss
            w_ts_cross (float): weight multiplier for across time series modalities via hierarchical loss
            device (int): The gpu used for training and inference.
            seed_used (int): the seed that was used in this iteration
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''

        super().__init__()
        self.device = device
        self.med_dims = med_dim
        self.preop_dims_o = preops_input_dims_o
        self.preop_dims_l = preops_input_dims_l
        self.cbow_dims = cbow_input_dims
        self.hm_dims = hm_input_dims
        self.pmh_dims = pmh_input_dims
        self.prob_list_dims = prob_list_input_dims
        self.outcome_dims = outcome_dim
        self.alerts_dims = alert_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.medid_embed_dim = medid_embed_dim
        self.preops_output_dims_o = preops_output_dims_o
        self.preops_output_dims_l = preops_output_dims_l
        self.cbow_output_dims = cbow_output_dims
        self.hm_output_dims = homemeds_rep_dims
        self.pmh_output_dims = pmh_rep_dims
        self.prob_list_output_dims = prob_list_rep_dims
        self.w_pr = w_pr
        self.w_ts_pr = w_ts_pr
        self.w_outcome = w_out
        self.w_std = w_std
        self.w_cov = w_cov
        self.w_mse_across = w_mse
        self.w_ts_cross = w_ts_cross
        self.alert_IDDim = alert_Ids
        self.alertId_embdim = alertID_embed_dim

        # self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, medid_embedDim=medid_embed_dim, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # self.net = torch.optim.swa_utils.AveragedModel(self._net)
        # self.net.update_parameters(self._net)

        # separate for flowsheets
        self._net_f = TSEncoder_f(input_dims=input_dims_f, output_dims=output_dims_f, hidden_dims=hidden_dims,
                                  depth=depth).to(self.device)
        self.net_f = torch.optim.swa_utils.AveragedModel(self._net_f)
        self.net_f.update_parameters(self._net_f)

        # separate for medications
        # self._net_m = TSEncoder_m(input_dims=input_dims_m, output_dims=output_dims_m, medid_embedDim=medid_embed_dim,
        #                           hidden_dims=hidden_dims, depth=depth).to(self.device)
        # self.net_m = torch.optim.swa_utils.AveragedModel(self._net_m)
        # self.net_m.update_parameters(self._net_m)

        # alternate strategy for combined medications processing
        self._net_m_alt = TSEncoder_m_alt(input_dims=med_dim, output_dims=output_dims_m,
                                  hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net_m_alt = torch.optim.swa_utils.AveragedModel(self._net_m_alt)
        self.net_m_alt.update_parameters(self._net_m_alt)


        # separate for alerts
        self._net_a = TSEncoder_a(input_dims=alert_dim, output_dims=output_dims_a, id_len=self.alert_IDDim, id_emb_dim= self.alertId_embdim, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # self._net_a = TSEncoder_f(input_dims=alert_dim, output_dims=output_dims_a, hidden_dims=hidden_dims,
        #                           depth=depth).to(self.device)
        self.net_a = torch.optim.swa_utils.AveragedModel(self._net_a)
        self.net_a.update_parameters(self._net_a)


        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0
        self.fd = save_dir
        self.seed = seed_used

        self.model_pr = preop_model.SCARF(input_dim=self.preop_dims_o, emb_dim=self.preops_output_dims_o,
                                          corruption_rate=0.6, ).to(self.device)  # preop model
        self.model_pr_l = preop_model.SCARF(input_dim=self.preop_dims_l, emb_dim=self.preops_output_dims_l,
                                          corruption_rate=0.6, ).to(self.device)  # preop model
        self.model_cbow = preop_model.SCARF(input_dim=self.cbow_dims, emb_dim=self.cbow_output_dims,
                                          corruption_rate=0.6, ).to(self.device)  # cbow model
        self.model_hm = preop_model.SCARF(input_dim=self.hm_dims, emb_dim=self.hm_output_dims,
                                          corruption_rate=0.6, ).to(self.device)  # hm model
        self.model_pmh = preop_model.SCARF(input_dim=self.pmh_dims, emb_dim=self.pmh_output_dims,
                                          corruption_rate=0.6, ).to(self.device)  # pmh model
        self.model_problist = preop_model.SCARF(input_dim=self.prob_list_dims, emb_dim=self.prob_list_output_dims,
                                          corruption_rate=0.6, ).to(self.device)  #prob list model

        self.model_outcomes = outcome_model.SubTab(outcome_dim, outcome_rep_dims).to(self.device)

        if self.w_ts_pr > 0:
            # self.ts_f_projection_head = MLP(output_dims_f, math.floor(0.5*proj_dim), head_depth).to(device) ## here the proj dimension is reduced because later on we are concatenating the flowsheets and meds for computing the loss
            # self.ts_m_projection_head = MLP(output_dims_m, math.ceil(0.5*proj_dim), head_depth).to(device)
            # self.pr_projection_head = MLP(preops_output_dims, math.floor(0.5*proj_dim), head_depth).to(device)
            # self.cbow_projection_head = MLP(cbow_output_dims, math.ceil(0.5*proj_dim), head_depth).to(device)
            self.ts_f_projection_head = MLP(output_dims_f, proj_dim, head_depth).to(device)
            self.ts_m_projection_head = MLP(output_dims_m, proj_dim, head_depth).to(device)
            self.ts_a_projection_head = MLP(output_dims_a, proj_dim, head_depth).to(device)
            # self.pr_projection_head = MLP(preops_output_dims, proj_dim, head_depth).to(device)
            self.pr_projection_head = MLP(preops_output_dims_o, proj_dim, head_depth).to(device)
            self.pr_projection_head_l = MLP(preops_output_dims_l, proj_dim, head_depth).to(device)
            self.cbow_projection_head = MLP(cbow_output_dims, proj_dim, head_depth).to(device)
            self.hm_projection_head = MLP(homemeds_rep_dims, proj_dim, head_depth).to(device)
            self.pmh_projection_head = MLP(pmh_rep_dims, proj_dim, head_depth).to(device)
            self.prob_list_projection_head = MLP(prob_list_rep_dims, proj_dim, head_depth).to(device)
            # self.outcome_proj_head = MLP(outcome_rep_dims, proj_dim, head_depth).to(device)
            self.outcome_proj_head = MLP(outcome_dim, proj_dim, head_depth).to(device) # this will just be a head on the outcome modality without learning any concreter representations here.


        elif self.w_mse_across > 0:
            self.ts_f_projection_head = MLP(output_dims_f, proj_dim, head_depth).to(device)
            self.ts_m_projection_head = MLP(output_dims_m, proj_dim, head_depth).to(device)
            self.ts_a_projection_head = MLP(output_dims_a, proj_dim, head_depth).to(device)
            self.pr_projection_head = MLP(preops_output_dims_o, proj_dim, head_depth).to(device)
            self.pr_projection_head_l = MLP(preops_output_dims_l, proj_dim, head_depth).to(device)
            self.cbow_projection_head = MLP(cbow_output_dims, proj_dim, head_depth).to(device)
            self.outcome_proj_head = MLP(outcome_rep_dims, proj_dim, head_depth).to(device)

    def fit(self, proc_modality_dict_train, n_epochs=None, n_iters=None, verbose=False, includePreops=True):
        ''' Training the MVCL model.

        Args:
            train_data_f (numpy.ndarray): The training data flowsheets.  It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            train_data_m (numpy.ndarray): The training data medications. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            train_ds_pr (torch tensor): Contains the preops data (without labs)
            train_ds_pr_l (torch tensor): Contains only preop labs
            train_ds_cbow (torch tensor) : Contains the cbow data
            train_ds_hm (torch tensor) : contains the home meds in embedded sum data form
            train_ds_pmh (torch tensor) : contains the pmh (from sherbet currently)
            train_ds_problist (torch tensor) : contains the prob list (from sherbet currently)
            train_outcomes (pandas df): Contains the outcomes dataframe (currently includes the the downstream label too)
            train_data_a (numpy.ndarray): Alerts training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''


        modalities_selected = proc_modality_dict_train.keys()
        # since the purpose of this whole set up os better ts representation so it makes sense to set atleast one ts modality as default
        assert ('flow' in modalities_selected) or ('meds' in modalities_selected)

        if 'flow' in modalities_selected:
            train_data_f = proc_modality_dict_train['flow']
            assert (train_data_f.ndim == 3)
            if self.max_train_length is not None:
                sections = train_data_f.shape[
                               1] // self.max_train_length  # checking only on flowsheets because during processing they have been brought to same time length of 511
                if sections >= 2:
                    train_data_f = np.concatenate(split_with_nan(train_data_f, sections, axis=1), axis=0)
            temporal_missing_f = np.isnan(train_data_f).all(axis=-1).any(axis=0)
            if temporal_missing_f[0] or temporal_missing_f[-1]:
                train_data_f = centerize_vary_length_series(train_data_f)
            train_data_f = train_data_f[~np.isnan(train_data_f).all(axis=2).all(axis=1)]

            # creating the loader only for flowsheets as the index can be used for others
            train_dataset = customdataset(torch.from_numpy(train_data_f).to(
                torch.float))  # this is being done to obtain indices of the samples in a batch
            # train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
            train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                      drop_last=True)
            optimizer_f = torch.optim.AdamW(self._net_f.parameters(), lr=self.lr)

        if 'meds' in modalities_selected:
            train_data_m = proc_modality_dict_train['meds']
            assert (train_data_m.ndim == 3)
            if self.max_train_length is not None:
                sections = train_data_m.shape[
                               1] // self.max_train_length
                if sections >= 2:
                    train_data_m = np.concatenate(split_with_nan(train_data_m, sections, axis=1), axis=0)
            temporal_missing_m = np.isnan(train_data_m).all(axis=-1).any(axis=0)
            if temporal_missing_m[0] or temporal_missing_m[-1]:
                train_data_m = centerize_vary_length_series(train_data_m)
            train_data_m = train_data_m[~np.isnan(train_data_m).all(axis=2).all(axis=1)]

            if 'flow' not in modalities_selected: # this is to take care of the case when its only med modalities
                train_dataset = customdataset(torch.from_numpy(train_data_m).to(torch.float))  # this is being done to obtain indices of the samples in a batch
                # train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
                train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)),
                                          shuffle=True,
                                          drop_last=True)
            else:
                # converting the medications to a tensor type from numpy type
                train_data_m = torch.from_numpy(train_data_m).to(torch.float)
            optimizer_m = torch.optim.AdamW(self._net_m_alt.parameters(), lr=self.lr)

        if n_iters is None and n_epochs is None and 'flow' in modalities_selected:
            n_iters = 200 if train_data_f.size <= 100000 else 600  # default param for n_iters
        elif n_iters is None and n_epochs is None and 'meds' in modalities_selected:
            n_iters = 200 if train_data_m.size <= 100000 else 600  # default param for n_iters


        if 'alerts' in modalities_selected:
            train_data_a = proc_modality_dict_train['alerts']
            assert (train_data_a.ndim == 4) # this is 4 because of multiple alerts at the same time
            if sections >= 2:
                train_data_a = np.concatenate(split_with_nan(train_data_a, sections, axis=1), axis=0)
            temporal_missing_a = np.isnan(train_data_a).all(axis=-1).any(axis=0).any(axis=1)
            if temporal_missing_a[0] or temporal_missing_a[-1]:
                train_data_a = centerize_vary_length_series(train_data_a)
            train_data_a = train_data_a[~np.isnan(train_data_a).all(axis=3).all(axis=2).all(axis=1)]
            # converting the medications to a tensor type from numpy type
            train_data_a = torch.from_numpy(train_data_a).to(torch.float)
            optimizer_a = torch.optim.AdamW(self._net_a.parameters(), lr=self.lr)

        loss_log = []

        if 'preops_o' in modalities_selected:
            train_ds_pr = proc_modality_dict_train['preops_o']
            train_ds_pr_l = proc_modality_dict_train['preops_l']
            train_ds_cbow = proc_modality_dict_train['cbow']
        if 'homemeds' in modalities_selected:
            train_ds_hm = proc_modality_dict_train['homemeds']
        if 'pmh' in modalities_selected:
            train_ds_pmh = proc_modality_dict_train['pmh']
        if 'problist' in modalities_selected:
            train_ds_prob_list = proc_modality_dict_train['problist']
        if 'postopcomp' in modalities_selected:
            train_outcomes = proc_modality_dict_train['postopcomp']

        # defining these by default; wont matter even if some modalities are not being ued
        optimizer_pr = Adam(self.model_pr.parameters(), lr=0.001)
        optimizer_pr_l = Adam(self.model_pr_l.parameters(), lr=0.001)
        optimizer_cbow = Adam(self.model_cbow.parameters(), lr=0.001)
        optimizer_hm = Adam(self.model_hm.parameters(), lr=0.001)
        optimizer_pmh = Adam(self.model_pmh.parameters(), lr=0.001)
        optimizer_prob_list = Adam(self.model_problist.parameters(), lr=0.001)
        ntxent_loss = loss.NTXent()
        ntxent_loss_pr_l = loss.NTXent()
        ntxent_loss_cbow = loss.NTXent()
        ntxent_loss_hm = loss.NTXent()
        ntxent_loss_pmh = loss.NTXent()
        ntxent_loss_prob_list = loss.NTXent()
        ts_vs_pr_loss = loss.NTXent(temperature=0.4)
        ts_vs_out_loss = loss.NTXent(temperature=0.4)
        pr_vs_out_loss = loss.NTXent(temperature=0.4)
        Rand_SUM_modCL_loss = loss.NTXent(temperature=0.4)

        # outcomes optimizer initialization
        optimizer_outcomes = torch.optim.AdamW(self.model_outcomes.parameters(), lr=0.0001, betas=(0.9, 0.999),
                                               eps=1e-07)
        withinCL_options = {'batch_size': self.batch_size, "tau": 0.1, "device": self.device, "cosine_similarity": True}
        outcome_within_cl_loss = loss.JointLoss(withinCL_options)

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            # breakpoint()
            for batch in train_loader:

                cross_modalities_loss = 0
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                Mcl_loss = 0

                if 'flow' in modalities_selected:
                    x_f = batch[0]
                    ts_l = x_f.size(1)
                elif 'meds' in modalities_selected:
                    x_m = batch[0]
                    ts_l = x_m.size(1)

                # this is the augmentation part (mainly cropping over the time dimension, masking is done inside the encoder training part)

                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=batch[0].size(0))

                if 'flow' in modalities_selected:
                    x_f = batch[0]
                    # breakpoint()
                    if self.max_train_length is not None and x_f.size(1) > self.max_train_length:
                        window_offset = np.random.randint(x_f.size(1) - self.max_train_length + 1)
                        x_f = x_f[:, window_offset: window_offset + self.max_train_length]
                    x_f = x_f.to(self.device)

                    optimizer_f.zero_grad()

                    out1_f = self._net_f(take_per_row(x_f, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_f = out1_f[:, -crop_l:]  # this is to make sure we have  only selected the overlapping part

                    out2_f = self._net_f(take_per_row(x_f, crop_offset + crop_left, crop_eright - crop_left))
                    out2_f = out2_f[:, :crop_l]

                    loss_ts_f = hierarchical_contrastive_loss(out1_f,out2_f,temporal_unit=self.temporal_unit)

                    Mcl_loss = Mcl_loss + loss_ts_f

                if 'alerts' in modalities_selected:
                    x_a = train_data_a[batch[1]]
                    if self.max_train_length is not None and x_a.size(1) > self.max_train_length:
                        window_offset = np.random.randint(x_a.size(1) - self.max_train_length + 1)
                        x_a = x_a[:, window_offset: window_offset + self.max_train_length]
                    x_a = x_a.to(self.device)

                    optimizer_a.zero_grad()
                    # using the same augmentation for alerts too
                    out1_a = self._net_a(take_per_row(x_a, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1_a = out1_a[:, -crop_l:]  # this is to make sure we have  only selected the overlapping part

                    out2_a = self._net_a(take_per_row(x_a, crop_offset + crop_left, crop_eright - crop_left))
                    out2_a = out2_a[:, :crop_l]

                    loss_ts_a = hierarchical_contrastive_loss(out1_a,out2_a,temporal_unit=self.temporal_unit)
                    Mcl_loss = Mcl_loss +  loss_ts_a
                    if 'flow' in modalities_selected:
                        cross_modalities_loss = cross_modalities_loss +\
                                                hierarchical_contrastive_loss(out1_a,out2_f,temporal_unit=self.temporal_unit) \
                                                + hierarchical_contrastive_loss(out1_f,out2_a,temporal_unit=self.temporal_unit)

                    out_ts_a = self._eval_with_pooling(x_a, 'a', encoding_window='full_series')
                    out_ts_a = out_ts_a.squeeze(1).to(device=self.device)

                if 'meds' in modalities_selected:
                    if 'flow' in modalities_selected:
                        x_m = train_data_m[batch[1]]
                    if self.max_train_length is not None and x_m.size(1) > self.max_train_length:
                        window_offset = np.random.randint(x_m.size(1) - self.max_train_length + 1)
                        x_m = x_m[:, window_offset: window_offset + self.max_train_length]
                    x_m = x_m.to(self.device)

                    optimizer_m.zero_grad()

                    ## alternative augmentation strategy is to convert the med id in each column for dose and units and use unit embed dim as 1

                    out1_m = self._net_m_alt(x_m, crop_offset + crop_eleft, crop_right - crop_eleft)
                    out1_m = out1_m[:, -crop_l:]  # this is to make sure we have  only selected the overlapping part
                    out2_m = self._net_m_alt(x_m, crop_offset + crop_left, crop_eright - crop_left)
                    out2_m = out2_m[:, :crop_l]
                    # breakpoint()

                    loss_ts_m = hierarchical_contrastive_loss(out1_m,out2_m,temporal_unit=self.temporal_unit)
                    Mcl_loss = Mcl_loss +  loss_ts_m

                    if 'flow' in modalities_selected:
                        cross_modalities_loss = cross_modalities_loss + \
                                                hierarchical_contrastive_loss(out1_f, out2_m, temporal_unit=self.temporal_unit) \
                                                + hierarchical_contrastive_loss(out1_m,out2_f,temporal_unit=self.temporal_unit)

                    out_ts_m = self._eval_with_pooling(x_m, 'm', encoding_window='full_series')
                    out_ts_m = out_ts_m.squeeze(1).to(device=self.device)

                if ('meds' in modalities_selected ) and ('alerts' in modalities_selected):
                    cross_modalities_loss = cross_modalities_loss \
                                            + hierarchical_contrastive_loss(out1_a,out2_m,temporal_unit=self.temporal_unit) \
                                            + hierarchical_contrastive_loss(out1_m,out2_a,temporal_unit=self.temporal_unit)

                """ Preops rep learning (labs have a seperate encoder) """
                if ('preops_o' in modalities_selected) or ('preops_l' in modalities_selected):
                    # learning the only preop representation
                    anchor, positive = train_ds_pr[batch[1]].to(self.device), train_ds_pr[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_pr.zero_grad()

                    # get embeddings
                    emb_anchor, emb_positive = self.model_pr(anchor, positive)

                    # compute loss
                    loss_pr = ntxent_loss(emb_anchor, emb_positive)

                    # learning the preop lab representation
                    anchor_l, positive_l = train_ds_pr_l[batch[1]].to(self.device), train_ds_pr_l[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_pr_l.zero_grad()

                    # get embeddings
                    emb_anchor_l, emb_positive_l = self.model_pr_l(anchor_l, positive_l)

                    # compute loss
                    loss_pr_l = ntxent_loss_pr_l(emb_anchor_l, emb_positive_l)

                    Mcl_loss = Mcl_loss + 10 * (self.w_pr * (loss_pr + loss_pr_l))

                """ CBOW rep learning """
                if 'cbow' in modalities_selected:
                    anchor_bw, positive_bw = train_ds_cbow[batch[1]].to(self.device), train_ds_cbow[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_cbow.zero_grad()

                    # get embeddings
                    emb_anchor_bw, emb_positive_bw = self.model_cbow(anchor_bw, positive_bw)

                    # compute loss
                    loss_bw = ntxent_loss_cbow(emb_anchor_bw, emb_positive_bw)
                    Mcl_loss = Mcl_loss + 10 * self.w_pr * loss_bw

                """ HM rep learning """
                if 'homemeds' in modalities_selected:
                    anchor_hm, positive_hm = train_ds_hm[batch[1]].to(self.device), train_ds_hm[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_hm.zero_grad()

                    # get embeddings
                    emb_anchor_hm, emb_positive_hm = self.model_hm(anchor_hm, positive_hm)

                    # compute loss
                    loss_hm = ntxent_loss_hm(emb_anchor_hm, emb_positive_hm)
                    Mcl_loss = Mcl_loss + 10 * self.w_pr * loss_hm


                """ pmh rep learning """
                if 'pmh' in modalities_selected:
                    anchor_pmh, positive_pmh = train_ds_pmh[batch[1]].to(self.device), train_ds_pmh[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_pmh.zero_grad()

                    # get embeddings
                    emb_anchor_pmh, emb_positive_pmh = self.model_pmh(anchor_pmh, positive_pmh)

                    # compute loss
                    loss_pmh = ntxent_loss_pmh(emb_anchor_pmh, emb_positive_pmh)
                    Mcl_loss = Mcl_loss + 10 * self.w_pr * loss_pmh


                """ prob list rep learning """
                if 'problist' in modalities_selected:
                    anchor_prob_list, positive_prob_list = train_ds_prob_list[batch[1]].to(self.device), train_ds_prob_list[batch[2]].to(self.device)

                    # reset gradients
                    optimizer_prob_list.zero_grad()

                    # get embeddings
                    emb_anchor_prob_list, emb_positive_prob_list = self.model_problist(anchor_prob_list, positive_prob_list)

                    # compute loss
                    loss_prob_list = ntxent_loss_prob_list(emb_anchor_prob_list, emb_positive_prob_list)
                    Mcl_loss = Mcl_loss + 10 * self.w_pr * loss_prob_list


                """ Outcome rep learning  """
                if 'postopcomp' in modalities_selected:
                # learning the outcome representation. Steps: generate subsets, make subset combinations, pass all the combinations through the encoder, aggregate the loss and then update
                    # breakpoint()
                    dt_dict = {} # this dictionary will, be used to separate the outcomes' reconstruction activation function
                    for i in range(len(train_outcomes.dtypes.value_counts().index)):
                        dt_dict[list(train_outcomes.dtypes.value_counts().index)[i]] = \
                        train_outcomes.dtypes.value_counts().iloc[i]
                    outcome_batch = train_outcomes.values[batch[1]]
                    x_orig = self.model_outcomes.process_batch(outcome_batch,
                                                               outcome_batch)  # this basically creates a copy
                    x_tilde_list = self.model_outcomes.subset_generator(outcome_batch)
                    x_subsetted = self.model_outcomes.get_combinations_of_subsets(x_tilde_list)

                    # breakpoint()
                    # x_subsetted = self.model_outcomes.train_epoch(outcome_batch)
                    # pass data through model
                    cont_loss = []  # list to keep track of the between subset pair losses
                    optimizer_outcomes.zero_grad()
                    for xi in x_subsetted:
                        z, latent, recon = self.model_outcomes(
                            xi, dt_dict)  # latent is just the output of the encoder and z is the projected output
                        tloss = outcome_within_cl_loss(z, recon, x_orig)
                        cont_loss.append(tloss)
                    # breakpoint()

                    # outcome_totalloss = sum(cont_loss) / len(cont_loss)
                    outcome_totalloss=0  # for now the outcome has only a projection head that is trained contrastively because there is no loss for it.
                    # there could be a cross entropy or mse here but then it will be more like recontruction loss??

                    # Aggregation of subset latent representations
                    latent_list = []  # this list to keep track of the repr of all the subsets to aggregate for the multiview loss
                    # breakpoint()
                    for xi in x_tilde_list:
                        _, latent, _ = self.model_outcomes(torch.tensor(xi, device=self.device), dt_dict)
                        # Collect latent
                        latent_list.append(latent)
                    latent_agg = self.model_outcomes.aggregate(latent_list)

                    Mcl_loss = Mcl_loss + 10 * self.w_outcome * outcome_totalloss

                """ Multiview loss preparation, separate head for each modality """

                # compute the multi-view loss between the different views

                if 'flow' in modalities_selected:
                    # this step will change the dimension from N * T * F --> N * ts_emb_dim by including max pooling too
                    out_ts_f = self._eval_with_pooling(x_f, 'f', encoding_window='full_series')
                    out_ts_f = out_ts_f.squeeze(1).to(device=self.device)

                if (self.w_ts_pr > 0) and (len(modalities_selected) >= 2):
                    ## alternate of selecting two sets of modalities, adding their representations and then taking the contrast between the two sets
                    # create a list of projected in order of flowsheets, med, alerts, preops, preop_lab, bow, hm, pmh, problist, outcomes
                    proj_list = []
                    if 'flow' in modalities_selected: proj_list.append(self.ts_f_projection_head(out_ts_f))
                    if 'meds' in modalities_selected: proj_list.append(self.ts_m_projection_head(out_ts_m))
                    if 'alerts' in modalities_selected: proj_list.append(self.ts_a_projection_head(out_ts_a))
                    if 'preops_o' in modalities_selected:
                        proj_list.append(self.pr_projection_head(emb_anchor))
                        proj_list.append(self.pr_projection_head_l(emb_anchor_l))
                        proj_list.append(self.cbow_projection_head(emb_anchor_bw))
                    if 'homemeds' in modalities_selected: proj_list.append(self.hm_projection_head(emb_anchor_hm))
                    if 'pmh' in modalities_selected: proj_list.append(self.pmh_projection_head(emb_anchor_pmh))
                    if 'problist' in modalities_selected: proj_list.append(self.prob_list_projection_head(emb_anchor_prob_list))
                    if 'postopcomp' in modalities_selected:
                        # when there is no rep learning for outcomes but just aproj head we feed the outcomes directly into that mlp as follows:
                        latent_agg = torch.from_numpy(outcome_batch).to(device=self.device).float()
                        proj_list.append(self.outcome_proj_head(latent_agg))

                    set1 = set(np.random.choice(np.arange(len(proj_list)), math.ceil(len(proj_list)*0.5)))
                    set2 = set(range(len(proj_list))) - set1

                    proj_set1 = torch.zeros(proj_list[0].shape).to(device=self.device)
                    proj_set2 = torch.zeros(proj_list[0].shape).to(device=self.device)
                    for i in set1: proj_set1 = proj_set1 + proj_list[i]
                    for i in set2: proj_set2 = proj_set2 + proj_list[i]

                    across_Rand_SUM_mod_loss = Rand_SUM_modCL_loss(proj_set1, proj_set2)

                if len(modalities_selected) < 2:
                    across_Rand_SUM_mod_loss = 0 # this is not utilizing the multiview component of the method

                # two kinds of regularizers : 1) to add variation and, 2) to avoid information collapse by decorrelating the embedding dimension

                std_loss = 0
                cov_loss = 0
                if 'flow' in modalities_selected:
                    f_em = self.ts_f_projection_head(out_ts_f) - self.ts_f_projection_head(out_ts_f).mean(dim=0)
                    std_f = torch.sqrt(f_em.var(dim=0) + 0.0001)
                    cov_f = (f_em.T @ f_em) / (self.batch_size - 1)
                    std_loss = torch.mean(F.relu(1 - std_f)) / 2
                    cov_loss = off_diagonal(cov_f).pow_(2).sum().div(f_em.shape[-1])

                if 'meds' in modalities_selected:
                    m_em = self.ts_m_projection_head(out_ts_m) - self.ts_m_projection_head(out_ts_m).mean(dim=0)
                    std_m = torch.sqrt(m_em.var(dim=0) + 0.0001)
                    cov_m = (m_em.T @ m_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_m)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_m).pow_(2).sum().div(m_em.shape[-1])

                if 'alerts' in modalities_selected:
                    a_em = self.ts_a_projection_head(out_ts_a) - self.ts_a_projection_head(out_ts_a).mean(dim=0)
                    std_a = torch.sqrt(a_em.var(dim=0) + 0.0001)
                    cov_a = (a_em.T @ a_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_a)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_a).pow_(2).sum().div(a_em.shape[-1])
                # following assumes that preops include preops_o, preops_l, and preops text
                if 'preops_o' in modalities_selected:
                    pr_em = self.pr_projection_head(emb_anchor) - self.pr_projection_head(emb_anchor).mean(dim=0)
                    pr_em_l = self.pr_projection_head_l(emb_anchor_l) - self.pr_projection_head_l(emb_anchor_l).mean(
                        dim=0)
                    bw_em = self.cbow_projection_head(emb_anchor_bw) - self.cbow_projection_head(emb_anchor_bw).mean(
                        dim=0)

                    std_pr = torch.sqrt(pr_em.var(dim=0) + 0.0001)
                    std_pr_l = torch.sqrt(pr_em_l.var(dim=0) + 0.0001)
                    std_bw = torch.sqrt(bw_em.var(dim=0) + 0.0001)

                    cov_pr = (pr_em.T @ pr_em) / (self.batch_size - 1)
                    cov_pr_l = (pr_em_l.T @ pr_em_l) / (self.batch_size - 1)
                    cov_bw = (bw_em.T @ bw_em) / (self.batch_size - 1)

                    std_loss = std_loss + torch.mean(F.relu(1 - std_pr)) / 2 + torch.mean(F.relu(1 - std_pr_l)) / 2 + torch.mean(
                        F.relu(1 - std_bw)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_pr).pow_(2).sum().div(pr_em.shape[-1]) + off_diagonal(cov_pr_l).pow_(
                        2).sum().div(pr_em_l.shape[-1]) \
                               + off_diagonal(cov_bw).pow_(2).sum().div(bw_em.shape[-1])

                if 'homemeds' in modalities_selected:
                    hm_em = self.hm_projection_head(emb_anchor_hm) - self.hm_projection_head(emb_anchor_hm).mean(dim=0)
                    std_hm = torch.sqrt(hm_em.var(dim=0) + 0.0001)
                    cov_hm = (hm_em.T @ hm_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_hm)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_hm).pow_(2).sum().div(hm_em.shape[-1])

                if 'pmh' in modalities_selected:
                    pmh_em = self.pmh_projection_head(emb_anchor_pmh) - self.pmh_projection_head(emb_anchor_pmh).mean(
                        dim=0)
                    std_pmh = torch.sqrt(pmh_em.var(dim=0) + 0.0001)
                    cov_pmh = (pmh_em.T @ pmh_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_pmh)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_pmh).pow_(2).sum().div(pmh_em.shape[-1])

                if 'problist' in modalities_selected:
                    prob_list_em = self.prob_list_projection_head(emb_anchor_prob_list) - self.prob_list_projection_head(emb_anchor_prob_list).mean(dim=0)
                    std_prob_list = torch.sqrt(prob_list_em.var(dim=0) + 0.0001)
                    cov_prob_list = (prob_list_em.T @ prob_list_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_prob_list)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_prob_list).pow_(2).sum().div(prob_list_em.shape[-1])

                if 'postopcomp' in modalities_selected:
                    outcome_em = self.outcome_proj_head(latent_agg) - self.outcome_proj_head(latent_agg).mean(dim=0)
                    std_out = torch.sqrt(outcome_em.var(dim=0) + 0.0001)
                    cov_out = (outcome_em.T @ outcome_em) / (self.batch_size - 1)
                    std_loss = std_loss + torch.mean(F.relu(1 - std_out)) / 2
                    cov_loss = cov_loss + off_diagonal(cov_out).pow_(2).sum().div(outcome_em.shape[-1])


                Mcl_loss = Mcl_loss + (self.w_ts_cross * cross_modalities_loss) \
                           + self.w_std * std_loss + self.w_cov * cov_loss # the last two terms are regularizers which stay for all cases

                # breakpoint()
                if self.w_mse_across >0:  # will tale care of this part later
                    # usual mse between the embeddings of different modalities that was originally there in the paper; will be used as a replacement of the between modality contrastive loss
                    f_m_rep_loss = F.mse_loss(self.ts_f_projection_head(out_ts_f), self.ts_m_projection_head(out_ts_m))
                    f_pr_rep_loss = F.mse_loss(self.ts_f_projection_head(out_ts_f), self.pr_projection_head(emb_anchor))
                    f_bw_rep_loss = F.mse_loss(self.ts_f_projection_head(out_ts_f),
                                               self.cbow_projection_head(emb_anchor_bw))
                    f_out_rep_loss = F.mse_loss(self.ts_f_projection_head(out_ts_f), self.outcome_proj_head(latent_agg))
                    m_pr_rep_loss = F.mse_loss(self.ts_m_projection_head(out_ts_m), self.pr_projection_head(emb_anchor))
                    m_bw_rep_loss = F.mse_loss(self.ts_m_projection_head(out_ts_m),
                                               self.cbow_projection_head(emb_anchor_bw))
                    m_out_rep_loss = F.mse_loss(self.ts_m_projection_head(out_ts_m), self.outcome_proj_head(latent_agg))
                    pr_bw_rep_loss = F.mse_loss(self.pr_projection_head(emb_anchor),
                                                self.cbow_projection_head(emb_anchor_bw))
                    pr_out_rep_loss = F.mse_loss(self.pr_projection_head(emb_anchor),
                                                 self.outcome_proj_head(latent_agg))
                    bw_out_rep_loss = F.mse_loss(self.cbow_projection_head(emb_anchor_bw),
                                                 self.outcome_proj_head(latent_agg))

                    mseL = f_m_rep_loss + f_pr_rep_loss + f_bw_rep_loss + f_out_rep_loss + \
                           m_pr_rep_loss + m_bw_rep_loss + m_out_rep_loss + \
                           pr_bw_rep_loss + pr_out_rep_loss + \
                           bw_out_rep_loss

                    Mcl_loss = Mcl_loss +  self.w_mse_across * mseL
                    assert self.w_ts_pr == 0, "mse instead of contrastive"
                elif self.w_ts_pr>0:
                    Mcl_loss = Mcl_loss + 10* (self.w_ts_pr * (across_Rand_SUM_mod_loss))

                # print(loss_ts_f, loss_ts_m, loss_pr, loss_pr_l, loss_bw, loss_hm, outcome_totalloss, across_Rand_SUM_mod_loss, cross_modalities_loss)

                Mcl_loss.backward(retain_graph=True)

                optimizer_pr.step()
                optimizer_pr_l.step()
                optimizer_cbow.step()
                optimizer_hm.step()
                # optimizer_outcomes.step() # dont want to learn outcome rep
                if 'flow' in modalities_selected:
                    optimizer_f.step()
                    self.net_f.update_parameters(self._net_f)
                if 'alerts' in modalities_selected:
                    optimizer_a.step()
                    self.net_a.update_parameters(self._net_a)
                if 'meds' in modalities_selected:
                    optimizer_m.step()
                    self.net_m_alt.update_parameters(self._net_m_alt)

                cum_loss += Mcl_loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, Mcl_loss.item())

                # del loss_ts_f, loss_ts_m, loss_ts_a, loss_pr, loss_bw, outcome_totalloss
                # gc.collect

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        # saving the preops model to be used later
        torch.save(self.net_f.state_dict(), f'{self.fd}/{self.seed}_model_f.pkl')
        torch.save(self.net_m_alt.state_dict(), f'{self.fd}/{self.seed}_model_m.pkl')
        torch.save(self.net_a.state_dict(), f'{self.fd}/{self.seed}_model_a.pkl')
        torch.save(self.model_pr.state_dict(), f'{self.fd}/{self.seed}_model_pr.pkl')
        torch.save(self.model_pr_l.state_dict(), f'{self.fd}/{self.seed}_model_pr_labs.pkl')
        torch.save(self.model_cbow.state_dict(), f'{self.fd}/{self.seed}_model_cbow.pkl')
        torch.save(self.model_hm.state_dict(), f'{self.fd}/{self.seed}_model_hm.pkl')
        torch.save(self.model_pmh.state_dict(), f'{self.fd}/{self.seed}_model_pmh.pkl')
        torch.save(self.model_problist.state_dict(), f'{self.fd}/{self.seed}_model_problist.pkl')
        torch.save(self.model_outcomes.state_dict(), f'{self.fd}/{self.seed}_model_outcome_rep.pkl')

        # saving the projection heads to be used later
        torch.save(self.ts_f_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_flow.pkl')
        torch.save(self.ts_m_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_meds.pkl')
        torch.save(self.ts_a_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_alerts.pkl')
        torch.save(self.pr_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_pr.pkl')
        torch.save(self.pr_projection_head_l.state_dict(), f'{self.fd}/{self.seed}_proj_head_pr_labs.pkl')
        torch.save(self.cbow_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_cbow.pkl')
        torch.save(self.hm_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_hm.pkl')
        torch.save(self.pmh_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_pmh.pkl')
        torch.save(self.prob_list_projection_head.state_dict(), f'{self.fd}/{self.seed}_proj_head_problist.pkl')

        # breakpoint()
        return loss_log

    def _eval_with_pooling(self, x, flag='f', mask=None, slicing=None, encoding_window=None):  # flag basically indicates which network to use out of flowsheets or meds
        if flag =='f':
            out = self.net_f(x.to(self.device, non_blocking=True), mask)
        if flag =='a':
            out = self.net_a(x.to(self.device, non_blocking=True), mask)
        if flag == 'm':
            # out = self.net_m(x.to(self.device, non_blocking=True), mask)
            out = self.net_m_alt(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(self, data, flag='f',mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0,
               batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            flag (str) : indicates whether to use flowsheet or med or alerts network
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        '''
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        if flag =='f':
            assert self.net_f is not None, 'please train or load a net first'
            org_training = self.net_f.training
            self.net_f.eval()
        if flag == 'm':
            # assert self.net_m is not None, 'please train or load a net first'
            assert self.net_m_alt is not None, 'please train or load a net first'
            org_training = self.net_m_alt.training
            self.net_m_alt.eval()
        if flag =='a':
            assert self.net_a is not None, 'please train or load a net first'
            org_training = self.net_a.training
            self.net_a.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    flag,
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                flag,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                flag,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    # breakpoint()
                    out = self._eval_with_pooling(x, flag,mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        # breakpoint()
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        if flag =='f':
            self.net_f.train(org_training)
        if flag == 'm':
            # self.net_m.train(org_training)
            self.net_m_alt.train(org_training)
        if flag =='a':
            self.net_a.train(org_training)

        return output.numpy()

    def save(self, fn, flag='f'):
        ''' Save the model to a file.

        Args:
            fn (str): filename.
        '''
        if flag =='f':
            torch.save(self.net_f.state_dict(), fn)
        if flag =='m':
            # torch.save(self.net_m.state_dict(), fn)
            torch.save(self.net_m_alt.state_dict(), fn)
        if flag =='a':
            torch.save(self.net_a.state_dict(), fn)

    def load(self, fn, flag='f'):
        ''' Load the model from a file.

        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        if flag =='f':
            self.net_f.load_state_dict(state_dict)
        if flag =='m':
            # self.net_m.load_state_dict(state_dict)
            self.net_m_alt.load_state_dict(state_dict)
        if flag =='a':
            self.net_a.load_state_dict(state_dict)

    def pr_dataset_embeddings(self, loader, inf=0):  # this is for the preops
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_pr.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_pr.load_state_dict(state_dict)
        self.model_pr.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_pr.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_pr.pkl', map_location=self.device)
            self.pr_projection_head.load_state_dict(state_dict_proj_head)
            self.pr_projection_head.eval()
            return self.pr_projection_head(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def pr_l_dataset_embeddings(self, loader, inf=0):  # this is for the preops lab
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_pr_labs.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_pr_l.load_state_dict(state_dict)
        self.model_pr_l.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_pr_l.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_pr_labs.pkl', map_location=self.device)
            self.pr_projection_head_l.load_state_dict(state_dict_proj_head)
            self.pr_projection_head_l.eval()
            return self.pr_projection_head_l(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def cbow_dataset_embeddings(self, loader, inf=0):  # this is for the cbow
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_cbow.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_cbow.load_state_dict(state_dict)
        self.model_cbow.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_cbow.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_cbow.pkl', map_location=self.device)
            self.cbow_projection_head.load_state_dict(state_dict_proj_head)
            self.cbow_projection_head.eval()
            return self.cbow_projection_head(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def hm_dataset_embeddings(self, loader, inf=0):  # this is for the hm
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_hm.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_hm.load_state_dict(state_dict)
        self.model_hm.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_hm.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_hm.pkl', map_location=self.device)
            self.hm_projection_head.load_state_dict(state_dict_proj_head)
            self.hm_projection_head.eval()
            return self.hm_projection_head(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def pmh_dataset_embeddings(self, loader, inf=0):  # this is for the pmh
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_pmh.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_pmh.load_state_dict(state_dict)
        self.model_pmh.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_pmh.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_pmh.pkl', map_location=self.device)
            self.pmh_projection_head.load_state_dict(state_dict_proj_head)
            self.pmh_projection_head.eval()
            return self.pmh_projection_head(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def problist_dataset_embeddings(self, loader, inf=0):  # this is for the problist
        # breakpoint()
        state_dict = torch.load(f'{self.fd}/{self.seed}_model_problist.pkl', map_location=self.device)
        # model_pr = self.model_pr.load_state_dict(state_dict)
        self.model_problist.load_state_dict(state_dict)
        self.model_problist.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _ in tqdm(loader):
                anchor = anchor.to(self.device)
                embeddings.append(self.model_problist.get_embeddings(anchor))
        # breakpoint()
        if inf==1:
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_problist.pkl', map_location=self.device)
            self.prob_list_projection_head.load_state_dict(state_dict_proj_head)
            self.prob_list_projection_head.eval()
            return self.prob_list_projection_head(torch.cat(embeddings)).cpu().detach().numpy()
        else:
            embeddings = torch.cat(embeddings).cpu().numpy()
            return embeddings

    def associationBTWalertsANDrestmodalities(self, proc_modality_dict_test):
        modalities_selected = proc_modality_dict_test.keys()

        if 'alerts' in modalities_selected:
            test_data_a = proc_modality_dict_test['alerts']
            temporal_missing_a = np.isnan(test_data_a).all(axis=-1).any(axis=0).any(axis=1)
            if temporal_missing_a[0] or temporal_missing_a[-1]:
                test_data_a = centerize_vary_length_series(test_data_a)
            test_data_a = test_data_a[~np.isnan(test_data_a).all(axis=3).all(axis=2).all(axis=1)]
            if True:
                test_data_a = torch.from_numpy(test_data_a).to(torch.float).to(device=self.device)
                timepoint_list = {}
                timepoint_list_a = {}
                for i in range(len(test_data_a)):
                    if torch.nonzero(test_data_a[i]).size()[0]>0:
                        if torch.nonzero(test_data_a[i,:,:,1]).shape[0] > 1:
                            randNchoice =  np.random.choice(np.arange(len(torch.nonzero(test_data_a[i,:,:,1]))), size=1,replace=False)[0]
                            timepoint_list_a[i] = torch.nonzero(test_data_a[i,:,:,1]).squeeze().cpu().detach().numpy()[randNchoice]
                            timepoint_list[i] = torch.nonzero(test_data_a[i,:,:,1]).squeeze().cpu().detach().numpy()[randNchoice][0]
                        else:
                            timepoint_list_a[i] = torch.nonzero(test_data_a[i,:,:,1]).cpu().detach().numpy()[0]
                            timepoint_list[i] = torch.nonzero(test_data_a[i,:,:,1]).cpu().detach().numpy()[0][0]
                samplewithalerts = [val for val in timepoint_list_a.keys()]
                timevals = [val[0] for val in timepoint_list_a.values()]
                alertpos = [val[1] for val in timepoint_list_a.values()]
                test_data_aSelected = test_data_a[samplewithalerts,timevals, alertpos,:]
                # test_data_a = test_data_a[list(timepoint_list_a.keys()),list(timepoint_list.values()), :,:]
            else:
                test_data_aSelected = test_data_a.sum(axis=1)  # this is being done to eliminate the time dimension that is not present in the other representations
        proj_list = []
        if 'flow' in modalities_selected:
            test_data_f = proc_modality_dict_test['flow']
            temporal_missing_f = np.isnan(test_data_f).all(axis=-1).any(axis=0)
            if temporal_missing_f[0] or temporal_missing_f[-1]:
                test_data_f = centerize_vary_length_series(test_data_f)
            test_data_f = test_data_f[~np.isnan(test_data_f).all(axis=2).all(axis=1)]
            test_data_f = torch.from_numpy(test_data_f).to(torch.float).to(device=self.device)

            state_dict = torch.load(f'{self.fd}/{self.seed}_model_f.pkl', map_location=self.device)
            self.net_f.load_state_dict(state_dict)
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_flow.pkl', map_location=self.device)
            self.ts_f_projection_head.load_state_dict(state_dict_proj_head)
            self.ts_f_projection_head.eval()

            if True:
                out_ts_f = self._eval_with_pooling(test_data_f[list(timepoint_list.keys())[0],:list(timepoint_list.values())[0]+1, :].unsqueeze(0), 'f', encoding_window='full_series')  # +1 because the list contains the exact indexes that start from zero
                out_ts_f = out_ts_f.squeeze(1).to(device=self.device)
                proj_rep_f = self.ts_f_projection_head(out_ts_f).cpu().detach()

                for i in range(1, len(timepoint_list)):
                    out_ts_f = self._eval_with_pooling(
                        test_data_f[list(timepoint_list.keys())[i], :list(timepoint_list.values())[i]+1, :].unsqueeze(0),
                        'f', encoding_window='full_series')
                    out_ts_f = out_ts_f.squeeze(1).to(device=self.device)
                    proj_rep_f = torch.concat([proj_rep_f, self.ts_f_projection_head(out_ts_f).cpu().detach()], dim=0)

            else:
                # this alernative uses all the time series data
                out_ts_f = self._eval_with_pooling(test_data_f[:self.batch_size,:,:], 'f', encoding_window='full_series')
                out_ts_f = out_ts_f.squeeze(1).to(device=self.device)
                proj_rep_f = self.ts_f_projection_head(out_ts_f).cpu().detach()

                # this roundabout is being done because all the data is not fitting on the gpu at once.
                n_batches, remaining_samples = divmod((test_data_f.shape[0]-self.batch_size), self.batch_size)
                for i in range(1, n_batches+1):
                    out_ts_f = self._eval_with_pooling(test_data_f[(i*self.batch_size):((i+1)*self.batch_size), :, :], 'f',encoding_window='full_series')
                    out_ts_f = out_ts_f.squeeze(1).to(device=self.device)
                    proj_rep_f = torch.concat([proj_rep_f,self.ts_f_projection_head(out_ts_f).cpu().detach()], dim=0)

                if remaining_samples>0:
                    out_ts_f = self._eval_with_pooling(test_data_f[((n_batches + 1) * self.batch_size):, :, :], 'f',encoding_window='full_series')
                    out_ts_f = out_ts_f.squeeze(1).to(device=self.device)
                    proj_rep_f = torch.concat([proj_rep_f, self.ts_f_projection_head(out_ts_f).cpu().detach()], dim=0)

            proj_list.append(proj_rep_f.numpy())

        if 'meds' in modalities_selected:
            test_data_m = proc_modality_dict_test['meds']
            temporal_missing_m = np.isnan(test_data_m).all(axis=-1).any(axis=0)
            if temporal_missing_m[0] or temporal_missing_m[-1]:
                test_data_m = centerize_vary_length_series(test_data_m)
            test_data_m = test_data_m[~np.isnan(test_data_m).all(axis=2).all(axis=1)]
            # converting the medications to a tensor type from numpy type
            test_data_m = torch.from_numpy(test_data_m).to(torch.float).to(device=self.device)

            state_dict = torch.load(f'{self.fd}/{self.seed}_model_m.pkl', map_location=self.device)
            self.net_m_alt.load_state_dict(state_dict)
            state_dict_proj_head = torch.load(f'{self.fd}/{self.seed}_proj_head_meds.pkl', map_location=self.device)
            self.ts_m_projection_head.load_state_dict(state_dict_proj_head)
            self.ts_m_projection_head.eval()

            if True:
                out_ts_m = self._eval_with_pooling(test_data_m[list(timepoint_list.keys())[0],:list(timepoint_list.values())[0]+1, :].unsqueeze(0), 'm', encoding_window='full_series')
                out_ts_m = out_ts_m.squeeze(1).to(device=self.device)
                proj_rep_m = self.ts_m_projection_head(out_ts_m).cpu().detach()

                for i in range(1, len(timepoint_list)):
                    out_ts_m = self._eval_with_pooling(
                        test_data_m[list(timepoint_list.keys())[i], :list(timepoint_list.values())[i]+1, :].unsqueeze(0),
                        'm', encoding_window='full_series')
                    out_ts_m = out_ts_m.squeeze(1).to(device=self.device)
                    proj_rep_m = torch.concat([proj_rep_m, self.ts_m_projection_head(out_ts_m).cpu().detach()], dim=0)

            else:
                out_ts_m = self._eval_with_pooling(test_data_m[:self.batch_size,:,:], 'm', encoding_window='full_series')
                out_ts_m = out_ts_m.squeeze(1).to(device=self.device)
                proj_rep_m = self.ts_m_projection_head(out_ts_m).cpu().detach()

                # this roundabout is being done because all the data is not fitting on the gpu at once.
                n_batches, remaining_samples = divmod((test_data_m.shape[0]-self.batch_size), self.batch_size)
                for i in range(1, n_batches+1):
                    out_ts_m = self._eval_with_pooling(test_data_m[(i*self.batch_size):((i+1)*self.batch_size), :, :], 'm',encoding_window='full_series')
                    out_ts_m = out_ts_m.squeeze(1).to(device=self.device)
                    proj_rep_m = torch.concat([proj_rep_m,self.ts_m_projection_head(out_ts_m).cpu().detach()], dim=0)

                if remaining_samples>0:
                    out_ts_m = self._eval_with_pooling(test_data_m[((n_batches + 1) * self.batch_size):, :, :], 'm',encoding_window='full_series')
                    out_ts_m = out_ts_m.squeeze(1).to(device=self.device)
                    proj_rep_m = torch.concat([proj_rep_m, self.ts_m_projection_head(out_ts_m).cpu().detach()], dim=0)

            proj_list.append(proj_rep_m.numpy())

        if ('preops_o' in modalities_selected) or ('preops_l' in modalities_selected):
            proc_modality_dict_test['preops_o'] = torch.tensor(proc_modality_dict_test['preops_o'], dtype=torch.float)
            proc_modality_dict_test['preops_l'] = torch.tensor(proc_modality_dict_test['preops_l'], dtype=torch.float)

            test_pr = proc_modality_dict_test['preops_o']
            test_pr_l = proc_modality_dict_test['preops_l']

            test_ds = preop_model.ExampleDataset(test_pr)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_pr = self.pr_dataset_embeddings(test_loader, inf=1)

            test_ds = preop_model.ExampleDataset(test_pr_l)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_pr_l = self.pr_l_dataset_embeddings(test_loader, inf=1)

            if True:
                test_repr_pr = test_repr_pr[list(timepoint_list.keys()),:]
                test_repr_pr_l = test_repr_pr_l[list(timepoint_list.keys()),:]

            proj_list.append(test_repr_pr)
            proj_list.append(test_repr_pr_l)

        if 'cbow' in modalities_selected:
            proc_modality_dict_test['cbow'] = torch.tensor(proc_modality_dict_test['cbow'], dtype=torch.float)
            test_bw = proc_modality_dict_test['cbow']
            test_ds = preop_model.ExampleDataset(test_bw)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_bw = self.cbow_dataset_embeddings(test_loader, inf=1)
            if True:
                test_repr_bw = test_repr_bw[list(timepoint_list.keys()),:]
            proj_list.append(test_repr_bw)

        if 'homemeds' in modalities_selected:
            proc_modality_dict_test['homemeds'] = torch.tensor(proc_modality_dict_test['homemeds'], dtype=torch.float)
            test_hm = proc_modality_dict_test['homemeds']
            test_ds = preop_model.ExampleDataset(test_hm)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_hm = self.hm_dataset_embeddings(test_loader, inf=1)
            if True:
                test_repr_hm = test_repr_hm[list(timepoint_list.keys()),:]
            proj_list.append(test_repr_hm)

        if 'pmh' in modalities_selected:
            proc_modality_dict_test['pmh'] = torch.tensor(proc_modality_dict_test['pmh'], dtype=torch.float)
            test_pmh = proc_modality_dict_test['pmh']
            test_ds = preop_model.ExampleDataset(test_pmh)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_pmh = self.pmh_dataset_embeddings(test_loader, inf=1)
            if True:
                test_repr_pmh = test_repr_pmh[list(timepoint_list.keys()),:]
            proj_list.append(test_repr_pmh)

        if 'problist' in modalities_selected:
            proc_modality_dict_test['problist'] = torch.tensor(proc_modality_dict_test['problist'], dtype=torch.float)
            test_pblist = proc_modality_dict_test['problist']
            test_ds = preop_model.ExampleDataset(test_pblist)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            test_repr_pblist = self.problist_dataset_embeddings(test_loader, inf=1)
            if True:
                test_repr_pblist = test_repr_pblist[list(timepoint_list.keys()),:]
            proj_list.append(test_repr_pblist)

        predictors = torch.zeros(proj_list[0].shape).numpy()
        for i in range(len(proj_list)): predictors = predictors + proj_list[i]
        print(predictors.shape)
        if True:
            labels = test_data_aSelected.cpu().detach().numpy()
            labels[labels==-1]=0  # this is being done to make it convenient for the xgbt classifier
            all_data = np.concatenate([predictors,labels], axis=1)
        else:
            # fitting a regression model to see the predictive power of the representation for alertsn
            all_data = np.concatenate([predictors,test_data_aSelected[:,-1].reshape(len(test_data_aSelected),1)], axis=1)
        shuffle_index = torch.randperm(n=all_data.shape[0]).numpy()
        all_data1 = all_data[shuffle_index]

        upto_test_idx = int(0.2 * len(all_data1))
        test_all = all_data1[:upto_test_idx]
        train_all = all_data1[upto_test_idx:]


        import xgboost as xgb
        from scipy.stats.stats import pearsonr
        from sklearn.metrics import r2_score, average_precision_score, roc_auc_score
        from sklearn.model_selection import GridSearchCV

        if True:
            if True:
                xgb_model = xgb.XGBClassifier()
                xgb_model_alertid = xgb.XGBClassifier(objective='multi:softmax')
                clf1_overall_rel =  GridSearchCV(xgb_model,{"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}, cv=3,verbose=1)
                clf1_overall_int =  GridSearchCV(xgb_model,{"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}, cv=3,verbose=1)
                clf1_overall_alertid =  GridSearchCV(xgb_model_alertid,{"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}, cv=3,verbose=1)

                clf1_overall_rel.fit(train_all[:, :-3], train_all[:, -3])
                pred_y_test_oRel = clf1_overall_rel.best_estimator_.predict_proba(test_all[:, :-3])
                acc_oRel = clf1_overall_rel.score(test_all[:, :-3], test_all[:,-3])

                clf1_overall_int.fit(train_all[:, :-3], train_all[:, -2])
                pred_y_test_oInter = clf1_overall_int.best_estimator_.predict_proba(test_all[:, :-3])
                acc_oInter = clf1_overall_int.score(test_all[:, :-3], test_all[:,-2])

                # TODO: the alert association part by predicting alert id in a multiclass setup
                # clf1_overall_alertid.fit(train_all[:, :-3], train_all[:, -1])
                # pred_y_test_AlertId = clf1_overall_alertid.best_estimator_.predict_proba(test_all[:, :-3])
                # acc_AlertId = clf1_overall_alertid.score(test_all[:, :-3], test_all[:,-1])

            else:
                xgb_model = xgb.XGBRegressor(random_state=42)
                reg1 = GridSearchCV(xgb_model,{"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}, cv=3,verbose=1,)
                # reg1 = GridSearchCV(xgb_model,{"max_depth": [4], "n_estimators": [50], "learning_rate": [0.01]}, cv=2,verbose=1,)
                reg1.fit(train_all[:,:-1], train_all[:,-1])
                pred_y_test = reg1.best_estimator_.predict(test_all[:, :-1])
        else:
            if True:
                clf = xgb.XGBClassifier(n_estimators=346, max_depth=8, learning_rate=0.001, random_state=42)
                clf.fit(train_all[:, :-1], train_all[:, -1])
                pred_y_test = clf.predict_proba(test_all[:, :-1])
                acc = clf.score(test_all[:, :-1], test_all[:,-1])
            else:
                reg = xgb.XGBRegressor(n_estimators=346, max_depth=8, learning_rate=0.001, random_state=42)
                reg.fit(train_all[:, :-1], train_all[:, -1])
                pred_y_test = reg.predict(test_all[:, :-1])

        if True:
            auprc_oRel = average_precision_score(np.array(test_all[:,-3]), np.array(pred_y_test_oRel)[:, 1])
            auroc_oRel = roc_auc_score(np.array(test_all[:,-3]), np.array(pred_y_test_oRel)[:, 1])
            alerts_vs_repr_dict_oRel = {'outcome_rate_test':test_all[:,-3].mean(), 'acc': acc_oRel, 'auprc': auprc_oRel, 'auroc': auroc_oRel, 'train_sample_size':len(train_all), 'test_sample_size':len(test_all)}

            auprc_oInter = average_precision_score(np.array(test_all[:,-2]), np.array(pred_y_test_oInter)[:, 1])
            auroc_oInter = roc_auc_score(np.array(test_all[:,-2]), np.array(pred_y_test_oInter)[:, 1])
            alerts_vs_repr_dict_oInter = {'outcome_rate_test':test_all[:,-2].mean(), 'acc': acc_oInter, 'auprc': auprc_oInter, 'auroc': auroc_oInter, 'train_sample_size':len(train_all), 'test_sample_size':len(test_all)}
        else:
            # this is for the earlier method which uses all the methods
            corr_value = np.round(pearsonr(np.array(test_all[:,-1]), np.array(pred_y_test))[0], 3)
            cor_p_value = np.round(pearsonr(np.array(test_all[:,-1]), np.array(pred_y_test))[1], 3)
            print(" alerts prediction with correlation ", corr_value, ' and corr p value of ', cor_p_value)
            r2value = r2_score(np.array(test_all[:,-1]), np.array(pred_y_test))  # inbuilt function also exists for R2
            print(" Value of R2 ", r2value)
            alerts_vs_repr_dict = {'corr':corr_value, 'corr_p_value':cor_p_value, 'r2_value':r2value, 'train_sample_size':len(train_all), 'test_sample_size':len(test_all)}

        print(" Association between representation and overall relevance \n", alerts_vs_repr_dict_oRel)
        print(" Association between representation and overall intervention \n",alerts_vs_repr_dict_oInter)

        return alerts_vs_repr_dict_oRel, alerts_vs_repr_dict_oInter