"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: SubTab class, the framework used for self-supervised representation learning.

Modified for use in a multiview setup by Sandhya Tripathi

"""

import gc
import itertools
import os
import functools
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
th.autograd.set_detect_anomaly(True)

class Outcome_net(nn.Module):
    def __init__(self, input_dim_outcome, output_rep_dim, hidden_dim=32):
        super().__init__()
        self.hid1 = nn.Linear(in_features=input_dim_outcome, out_features=hidden_dim)
        self.hid2 = nn.Linear(in_features=hidden_dim, out_features=output_rep_dim)

    def forward(self, outcomes):
        out = F.relu(self.hid1(outcomes.to(th.float32)))
        out = F.relu(self.hid2(out))

        return out


class Outcome_decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        """
        input_dim : since this is a decoder network, this is the dimension of  encoder's output
        output_dim : as SubTab paper showed that reconstructing the whole  table is more effective, this is the total number of columns in the initial dataset (outcome data size)
        """
        super().__init__()
        self.hid1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.hid2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, latent, dt_dict):
        out = F.relu(self.hid1(latent.to(th.float32)))
        dt_dict_up = dict(zip(['int', 'float'], list(dt_dict.values()))) # this is very adhoc
        out = self.hid2(out)
        out = th.cat([F.relu(out[:,:dt_dict_up['float']]),th.sigmoid(out[:,dt_dict_up['float']:])],dim=1)
        # breakpoint()
        # out = F.relu() # ideally should be a mix of sigmoid and relu since the input data is a mix of boolean and float outcomes

        return out

class SubTab(nn.Module):
    def __init__(
            self,
            outcome_dims,
            outcome_rep_dims,
            n_subsets=4,
            overlap_ratio=0.75,
            noise_type='swap_noise',
            masking_ratio=0.3,
            noise_level=0.1,
            agg_method='mean'
    ):
        """
        Model: Trains an Encoder with a Projection network, using SubTab framework.

        outcome_dims (int) : number of columns in the full dataset
        outcome_rep_dims (int) : dimension of the outcome embedding space
        n_subsets (int) : number of subsets in which the dataset can be partitioned
        overlap_ratio (float) : A ratio [0,1) that defines how many features are overlapped between subsets.
        noise_type (str) : Type of noise to add to. Choices: swap_noise, gaussian_noise, zero_out
        masking_ratio (float): Percentage of the feature to add noise to
        noiselevel (float) : stdev defined for Gaussian noise
        agg_method (str) : method to aggreagte the subset reps, options:  mean (default), sum, max, min, and concat
        """

        super().__init__()

        self.dim_total = outcome_dims
        self.nsubsets = n_subsets
        self.overlap = overlap_ratio
        self.noisetype = noise_type
        self.mask_ratio = masking_ratio
        self.noiselevel = noise_level
        self.rep_dim = outcome_rep_dims
        self.agg_method = agg_method

        # Compute the shrunk size of input dimension
        n_column_subset = int(outcome_dims/n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap_ratio*n_column_subset)


        self.encoder = Outcome_net(n_column_subset + n_overlap, outcome_rep_dims)
        self.decoder = Outcome_decoder(outcome_rep_dims, outcome_dims)

        # Two-Layer Projection Network
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(outcome_rep_dims, outcome_rep_dims)
        # Last linear layer for final projection
        self.linear_layer2 = nn.Linear(outcome_rep_dims, outcome_rep_dims)

    def forward(self, x, dt_dict):
        # breakpoint()
        # Forward pass on Encoder
        latent = self.encoder(x)
        # Forward pass on Projection
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = F.leaky_relu(self.linear_layer1(latent))
        # Apply final linear layer
        z = self.linear_layer2(z)
        # Do L2 normalization
        z = F.normalize(z, p=2, dim=1)

        x_recon = self.decoder(latent, dt_dict)

        return z, latent, x_recon

    def train_epoch(self, x):
        # Generate subsets with added noise -- labels are not used
        x_tilde_list = self.subset_generator(x, mode="train")
        # Get combinations of subsets [(x1, x2), (x1, x3)...]
        x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)

        return x_tilde_list


    def get_combinations_of_subsets(self, x_tilde_list):
        """Generate a list of combinations of subsets from the list of subsets

        Args:
            x_tilde_list (list): List of subsets e.g. [x1, x2, x3, ...]

        Returns:
            (list): A list of combinations of subsets e.g. [(x1, x2), (x1, x3), ...]

        """

        # Compute combinations of subsets [(x1, x2), (x1, x3)...]
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        # List to store the concatenated subsets
        concatenated_subsets_list = []

        # Go through combinations
        for (xi, xj) in subset_combinations:
            # Concatenate xi, and xj, and turn it into a tensor
            Xbatch = self.process_batch(xi, xj)
            # Add it to the list
            concatenated_subsets_list.append(Xbatch)

        # Return the list of combination of subsets
        return concatenated_subsets_list

    def mask_generator(self, p_m, x):
        """Generate mask vector."""
        mask = np.random.binomial(1, p_m, x.shape)
        return mask

    def subset_generator(self, x, mode="test", skip=[-1]):
        """Generate subsets and adds noise to them

        Args:
            x (np.ndarray): Input data, which is divded to the subsets
            mode (bool): Indicates whether we are training a model, or testing it
            skip (list): List of integers, showing which subsets to skip when training the model

        Returns:
            (list): A list of np.ndarrays, each of which is one subset
            (list): A list of lists, each list of which indicates locations of added noise in a subset

        """

        n_subsets = self.nsubsets
        n_column = self.dim_total
        overlap = self.overlap
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)

        # Get the range over the number of features
        column_idx = list(range(n_column))
        # Permute the order of subsets to avoid any bias during training. The order is unchanged at the test time.
        permuted_order = np.random.permutation(n_subsets) if mode == "train" else range(n_subsets)
        # Pick subset of columns (equivalent of cropping)
        subset_column_idx_list = []

        # In test mode, we are using all subsets, i.e. [-1]. But, we can choose to skip some subsets during training.
        skip = [-1] if mode == "test" else skip

        # Generate subsets.
        for i in permuted_order:
            # If subset is in skip, don't include it in training. Otherwise, continue.
            if i not in skip:
                if i == 0:
                    start_idx = 0
                    stop_idx = n_column_subset + n_overlap
                else:
                    start_idx = i * n_column_subset - n_overlap
                    stop_idx = (i + 1) * n_column_subset
                # Get the subset
                subset_column_idx_list.append(column_idx[start_idx:stop_idx])

        # Add a dummy copy if there is a single subset
        if len(subset_column_idx_list) == 1:
            subset_column_idx_list.append(subset_column_idx_list[0])

        # Get subset of features to create list of cropped data
        x_tilde_list = []
        # breakpoint()
        for subset_column_idx in subset_column_idx_list:
            x_bar = x[:, subset_column_idx]
            # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
            x_bar_noisy = self.generate_noisy_xbar(x_bar)

            # Generate binary mask
            p_m = self.mask_ratio
            mask = np.random.binomial(1, p_m, x_bar.shape)

            # Replace selected x_bar features with the noisy ones
            x_bar = x_bar * (1 - mask) + x_bar_noisy * mask

            # Add the subset to the list
            x_tilde_list.append(x_bar)

        return x_tilde_list

    def generate_noisy_xbar(self, x):
        """Generates noisy version of the samples x

        Args:
            x (np.ndarray): Input data to add noise to

        Returns:
            (np.ndarray): Corrupted version of input x

        """
        # Dimensions
        no, dim = x.shape

        # Get noise type
        noise_type = self.noisetype
        noise_level = self.noiselevel

        # Initialize corruption array
        x_bar = np.zeros([no, dim])

        # Randomly (and column-wise) shuffle data
        if noise_type == "swap_noise":
            for i in range(dim):
                idx = np.random.permutation(no)
                x_bar[:, i] = x[idx, i]
        # Elif, overwrite x_bar by adding Gaussian noise to x
        elif noise_type == "gaussian_noise":
            x_bar = x + np.random.normal(0, noise_level, x.shape)
        else:
            x_bar = x_bar

        return x_bar


    def process_batch(self, xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def _tensor(self, data):
        """Turns numpy arrays to torch tensors"""
        if type(data).__module__ == np.__name__:
            data = th.from_numpy(data)
        return data.to(self.device).float()

    def aggregate(self, latent_list):
        """Aggregates the latent representations of subsets to obtain joint representation

        Args:
            latent_list (list[torch.FloatTensor]): List of latent variables, one for each subset

        Returns:
            (torch.FloatTensor): Joint representation

        """
        # Initialize the joint representation
        latent = None

        # Aggregation of latent representations
        if self.agg_method == "mean":
            latent = sum(latent_list) / len(latent_list)
        elif self.agg_method == "sum":
            latent = sum(latent_list)
        elif self.agg_method == "concat":
            latent = th.cat(latent_list, dim=-1)
        elif self.agg_method == "max":
            latent = functools.reduce(th.max, latent_list)
        elif self.agg_method == "min":
            latent = functools.reduce(th.min, latent_list)
        else:
            print("Proper aggregation option is not provided. Please check the config file.")
            exit()

        return latent

    @property
    def device(self):
        return next(self.parameters()).device


