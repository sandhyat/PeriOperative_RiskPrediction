import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from  utils import take_per_row

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

def tp_noneffect(func, x, **kwargs):
    tp = x[..., -1:]
    x = func(x[..., :-1], **kwargs)   # removing the artificial time density dimension before computing freq_mix
    #breakpoint()
    return torch.cat([x, tp], dim=-1)


def freq_mix(x, rate=0.5, dim=1):
    x_f = torch.fft.fft(x, dim=dim)
    #breakpoint()
    m = torch.cuda.FloatTensor(x_f.shape).uniform_() < rate
    amp = abs(x_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal = x_f.real.masked_fill(m, 0)
    fimag = x_f.imag.masked_fill(m, 0)

    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2 = x[b_idx]  #this is another random training instance whose same frequency components are used for replacement and then mixing.
    x2_f = torch.fft.fft(x2, dim=dim)

    m = torch.bitwise_not(m)
    freal2 = x2_f.real.masked_fill(m, 0)
    fimag2 = x2_f.imag.masked_fill(m, 0)

    freal += freal2
    fimag += fimag2

    x_f = torch.complex(freal, fimag)
    #breakpoint()
    x = torch.abs(torch.fft.ifft(x_f, dim=dim))
    return x

class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):  # not clear about the ''pool'' part here
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output

class TSEncoder_tUrl(nn.Module):
    def __init__(self, input_dims, output_dims, medid_embedDim=0, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.med_embed_dim = medid_embedDim
        self.embed_layer_units = nn.Embedding(num_embeddings=218 + 1,embedding_dim=1,padding_idx=0)
        self.embed_layer_med_ids = nn.Embedding(num_embeddings=92 + 1,embedding_dim=self.med_embed_dim,padding_idx=0)  # there is NA already in the dictionary that can be used a padding token
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims

        # breakpoint()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0  # B x T x input_dims
        # the following if condition is very adhoc and needs updating, 40 because there are 13*3 med columns at max, 70 because combined of flowsheets and meds is more than 69 flowsheets
        if x.shape[2] > 70:
            temp = x[:,:,-39:] # med part of the ts
            med_ids_embed = self.embed_layer_med_ids(temp[:,:,:13].long()) # B x T x input_dims x Med_id_dim
            med_doses = temp[:,:,13:26]
            med_units_embed = self.embed_layer_units(temp[:,:,-13:].long()) # B x T x input_dims x Med_id_dim
            med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
            temp = torch.sum(med_combined_embed, 2) # B x T x Med_id_dim
            x = torch.concat([x[:,:,:-39],temp], dim=2)
        if x.shape[2] < 40:
            med_ids_embed = self.embed_layer_med_ids(x[:,:,:13].long()) # B x T x input_dims x Med_id_dim
            med_doses = x[:,:,13:26]
            med_units_embed = self.embed_layer_units(x[:,:,-13:].long()) # B x T x input_dims x Med_id_dim
            # breakpoint()
            med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
            x = torch.sum(med_combined_embed, 2) # B x T x Med_id_dim
            #
            # x = self.embed_layer_med_ids(x.long()) # B x T x input_dims x Med_id_dim
            # x = torch.sum(x, 2) # B x T x Med_id_dim
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask # reason for masking the latent vector is that the value range for time series is possibly unbounded and it is impossible to find a special token for raw data
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        breakpoint()
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x


class TSEncoder_f_tUrl(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims], # this is creating a list with first depth number of elements equaling to hidden_dims and the alst element as output_dims
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)  # this is new: the projection head to reconstruct the masked input

    def forward(self, x, mask=None):  # x: B x T x input_dims
        if isinstance(x, dict):
            input_all = copy.deepcopy(x)
            m = x['mask']
            x = x['data'] if 'data' in x.keys() else x['x']
        else:
            input_all = copy.deepcopy(x)
            m = x[..., -(x.shape[-1] // 2):]
            x = x[..., :-(x.shape[-1] // 2)]

        t = x[..., -1]
        x = x[..., :-1]

        if mask == 'mask_last':
            nan_mask = ~x.isnan().any(axis=-1)

        x[torch.isnan(x)], m[torch.isnan(m)] = 0, 0

        # whole series without missing
        if self.training:
            x_whole = self.input_fc(x * input_all['mask_origin']) # the mask_origin is to flag the nan missing values in the input data
            x_whole = x_whole.transpose(1, 2)
            x_whole = self.feature_extractor(x_whole)  # B x Ch x T
            x_whole = x_whole.transpose(1, 2)  # B x T x Co
            x_whole = self.repr_dropout(x_whole)

        # recon mask part
        if self.training:
            x_interp = self.input_fc(x * input_all['mask']) # the mask here is for the time reconstruction bit
            x_interp = x_interp.transpose(1, 2)
            x_interp = self.feature_extractor(x_interp)  # B x Ch x T
            x_interp = x_interp.transpose(1, 2)  # B x T x Co
            x_interp = self.repr_dropout(x_interp)

        if mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            mask &= nan_mask
            x[~mask] = 0

        x = self.input_fc(x * m)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)  # B x Ch x T
        x = x.transpose(1, 2)  # B x T x Co
        x = self.repr_dropout(x)

        if self.training:
            return x_whole, self.interphead(x_interp)
        else:
            return x


class TSEncoder_m_tUrl(nn.Module):
    def __init__(self, input_dims, output_dims, medid_embedDim=0, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.med_embed_dim = medid_embedDim
        self.embed_layer_units = nn.Embedding(num_embeddings=218 + 1, embedding_dim=1, padding_idx=0)
        self.embed_layer_med_ids = nn.Embedding(num_embeddings=92 + 1, embedding_dim=self.med_embed_dim,
                                                padding_idx=0)  # there is NA already in the dictionary that can be used a padding token
        self.input_fc = nn.Linear(self.med_embed_dim, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims

        # breakpoint()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0  # B x T x input_dims
        # # the following if condition is very adhoc and needs updating, 40 because there are 13*3 med columns at max, 70 because combined of flowsheets and meds is more than 69 flowsheets
        # if x.shape[2] > 70:
        #     temp = x[:, :, -39:]  # med part of the ts
        #     med_ids_embed = self.embed_layer_med_ids(temp[:, :, :13].long())  # B x T x input_dims x Med_id_dim
        #     med_doses = temp[:, :, 13:26]
        #     med_units_embed = self.embed_layer_units(temp[:, :, -13:].long())  # B x T x input_dims x Med_id_dim
        #     med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
        #     temp = torch.sum(med_combined_embed, 2)  # B x T x Med_id_dim
        #     x = torch.concat([x[:, :, :-39], temp], dim=2)
        # if x.shape[2] < 40:
        med_ids_embed = self.embed_layer_med_ids(x[:, :, :13].long())  # B x T x input_dims x Med_id_dim
        med_doses = x[:, :, 13:26]
        med_units_embed = self.embed_layer_units(x[:, :, -13:].long())  # B x T x input_dims x Med_id_dim
        # breakpoint()
        med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
        x = torch.sum(med_combined_embed, 2)  # B x T x Med_id_dim
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask # reason for masking the latent vector is that the value range for time series is possibly unbounded and it is impossible to find a special token for raw data
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        # breakpoint()
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

class TSEncoder_m_alt_tUrl(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='continuous'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.embed_layer_units = nn.Embedding(num_embeddings=218 + 1, embedding_dim=1, padding_idx=0)
        # self.embed_layer_med_ids = nn.Embedding(num_embeddings=92 + 1, embedding_dim=self.med_embed_dim,
        #                                         padding_idx=0)  # there is NA already in the dictionary that can be used a padding token
        self.input_fc = nn.Linear(self.input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)  # this is new: the projection head to reconstruct the masked input

    def forward(self, x0, idx_aug=None, num_elem_aug=None, mask=None, freq_mix_func = None):  # x: B x T x input_dims  # idx_aug and num_elem_aug is to keeo the cropping consistent across all the modalities
        # freq_mix is the func to go into the frequency space
        if isinstance(x0, list):
            input_all = copy.deepcopy(x0)
            m = x0[1]  # second element of list
            x0 = x0[0]
        else:
            input_all = copy.deepcopy(x0)
            m = x0[..., -(x0.shape[-1] // 2):]
            x0 = x0[..., :-(x0.shape[-1] // 2)]

        t = x0[..., -1]
        x0 = x0[..., :-1]
        m = m[:,:,13:26]
        if self.training:
            all_indx = idx_aug[:, None] + np.arange(num_elem_aug)
            x0 = x0[torch.arange(all_indx.shape[0])[:,None], all_indx]  # the cropping bit
            meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(device=x0.device) # this tensor will contain the product of dose and the unit embeddings
            med_ids = x0[:, :, :13] * input_all[2][:, :, :13]
            med_dose = x0[:, :, 13:26] * input_all[2][:, :, 13:26]
            med_unit = x0[:, :, -13:] * input_all[2][:, :, -13:]
        else:
            meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(device=x0.device) # this tensor will contain the product of dose and the unit embeddings
            med_ids = x0[:, :, :13] * m
            med_dose = x0[:, :, 13:26] * m
            med_unit = x0[:, :, -13:] * m

        med_units_embed = self.embed_layer_units(med_unit.long())  # B x T x input_dims x Med_id_dim
        # med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
        # x = torch.sum(med_combined_embed, 2)  # B x T x Med_id_dim

        # num_non_zero_vals_meds = torch.nonzero(med_ids).shape[0]


        # # not an efficient way to do this probably if this strategy works then the initial data will be updated accordingly
        # for i in range(num_non_zero_vals_meds):
        #     meds_tensor[torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], int(
        #         med_ids[torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], torch.nonzero(med_dose)[i][2]])] = \
        #         med_dose[torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], torch.nonzero(med_dose)[i][2]] * \
        #         med_units_embed.squeeze()[
        #             torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], torch.nonzero(med_dose)[i][2]]
        # breakpoint()

        # meds_tensor_accum = torch.zeros_like(meds_tensor)  # Create an accumulator tensor
        #
        # for i in range(num_non_zero_vals_meds):
        #     index0 = (torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], int(med_ids[torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], torch.nonzero(med_dose)[i][2]]))
        #     index = (torch.nonzero(med_dose)[i][0], torch.nonzero(med_dose)[i][1], torch.nonzero(med_dose)[i][2])
        #
        #     # Perform the multiplication without modifying the original tensor
        #     value_to_add = med_dose[index] * med_units_embed.squeeze()[index]
        #
        #     # Accumulate the values in the accumulator tensor
        #     meds_tensor_accum[index0] = value_to_add
        #
        # # Assign the accumulated values to meds_tensor
        # meds_tensor = meds_tensor_accum

        # meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(device=x0.device)

        # Create indices for non-zero values
        # breakpoint()
        nonzero_indices = torch.nonzero(med_dose)
        if nonzero_indices == []:
            print("Needs debugging")
            breakpoint()
        try:
            # Perform the multiplication without modifying the original tensor
            if nonzero_indices.ndim !=3:  # this is to accomodate the single example case
                values_to_add = med_dose[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * med_units_embed[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].squeeze()
            else:
                values_to_add = med_dose[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * med_units_embed.squeeze()[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
        except IndexError:
            print("Needs debugging")
            breakpoint()

        # Use advanced indexing to update the meds_tensor  ; THIS IS THE ERROR LINE
        meds_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1],med_ids[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].long()] = values_to_add
        # not using the freq_mix augmentation in the meds
        cropped_Aug = meds_tensor

        if mask == 'mask_last':
            nan_mask = ~cropped_Aug.isnan().any(axis=-1)

        cropped_Aug[torch.isnan(cropped_Aug)], m[torch.isnan(m)] = 0, 0
        try:
            x = self.input_fc(cropped_Aug)  # B x T x Ch
        except RuntimeError:
            print("Needs debugging")
            breakpoint()

        # whole series without missing
        if self.training:
            # meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(
            #     device=x0.device)  # this tensor will contain the product of dose and the unit embeddings
            # med_ids_cl = x0[:, :, :13]*input_all[2][:, :, :13]
            # med_dose_cl = x0[:, :, 13:26]*input_all[2][:, :, 13:26]
            # med_unit_cl = x0[:, :, -13:]*input_all[2][:, :, -13:]
            #
            # med_units_embed = self.embed_layer_units(med_unit_cl.long())  # B x T x input_dims x Med_id_dim
            # # Create indices for non-zero values
            # nonzero_indices = torch.nonzero(med_dose_cl)
            # try:
            #     # Perform the multiplication without modifying the original tensor
            #     if nonzero_indices.ndim != 3:  # this is to accomodate the single example case
            #         values_to_add = x0[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * \
            #                         med_units_embed[
            #                             nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].squeeze()
            #     else:
            #         values_to_add = x0[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * \
            #                         med_units_embed.squeeze()[
            #                             nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
            # except IndexError:
            #     print("Needs debugging")
            #     breakpoint()
            # # Use advanced indexing to update the meds_tensor
            # meds_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1], med_ids_cl[
            #     nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].long()] = values_to_add
            #
            # if (idx_aug is not None) and (freq_mix_func is not None):
            #     # Random cropping augmentation for medications
            #     cropped_Aug = tp_noneffect(freq_mix, take_per_row(meds_tensor, idx_aug, num_elem_aug), rate=0.5)
            # elif (idx_aug is not None):
            #     cropped_Aug = take_per_row(meds_tensor, idx_aug, num_elem_aug)
            # else:
            #     cropped_Aug = meds_tensor
            #
            # if mask == 'mask_last':
            #     nan_mask = ~cropped_Aug.isnan().any(axis=-1)
            #
            # cropped_Aug[torch.isnan(cropped_Aug)], m[torch.isnan(m)] = 0, 0
            # x_whole = self.input_fc(cropped_Aug)  # B x T x Ch
            # x_whole = self.input_fc(x * input_all[2][:, :, 13:26]) # the mask_origin is to flag the nan missing values in the input data
            x_whole = x.transpose(1, 2)
            x_whole = self.feature_extractor(x_whole)  # B x Ch x T
            x_whole = x_whole.transpose(1, 2)  # B x T x Co
            x_whole = self.repr_dropout(x_whole)

        # recon mask part
        # if self.training:
        #     meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(
        #         device=x0.device)  # this tensor will contain the product of dose and the unit embeddings
        #     med_ids_r = x0[:, :, :13]*input_all[1][:, :, :13]
        #     med_dose_r = x0[:, :, 13:26]*input_all[1][:, :, 13:26]
        #     med_unit_r = x0[:, :, -13:]*input_all[1][:, :, -13:]
        #
        #     med_units_embed = self.embed_layer_units(med_unit_r.long())  # B x T x input_dims x Med_id_dim
        #     # Create indices for non-zero values
        #     nonzero_indices = torch.nonzero(med_dose_r)
        #     try:
        #         # Perform the multiplication without modifying the original tensor
        #         if nonzero_indices.ndim != 3:  # this is to accomodate the single example case
        #             values_to_add = x0[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * \
        #                             med_units_embed[
        #                                 nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].squeeze()
        #         else:
        #             values_to_add = x0[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * \
        #                             med_units_embed.squeeze()[
        #                                 nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
        #     except IndexError:
        #         print("Needs debugging")
        #         breakpoint()
        #     # Use advanced indexing to update the meds_tensor
        #     meds_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1], med_ids_r[
        #         nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].long()] = values_to_add
        #
        #     if (idx_aug is not None) and (freq_mix_func is not None):
        #         # Random cropping augmentation for medications
        #         cropped_Aug = tp_noneffect(freq_mix, take_per_row(meds_tensor, idx_aug, num_elem_aug), rate=0.5)
        #     elif (idx_aug is not None):
        #         cropped_Aug = take_per_row(meds_tensor, idx_aug, num_elem_aug)
        #     else:
        #         cropped_Aug = meds_tensor
        #
        #     if mask == 'mask_last':
        #         nan_mask = ~cropped_Aug.isnan().any(axis=-1)
        #
        #     cropped_Aug[torch.isnan(cropped_Aug)], m[torch.isnan(m)] = 0, 0
        #     x_interp = self.input_fc(cropped_Aug)  # B x T x Ch
        #     # x_interp = self.input_fc(x * input_all[1][:, :, 13:26]) # the mask here is for the time reconstruction bit
        #     x_interp = x_interp.transpose(1, 2)
        #     x_interp = self.feature_extractor(x_interp)  # B x Ch x T
        #     x_interp = x_interp.transpose(1, 2)  # B x T x Co
        #     x_interp = self.repr_dropout(x_interp)

        if mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            mask &= nan_mask
            x[~mask] = 0


        # x = self.input_fc(x * m)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)  # B x Ch x T
        x = x.transpose(1, 2)  # B x T x Co
        x = self.repr_dropout(x)

        if self.training:
            return x_whole  #, self.interphead(x_interp), cropped_Aug
        else:
            return x