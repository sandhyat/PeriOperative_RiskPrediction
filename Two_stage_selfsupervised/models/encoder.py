import torch
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

class TSEncoder(nn.Module):
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


class TSEncoder_f(nn.Module):
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
        #     med_ids_embed = self.embed_layer_med_ids(x[:, :, :13].long())  # B x T x input_dims x Med_id_dim
        #     med_doses = x[:, :, 13:26]
        #     med_units_embed = self.embed_layer_units(x[:, :, -13:].long())  # B x T x input_dims x Med_id_dim
        #     # breakpoint()
        #     med_combined_embed = torch.mul(torch.mul(med_units_embed, med_doses.unsqueeze(-1)), med_ids_embed)
        #     x = torch.sum(med_combined_embed, 2)  # B x T x Med_id_dim
        #     #
        #     # x = self.embed_layer_med_ids(x.long()) # B x T x input_dims x Med_id_dim
        #     # x = torch.sum(x, 2) # B x T x Med_id_dim
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

class TSEncoder_a(nn.Module):
    def __init__(self, input_dims, output_dims,id_len, id_emb_dim, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = id_emb_dim  # this is alert_Embeddim
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.embed_layer_alerttype = nn.Embedding(num_embeddings=id_len + 1, embedding_dim=id_emb_dim, padding_idx=0)
        if False:
            # attention layer to aggregate the medication at any given time; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, id_emb_dim))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            self.attention_alert_pertime = nn.MultiheadAttention(id_emb_dim, 5,batch_first=True)

        self.input_fc = nn.Linear(self.input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims], # this is creating a list with first depth number of elements equaling to hidden_dims and the alst element as output_dims
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, idx_aug=None, num_elem_aug=None, mask=None):  # x: B x T x input_dims
        # breakpoint()
        alerttype = x[:, :,:, -1]
        others = x[:,:,:,:-1]

        alerttype_embed = self.embed_layer_alerttype(alerttype.long())

        alerts_embedded = alerttype_embed*others[:,:,:,0].unsqueeze(-1) + alerttype_embed*others[:,:,:,1].unsqueeze(-1) # 0 index is relevance and 1 is intervention
        # breakpoint()
        if False:
            idxname = alerts_embedded.shape
            alert_combined_embed_pertime = alerts_embedded.contiguous().view(idxname[0] * idxname[1], *idxname[2:])
            bs = alert_combined_embed_pertime.size(0)
            alert_combined_embed_pertime = torch.cat([self.class_token.expand(bs, -1, -1), alert_combined_embed_pertime],dim=1)
            encodings, _ = self.attention_alert_pertime(alert_combined_embed_pertime, alert_combined_embed_pertime, alert_combined_embed_pertime)
            alert_combined_embed0 = encodings[:, 0, :]
            alerts_final = alert_combined_embed0.contiguous().view(idxname[0], idxname[1], idxname[-1])
        else:
            alerts_final = torch.sum(alerts_embedded, 2)

        if idx_aug is not None:
            # Random cropping augmentation for alerts
            cropped_Aug = take_per_row(alerts_final, idx_aug, num_elem_aug)
        else:
            cropped_Aug = alerts_final

        nan_mask = ~cropped_Aug.isnan().any(axis=-1)
        cropped_Aug[~nan_mask] = 0  # B x T x input_dims
        # breakpoint()
        try:
            x = self.input_fc(cropped_Aug)  # B x T x Ch
        except:
            print('debug')
            breakpoint()
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

class TSEncoder_m_alt(nn.Module):
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

    def forward(self, x0, idx_aug=None, num_elem_aug=None, mask=None):  # x: B x T x input_dims  # idx_aug and num_elem_aug is to keeo the cropping consistent across all the modalities
        meds_tensor = torch.zeros(x0.shape[0], x0.shape[1], self.input_dims).to(device=x0.device) # this tensor will contain the product of dose and the unit embeddings
        med_ids = x0[:, :, :13]
        med_dose = x0[:, :, 13:26]
        med_unit = x0[:, :, -13:]
        # breakpoint()
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
        try:
            # Perform the multiplication without modifying the original tensor
            if nonzero_indices.ndim !=3:  # this is to accomodate the single example case
                values_to_add = med_dose[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * med_units_embed[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].squeeze()
            else:
                values_to_add = med_dose[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * med_units_embed.squeeze()[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
        except IndexError:
            print("Needs debugging")
            breakpoint()

        # # Perform the multiplication without modifying the original tensor
        # values_to_add = x0[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] * \
        #                 med_units_embed.squeeze()[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

        # Use advanced indexing to update the meds_tensor
        meds_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1],med_ids[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].long()] = values_to_add


        # breakpoint()

        if idx_aug is not None:
            # Random cropping augmentation for medications
            cropped_Aug = take_per_row(meds_tensor, idx_aug, num_elem_aug)
        else:
            cropped_Aug = meds_tensor

        nan_mask = ~cropped_Aug.isnan().any(axis=-1)
        cropped_Aug[~nan_mask] = 0  # B x T x input_dims

        x = self.input_fc(cropped_Aug)  # B x T x Ch

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

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        # breakpoint()

        return x

class TSEncoder_m(nn.Module):
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

