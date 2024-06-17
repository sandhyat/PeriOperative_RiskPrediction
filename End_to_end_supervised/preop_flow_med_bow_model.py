"""
all classes take the preops, embedded preoperative text as input. The fllowsheets and medications have already been imputed.

1) Class TS_lstm_Med_index : LSTM models for meds and flowsheets.
Option of attention on home meds  for aggregation for a fixed patient. Option of attention on intraop med at any given time after the product. This is followed by an LSTM.

2) Class TS_Attention_Med_index : Option of attention over intraop meds at any given time point or summing over the med embedding dim after the product of unit, dose and med id.
1D CNN and Max pooling for both medication and flowsheets followed by concatenating them. Appending the preop and embedded text initialization before feeding into MultiheadAttention.

3) Class TS_Transformer_Med_index : Similar to TS_Attention_Med_index except the MultiHead attention is replaced by off the shelf transformer encoder layer from pytorch.

4) Class TS_lstm_Med_index_Sum_AndORFirst_Last : Attention over the combined output of Medication and flowsheet to eliminate the time dimension and possibly attending to important time points..
"""

import numpy as np
import torch
import torch.nn as nn

class TS_Transformer_Med_index(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.e_dim_flow = kwargs["e_dim_med_ids"]
        self.lstm_hid = kwargs["lstm_hid"]
        self.lstm_flow_hid = kwargs["lstm_flow_hid"]
        self.preops_init_med = kwargs["preops_init_med"]
        self.bilstm_med = kwargs["bilstm_med"]
        self.Att_MedAgg = kwargs["Att_MedAgg"]
        self.AttMedAgg_Heads = kwargs["AttMedAgg_Heads"]
        self.linear_out = kwargs["linear_out"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]
        self.drp_rate = kwargs["p_rows"]
        self.drp_rate_time = kwargs["p_time"]
        self.drp_rate_time_flow = kwargs["p_flow"]
        self.drp_final= kwargs["p_final"]
        self.lstm_num_layers = kwargs["lstm_num_layers"]
        self.lstm_flow_num_layers = kwargs["lstm_flow_num_layers"]
        self.preops_init_flow = kwargs["preops_init_flow"]
        self.bilstm_flow = kwargs["bilstm_flow"]
        self.binary_outcome = kwargs["binary"]
        self.hidden = kwargs["hidden_units"]
        self.hidden_final = kwargs["hidden_units_final"]
        self.hidden_depth = kwargs["hidden_depth"]
        self.input_size = kwargs["input_shape"]
        self.hidden_bow = kwargs["hidden_units_bow"]
        self.hidden_final_bow = kwargs["hidden_units_final_bow"]
        self.hidden_depth_bow = kwargs["hidden_depth_bow"]
        self.input_size_bow = kwargs["input_shape_bow"]
        self.input_flow_size = kwargs['num_flowsheet_feat']
        self.finalBN = kwargs["finalBN"]
        self.preopsWDRateL2 = kwargs["weight_decay_preopsL2"]
        self.preopsWDRateL1 = kwargs["weight_decay_preopsL1"]
        self.bowWDRateL2 = kwargs["weight_decay_bowL2"]
        self.bowWDRateL1 = kwargs["weight_decay_bowL1"]
        self.lstmMedWDRateL2 = kwargs["weight_decay_LSTMmedL2"]
        self.lstmFlowWDRateL2 = kwargs["weight_decay_LSTMflowL2"]
        self.weightInt = kwargs["weightInt"]
        self.AttTS_depth = kwargs['AttTS_depth']
        self.Att_TSComb = kwargs["Att_TSComb"]
        self.AttTS_Heads = kwargs["AttTS_Heads"]
        self.cnn_before_Att = kwargs['cnn_before_Att']
        self.kernel_size = kwargs['kernel_size_conv']
        self.stride = kwargs['stride_conv']
        self.ats_dropout = kwargs['ats_dropout']

        """ static preops bit """
        self.hidden_layers = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
        ## sizes for subsequent layers
        hidensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype(
            'int64')
        for thisindex in range(len(hidensizes) - 1):
            self.hidden_layers.append(
                nn.Linear(in_features=hidensizes[thisindex], out_features=hidensizes[thisindex + 1]))

        """ static preops (BOW) bit """
        self.hidden_layers_bow = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_bow.append(nn.Linear(in_features=self.input_size_bow, out_features=self.hidden_bow))
        ## sizes for subsequent layers
        hidensizes_bow = np.ceil(np.linspace(start=self.hidden_bow, stop=self.hidden_final_bow, num=self.hidden_depth_bow)).astype(
            'int64')
        for thisindex in range(len(hidensizes_bow) - 1):
            self.hidden_layers_bow.append(
                nn.Linear(in_features=hidensizes_bow[thisindex], out_features=hidensizes_bow[thisindex + 1]))


        """ Med TS bit """
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

        if self.Att_MedAgg == True:
            # attention layer to aggregate the medication at any given time; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.e_dim_med_ids % self.AttMedAgg_Heads == 0:
                self.attention_med_pertime = nn.MultiheadAttention(self.e_dim_med_ids, self.AttMedAgg_Heads, batch_first=True)
            else:
                raise Exception("embedding dim needs to be divisible by number of attention heads")


        if self.cnn_before_Att == True:
            # cnn on meds to consolidate the time information; first (in forward step) need to swap the time and feature axis to be compatible with the conv1d api
            self.conv1d_meds = nn.Conv1d(self.e_dim_med_ids, self.e_dim_med_ids, self.kernel_size, self.stride)
            self.maxpool_meds = nn.MaxPool1d(self.kernel_size,self.stride)

        # self.name_lstm = nn.LSTM(input_size=self.e_dim_words, hidden_size=self.lstm_hid, num_layers=1, batch_first=True)
        self.drop_layer1 = nn.Dropout(p=self.drp_rate)  ## this is element-wise drop in doses
        self.drop_layer2 = nn.Dropout(p=self.drp_final)  ## this is element-wise drop in doses

        if (self.preops_init_med == True) and (self.preops_init_flow == True):
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_hid (med) is too restrictive
            self.preopbow_init_layer = nn.Linear(in_features=self.hidden_final+self.hidden_final_bow, out_features=self.e_dim_med_ids+self.e_dim_flow)
            # self.bow_init_layer = nn.Linear(in_features=self.hidden_final_bow, out_features=self.e_dim_med_ids+self.input_flow_size)

        # if self.lstm_num_layers > 1:
        #     self.pre_final_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.e_dim_med_ids,
        #         hidden_size=self.lstm_hid,
        #         num_layers=self.lstm_num_layers,
        #         dropout=self.drp_rate_time,
        #         bidirectional=self.bilstm_med)
        # else:
        #     self.pre_final_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.e_dim_med_ids,
        #         hidden_size=self.lstm_hid,
        #         num_layers=self.lstm_num_layers,
        #         bidirectional=self.bilstm_med
        #     )
        # self.pre_final_lstm = self.pre_final_lstm.float()  # this was required to solve the inconsistencies between the datatypes TODO this suggests something wrong - it should already be floats (its inputs are floats)
        # if self.finalBN == 1:
        #     self.bnlayer1 = nn.BatchNorm1d(self.hidden_final)



        """ TS flowsheet bit """

        # if self.preops_init_flow == True:
        #     # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_flow_hid is too restrictive
        #     self.preop_init_layer_flow = nn.Linear(in_features=self.hidden_final, out_features=self.input_flow_size)
        #     self.bow_init_layer_flow = nn.Linear(in_features=self.hidden_final_bow, out_features=self.input_flow_size)

        # if self.lstm_flow_num_layers > 1:
        #     self.flowsheet_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.input_flow_size,
        #         hidden_size=self.lstm_flow_hid,
        #         num_layers=self.lstm_flow_num_layers,
        #         dropout=self.drp_rate_time_flow)
        # else:
        #     self.flowsheet_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.input_flow_size,
        #         hidden_size=self.lstm_flow_hid,
        #         num_layers=self.lstm_flow_num_layers)

        if self.cnn_before_Att == True:
            self.conv1d_flow = nn.Conv1d(self.input_flow_size, self.input_flow_size, self.kernel_size,self.stride)
            self.maxpool_flow = nn.MaxPool1d(self.kernel_size, self.stride)

        # this is needed because when combining the meds and flowsheets for attention, meds have been emmbedded but flowsheets are raw
        self.flowsheet_projection_layer  = nn.Linear(in_features= self.input_flow_size, out_features= self.e_dim_flow)



        """ Static + TS """
        if self.Att_TSComb ==True:
            """ Attention over Meds and flowsheets """

            # positional encoding initialization; this is fixed encoding
            max_len = 1000
            self.dropout_ats = nn.Dropout(self.ats_dropout)
            self.P = torch.zeros((1, max_len, self.e_dim_med_ids+self.e_dim_flow))
            X = torch.arange(max_len, dtype=torch.float32).reshape(
                -1, 1) / torch.pow(10000, torch.arange(
                0, (self.e_dim_med_ids+self.e_dim_flow), 2, dtype=torch.float32) / (self.e_dim_med_ids+self.e_dim_flow))
            self.P[:, :, 0::2] = torch.sin(X)
            self.P[:, :, 1::2] = torch.cos(X)

            self.class_token_TS = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids + self.e_dim_flow))  # "global information"
            # breakpoint()
            # self.class_token_TS = self.preop_init_layer_med + self.preop_init_layer_flow
            torch.nn.init.normal_(self.class_token_TS, std=0.08)
            if (self.e_dim_med_ids + self.e_dim_flow) % self.AttTS_Heads == 0:
                self.encoderlayer = nn.TransformerEncoderLayer(d_model=self.e_dim_med_ids + self.e_dim_flow, nhead=self.AttTS_Heads, dim_feedforward=512)
                self.transformerencoder = nn.TransformerEncoder(self.encoderlayer, num_layers=self.AttTS_depth)
                # self.attention_TS_layers = torch.nn.ModuleList()
                # for i  in range(self.AttTS_depth):
                #     self.attention_TS_layers.append(nn.MultiheadAttention(self.e_dim_med_ids + self.e_dim_flow, self.AttTS_Heads,
                #                                                    batch_first=True))
            else:
                raise Exception("model dim needs to be divisible by number of attention heads")


        self.pre_final_linear = nn.Linear(in_features=self.e_dim_med_ids + self.e_dim_flow +self.hidden_final + self.hidden_final_bow, out_features=self.hidden_final)
        self.final_linear = nn.Linear(in_features=self.hidden_final, out_features=self.linear_out)

        if(self.weightInt):
            self._reinitialize()


    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif ('hidden' in name) or ('linear' in name):
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    # class PositionalEncoding(nn.Module):  # @save
    #     """Positional encoding."""
    #
    #     def __init__(self, num_hiddens, dropout, max_len=1000):
    #         super().__init__()
    #         self.dropout = nn.Dropout(dropout)
    #         # Create a long enough P
    #         self.P = torch.zeros((1, max_len, num_hiddens))
    #         X = torch.arange(max_len, dtype=torch.float32).reshape(
    #             -1, 1) / torch.pow(10000, torch.arange(
    #             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    #         self.P[:, :, 0::2] = torch.sin(X)
    #         self.P[:, :, 1::2] = torch.cos(X)
    #
    #     def forward(self, X):
    #         X = X + self.P[:, :X.shape[1], :].to(X.device)
    #         return self.dropout(X)

    def forward(self, preops, seq_len, bow_data, med_ids, dose, units, flowsheets,  ):

        """ preops MLP"""
        my_device = next(self.hidden_layers[0].parameters()).device
        preop_path = torch.nn.ReLU()(self.hidden_layers[0](preops) )
        if(len(self.hidden_layers) > 1) :
          for thisindex in range(len(self.hidden_layers)-1 ):
            preop_path = torch.nn.ReLU()(self.hidden_layers[thisindex+1](preop_path) )

        preops_l1_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=1) for i in range(self.hidden_depth)]
        preops_l2_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=2) for i in range(self.hidden_depth)]

        """ preops (BOW) MLP"""
        bow_path = torch.nn.ReLU()(self.hidden_layers_bow[0](bow_data) )
        if(len(self.hidden_layers_bow) > 1) :
          for thisindex in range(len(self.hidden_layers_bow)-1 ):
            bow_path = torch.nn.ReLU()(self.hidden_layers_bow[thisindex+1](bow_path) )

        bow_l1_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=1) for i in range(self.hidden_depth_bow)]
        bow_l2_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=2) for i in range(self.hidden_depth_bow)]

        """ meds """

        # dropout before embedding layer. Using doses since we multiply the doses with units annd embedding. It circumvents the problem of  passing dropped out  values to the embedding layer
        # this is okay even when the there is dropout is included in the lstm with more than one layer because the dropout is applied to the outpput of lstm layer.
        dose = self.drop_layer1(dose)

        # embedding the names and units
        units_embedding = self.embed_layer_units(units.long())
        med_ids_temp_embedding = self.embed_layer_med_ids(med_ids.long())

        if True:
            med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        else:
            med_combined_embed = torch.cat((dose.unsqueeze(-1), units_embedding, med_ids_temp_embedding), 3)  # was used as an alternative

        if self.Att_MedAgg == True:  # attention for aggregation instead of summing over the medication dimension
            idxname = med_combined_embed.shape
            med_combined_embed_pertime = med_combined_embed.contiguous().view(idxname[0] * idxname[1], *idxname[2:])
            bs = med_combined_embed_pertime.size(0)
            med_combined_embed_pertime = torch.cat([self.class_token.expand(bs, -1, -1), med_combined_embed_pertime], dim=1)
            encodings, _ = self.attention_med_pertime(med_combined_embed_pertime,med_combined_embed_pertime,med_combined_embed_pertime)
            med_combined_embed0 = encodings[:,0,:]
            med_combined_embed = med_combined_embed0.contiguous().view(idxname[0], idxname[1], idxname[-1])
        else: # summing the values across the medication dimension
            med_combined_embed = torch.sum(med_combined_embed, 2)

        # breakpoint()
        if self.cnn_before_Att == True:
            #cnn on meds; need to swap the axis first
            med_combined_embed = torch.transpose(med_combined_embed, 1,2)
            meds_aftercnn = self.conv1d_meds(med_combined_embed)
            meds_aftercnn = self.maxpool_meds(meds_aftercnn)
            meds_aftercnn = torch.transpose(meds_aftercnn, 1,2) # need to get back to batch, time, feature shape
            med_combined_embed = meds_aftercnn

        """ flowsheets """

        if self.cnn_before_Att == True:
            flowsheets0 = torch.transpose(flowsheets, 1,2)
            flowsheets_aftercnn = self.conv1d_flow(flowsheets0)
            flowsheets_aftercnn = self.maxpool_flow(flowsheets_aftercnn)
            flowsheets_aftercnn = torch.transpose(flowsheets_aftercnn, 1,2)
            flowsheets_embedded = self.flowsheet_projection_layer(flowsheets_aftercnn)
        else:
            flowsheets_embedded = self.flowsheet_projection_layer(flowsheets)

        # initializing the hidden state with preop and bow mlp's output
        if (self.preops_init_med == True) and (self.preops_init_flow == True):
            init_token_TS_preop_bow = self.preopbow_init_layer(torch.concat((preop_path, bow_path),1))
            init_token_TS_preop_bow = torch.unsqueeze(init_token_TS_preop_bow,1)

        """ concatenate the hidden from mlp and lstm """

        if self.Att_TSComb ==True:
            # attention based concatenation
            # breakpoint()
            bs = med_combined_embed.size(0)
            # final_for_TEncoder = torch.cat((init_token_TS_preop_bow, torch.cat((med_combined_embed, flowsheets_embedded), 2)),1)  # the init token does the part of cls
            final_for_TEncoder = torch.cat((self.class_token_TS.expand(bs, -1, -1), torch.cat((med_combined_embed, flowsheets_embedded), 2)),1)  # the init token does the part of cls

            # positional encoding part
            final_for_TEncoder = final_for_TEncoder + self.P[:, :final_for_TEncoder.shape[1], :].to(final_for_TEncoder.device)

            attTS_path = self.transformerencoder(final_for_TEncoder)

            final_for_mlp = attTS_path[:, 0, :]
        else:
            # final_for_mlp = torch.cat((preop_path, seq_len.to(device).unsqueeze(-1), prefinal_for_mlp_fl), 1)
            final_for_mlp = torch.cat((prefinal_for_mlp, bow_path, preop_path, prefinal_for_mlp_fl), 1)  #

        final_for_mlp = torch.cat((final_for_mlp, bow_path, preop_path, prefinal_for_mlp_fl), 1)  #appending the preops and bow later
        final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

        if self.finalBN == 1:
            final_for_mlp = self.bnlayer1(final_for_mlp)

        outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp) )
        if self.binary_outcome == 1:
          outcome_pred = torch.sigmoid(outcome_pred)

        # total weight decay regularization loss
        reg_wd = self.preopsWDRateL1* sum(preops_l1_reg_loss) + self.preopsWDRateL2* sum(preops_l2_reg_loss) + \
                 self.bowWDRateL1* sum(bow_l1_reg_loss) + self.bowWDRateL2* sum(bow_l2_reg_loss)
                 # +self.lstmMedWDRateL2* sum(meds_l2_reg_loss) + self.lstmFlowWDRateL2 * sum(flowsheet_l2_reg_loss)

        return outcome_pred, reg_wd

class TS_Attention_Med_index(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.e_dim_flow = kwargs["e_dim_med_ids"]
        self.lstm_hid = kwargs["lstm_hid"]
        self.lstm_flow_hid = kwargs["lstm_flow_hid"]
        self.preops_init_med = kwargs["preops_init_med"]
        self.bilstm_med = kwargs["bilstm_med"]
        self.Att_MedAgg = kwargs["Att_MedAgg"]
        self.AttMedAgg_Heads = kwargs["AttMedAgg_Heads"]
        self.linear_out = kwargs["linear_out"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]
        self.drp_rate = kwargs["p_rows"]
        self.drp_rate_time = kwargs["p_time"]
        self.drp_rate_time_flow = kwargs["p_flow"]
        self.drp_final= kwargs["p_final"]
        self.lstm_num_layers = kwargs["lstm_num_layers"]
        self.lstm_flow_num_layers = kwargs["lstm_flow_num_layers"]
        self.preops_init_flow = kwargs["preops_init_flow"]
        self.bilstm_flow = kwargs["bilstm_flow"]
        self.binary_outcome = kwargs["binary"]
        self.hidden = kwargs["hidden_units"]
        self.hidden_final = kwargs["hidden_units_final"]
        self.hidden_depth = kwargs["hidden_depth"]
        self.input_size = kwargs["input_shape"]
        self.hidden_bow = kwargs["hidden_units_bow"]
        self.hidden_final_bow = kwargs["hidden_units_final_bow"]
        self.hidden_depth_bow = kwargs["hidden_depth_bow"]
        self.input_size_bow = kwargs["input_shape_bow"]
        self.input_flow_size = kwargs['num_flowsheet_feat']
        self.finalBN = kwargs["finalBN"]
        self.preopsWDRateL2 = kwargs["weight_decay_preopsL2"]
        self.preopsWDRateL1 = kwargs["weight_decay_preopsL1"]
        self.bowWDRateL2 = kwargs["weight_decay_bowL2"]
        self.bowWDRateL1 = kwargs["weight_decay_bowL1"]
        self.lstmMedWDRateL2 = kwargs["weight_decay_LSTMmedL2"]
        self.lstmFlowWDRateL2 = kwargs["weight_decay_LSTMflowL2"]
        self.weightInt = kwargs["weightInt"]
        self.AttTS_depth = kwargs['AttTS_depth']
        self.Att_TSComb = kwargs["Att_TSComb"]
        self.AttTS_Heads = kwargs["AttTS_Heads"]
        self.kernel_size = kwargs['kernel_size_conv']
        self.stride = kwargs['stride_conv']
        self.ats_dropout = kwargs['ats_dropout']

        """ static preops bit """
        self.hidden_layers = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
        ## sizes for subsequent layers
        hidensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype(
            'int64')
        for thisindex in range(len(hidensizes) - 1):
            self.hidden_layers.append(
                nn.Linear(in_features=hidensizes[thisindex], out_features=hidensizes[thisindex + 1]))

        """ static preops (BOW) bit """
        self.hidden_layers_bow = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_bow.append(nn.Linear(in_features=self.input_size_bow, out_features=self.hidden_bow))
        ## sizes for subsequent layers
        hidensizes_bow = np.ceil(np.linspace(start=self.hidden_bow, stop=self.hidden_final_bow, num=self.hidden_depth_bow)).astype(
            'int64')
        for thisindex in range(len(hidensizes_bow) - 1):
            self.hidden_layers_bow.append(
                nn.Linear(in_features=hidensizes_bow[thisindex], out_features=hidensizes_bow[thisindex + 1]))


        """ Med TS bit """
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

        if self.Att_MedAgg == True:
            # attention layer to aggregate the medication at any given time; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.e_dim_med_ids % self.AttMedAgg_Heads == 0:
                self.attention_med_pertime = nn.MultiheadAttention(self.e_dim_med_ids, self.AttMedAgg_Heads, batch_first=True)
            else:
                raise Exception("embedding dim needs to be divisible by number of attention heads")


        # cnn on meds to consolidate the time information; first (in forward step) need to swap the time and feature axis to be compatible with the conv1d api
        self.conv1d_meds = nn.Conv1d(self.e_dim_med_ids, self.e_dim_med_ids, self.kernel_size, self.stride)
        self.maxpool_meds = nn.MaxPool1d(self.kernel_size,self.stride)

        # self.name_lstm = nn.LSTM(input_size=self.e_dim_words, hidden_size=self.lstm_hid, num_layers=1, batch_first=True)
        self.drop_layer1 = nn.Dropout(p=self.drp_rate)  ## this is element-wise drop in doses
        self.drop_layer2 = nn.Dropout(p=self.drp_final)  ## this is element-wise drop in doses

        if (self.preops_init_med == True) and (self.preops_init_flow == True):
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_hid (med) is too restrictive
            self.preopbow_init_layer = nn.Linear(in_features=self.hidden_final+self.hidden_final_bow, out_features=self.e_dim_med_ids+self.e_dim_flow)
            # self.bow_init_layer = nn.Linear(in_features=self.hidden_final_bow, out_features=self.e_dim_med_ids+self.input_flow_size)

        # if self.lstm_num_layers > 1:
        #     self.pre_final_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.e_dim_med_ids,
        #         hidden_size=self.lstm_hid,
        #         num_layers=self.lstm_num_layers,
        #         dropout=self.drp_rate_time,
        #         bidirectional=self.bilstm_med)
        # else:
        #     self.pre_final_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.e_dim_med_ids,
        #         hidden_size=self.lstm_hid,
        #         num_layers=self.lstm_num_layers,
        #         bidirectional=self.bilstm_med
        #     )
        # self.pre_final_lstm = self.pre_final_lstm.float()  # this was required to solve the inconsistencies between the datatypes TODO this suggests something wrong - it should already be floats (its inputs are floats)
        # if self.finalBN == 1:
        #     self.bnlayer1 = nn.BatchNorm1d(self.hidden_final)



        """ TS flowsheet bit """

        # if self.preops_init_flow == True:
        #     # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_flow_hid is too restrictive
        #     self.preop_init_layer_flow = nn.Linear(in_features=self.hidden_final, out_features=self.input_flow_size)
        #     self.bow_init_layer_flow = nn.Linear(in_features=self.hidden_final_bow, out_features=self.input_flow_size)

        # if self.lstm_flow_num_layers > 1:
        #     self.flowsheet_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.input_flow_size,
        #         hidden_size=self.lstm_flow_hid,
        #         num_layers=self.lstm_flow_num_layers,
        #         dropout=self.drp_rate_time_flow)
        # else:
        #     self.flowsheet_lstm = nn.LSTM(
        #         batch_first=True,
        #         input_size=self.input_flow_size,
        #         hidden_size=self.lstm_flow_hid,
        #         num_layers=self.lstm_flow_num_layers)

        self.conv1d_flow = nn.Conv1d(self.input_flow_size, self.input_flow_size, self.kernel_size,self.stride)
        self.maxpool_flow = nn.MaxPool1d(self.kernel_size, self.stride)

        # this is needed because when combining the meds and flowsheets for attention, meds have been emmbedded but flowsheets are raw
        self.flowsheet_projection_layer  = nn.Linear(in_features= self.input_flow_size, out_features= self.e_dim_flow)



        """ Static + TS """
        if self.Att_TSComb ==True:
            """ Attention over Meds and flowsheets """

            # positional encoding initialization; this is fixed encoding
            max_len = 1000
            self.dropout_ats = nn.Dropout(self.ats_dropout)
            self.P = torch.zeros((1, max_len, self.e_dim_med_ids+self.e_dim_flow))
            X = torch.arange(max_len, dtype=torch.float32).reshape(
                -1, 1) / torch.pow(10000, torch.arange(
                0, (self.e_dim_med_ids+self.e_dim_flow), 2, dtype=torch.float32) / (self.e_dim_med_ids+self.e_dim_flow))
            self.P[:, :, 0::2] = torch.sin(X)
            self.P[:, :, 1::2] = torch.cos(X)

            # self.class_token_TS = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids + self.e_dim_flow))  # "global information"
            # breakpoint()
            # self.class_token_TS = self.preop_init_layer_med + self.preop_init_layer_flow
            # torch.nn.init.normal_(self.class_token_TS, std=0.08)
            if (self.e_dim_med_ids + self.e_dim_flow) % self.AttTS_Heads == 0:
                self.attention_TS_layers = torch.nn.ModuleList()
                for i  in range(self.AttTS_depth):
                    self.attention_TS_layers.append(nn.MultiheadAttention(self.e_dim_med_ids + self.e_dim_flow, self.AttTS_Heads,
                                                                   batch_first=True))
            else:
                raise Exception("model dim needs to be divisible by number of attention heads")


        self.pre_final_linear = nn.Linear(in_features=self.e_dim_med_ids + self.e_dim_flow , out_features=self.hidden_final)
        self.final_linear = nn.Linear(in_features=self.hidden_final, out_features=self.linear_out)

        if(self.weightInt):
            self._reinitialize()


    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif ('hidden' in name) or ('linear' in name):
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    # class PositionalEncoding(nn.Module):  # @save
    #     """Positional encoding."""
    #
    #     def __init__(self, num_hiddens, dropout, max_len=1000):
    #         super().__init__()
    #         self.dropout = nn.Dropout(dropout)
    #         # Create a long enough P
    #         self.P = torch.zeros((1, max_len, num_hiddens))
    #         X = torch.arange(max_len, dtype=torch.float32).reshape(
    #             -1, 1) / torch.pow(10000, torch.arange(
    #             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    #         self.P[:, :, 0::2] = torch.sin(X)
    #         self.P[:, :, 1::2] = torch.cos(X)
    #
    #     def forward(self, X):
    #         X = X + self.P[:, :X.shape[1], :].to(X.device)
    #         return self.dropout(X)

    def forward(self, preops, seq_len, bow_data, med_ids, dose, units, flowsheets,  ):

        """ preops MLP"""
        my_device = next(self.hidden_layers[0].parameters()).device
        preop_path = torch.nn.ReLU()(self.hidden_layers[0](preops) )
        if(len(self.hidden_layers) > 1) :
          for thisindex in range(len(self.hidden_layers)-1 ):
            preop_path = torch.nn.ReLU()(self.hidden_layers[thisindex+1](preop_path) )

        preops_l1_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=1) for i in range(self.hidden_depth)]
        preops_l2_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=2) for i in range(self.hidden_depth)]

        """ preops (BOW) MLP"""
        bow_path = torch.nn.ReLU()(self.hidden_layers_bow[0](bow_data) )
        if(len(self.hidden_layers_bow) > 1) :
          for thisindex in range(len(self.hidden_layers_bow)-1 ):
            bow_path = torch.nn.ReLU()(self.hidden_layers_bow[thisindex+1](bow_path) )

        bow_l1_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=1) for i in range(self.hidden_depth_bow)]
        bow_l2_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=2) for i in range(self.hidden_depth_bow)]

        """ meds """

        # dropout before embedding layer. Using doses since we multiply the doses with units annd embedding. It circumvents the problem of  passing dropped out  values to the embedding layer
        # this is okay even when the there is dropout is included in the lstm with more than one layer because the dropout is applied to the outpput of lstm layer.
        dose = self.drop_layer1(dose)

        # embedding the names and units
        units_embedding = self.embed_layer_units(units.long())
        med_ids_temp_embedding = self.embed_layer_med_ids(med_ids.long())

        if True:
            med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        else:
            med_combined_embed = torch.cat((dose.unsqueeze(-1), units_embedding, med_ids_temp_embedding), 3)  # was used as an alternative

        if self.Att_MedAgg == True:  # attention for aggregation instead of summing over the medication dimension
            idxname = med_combined_embed.shape
            med_combined_embed_pertime = med_combined_embed.contiguous().view(idxname[0] * idxname[1], *idxname[2:])
            bs = med_combined_embed_pertime.size(0)
            med_combined_embed_pertime = torch.cat([self.class_token.expand(bs, -1, -1), med_combined_embed_pertime], dim=1)
            encodings, _ = self.attention_med_pertime(med_combined_embed_pertime,med_combined_embed_pertime,med_combined_embed_pertime)
            med_combined_embed0 = encodings[:,0,:]
            med_combined_embed = med_combined_embed0.contiguous().view(idxname[0], idxname[1], idxname[-1])
        else: # summing the values across the medication dimension
            med_combined_embed = torch.sum(med_combined_embed, 2)

        # breakpoint()
        #cnn on meds; need to swap the axis first
        med_combined_embed = torch.transpose(med_combined_embed, 1,2)
        meds_aftercnn = self.conv1d_meds(med_combined_embed)
        meds_aftercnn = self.maxpool_meds(meds_aftercnn)
        meds_aftercnn = torch.transpose(meds_aftercnn, 1,2) # need to get back to batch, time, feature shape
        med_combined_embed = meds_aftercnn

        """ flowsheets """

        flowsheets0 = torch.transpose(flowsheets, 1,2)
        flowsheets_aftercnn = self.conv1d_flow(flowsheets0)
        flowsheets_aftercnn = self.maxpool_flow(flowsheets_aftercnn)
        flowsheets_aftercnn = torch.transpose(flowsheets_aftercnn, 1,2)
        flowsheets_embedded = self.flowsheet_projection_layer(flowsheets_aftercnn)

        # initializing the hidden state with preop and bow mlp's output
        if (self.preops_init_med == True) and (self.preops_init_flow == True):
            init_token_TS_preop_bow = self.preopbow_init_layer(torch.concat((preop_path, bow_path),1))
            init_token_TS_preop_bow = torch.unsqueeze(init_token_TS_preop_bow,1)

        # packing unpacking the flowsheet data too for lstm
        # flowsheet_packed = torch.nn.utils.rnn.pack_padded_sequence(flowsheets, seq_len, batch_first=True, enforce_sorted=False)
        #
        # if self.preops_init_flow == True:
        #     lstm_output_fl, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed , (hidden_state_lstm_flow.contiguous(), hidden_state_lstm_flow.contiguous()))
        # else:
        #     lstm_output_fl, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed )
        # prefinal_for_mlp_fl = final_embed_comb_ht_f[-1]
        #
        # flowsheet_l2_reg_loss = [torch.norm(self.flowsheet_lstm.all_weights[i][0], p=2) +torch.norm(self.flowsheet_lstm.all_weights[i][1], p=2) for i in range(self.lstm_flow_num_layers)]

        # for attention concatenation
        # torch.nn.utils.rnn.pad_packed_sequence(lstm_output_fl, batch_first=True)[0]
        # breakpoint()

        """ concatenate the hidden from mlp and lstm """

        if self.Att_TSComb ==True:
            # attention based concatenation
            # breakpoint()
            final_for_mlp0 = torch.cat((init_token_TS_preop_bow, torch.cat((med_combined_embed, flowsheets_embedded), 2)),1)
            # bs = final_for_mlp0.size(0)
            # lstm_outout_TS_comb = torch.cat([self.class_token_TS.expand(bs, -1, -1), final_for_mlp0], dim=1)
            # position_encod = PositionalEncoding(final_for_mlp0.shape[-1],0)

            # print("final input size going into the attention models ", final_for_mlp0.shape)
            # positional encoding part
            final_for_mlp0 = final_for_mlp0 + self.P[:, :final_for_mlp0.shape[1], :].to(final_for_mlp0.device)
            final_for_mlp0 = self.dropout_ats(final_for_mlp0)

            lstm_outout_TS_comb = final_for_mlp0


            attTS_path, _  = self.attention_TS_layers[0](lstm_outout_TS_comb, lstm_outout_TS_comb,
                                                      lstm_outout_TS_comb)
            for i in range(self.AttTS_depth-1):
                attTS_path, _ = self.attention_TS_layers[i+1](attTS_path, attTS_path,
                                                      attTS_path)

            # encodings_TS1, _ = self.attention_TS1(lstm_outout_TS_comb, lstm_outout_TS_comb,
            #                                           lstm_outout_TS_comb)
            # encodings_TS2, _ = self.attention_TS2(encodings_TS1, encodings_TS1,
            #                                           encodings_TS1)
            # encodings_TS3, _ = self.attention_TS3(encodings_TS2, encodings_TS2,
            #                                           encodings_TS2)
            # final_for_mlp_TS = encodings_TS3[:, 0, :]
            final_for_mlp = attTS_path[:, 0, :]
            # final_for_mlp = torch.cat((final_for_mlp_TS, bow_path, preop_path), 1)  #
        else:
            # final_for_mlp = torch.cat((preop_path, seq_len.to(device).unsqueeze(-1), prefinal_for_mlp_fl), 1)
            final_for_mlp = torch.cat((prefinal_for_mlp, bow_path, preop_path, prefinal_for_mlp_fl), 1)  #


        final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

        if self.finalBN == 1:
            final_for_mlp = self.bnlayer1(final_for_mlp)

        outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp) )
        if self.binary_outcome == 1:
          outcome_pred = torch.sigmoid(outcome_pred)

        # total weight decay regularization loss
        reg_wd = self.preopsWDRateL1* sum(preops_l1_reg_loss) + self.preopsWDRateL2* sum(preops_l2_reg_loss) + \
                 self.bowWDRateL1* sum(bow_l1_reg_loss) + self.bowWDRateL2* sum(bow_l2_reg_loss)
                 # +self.lstmMedWDRateL2* sum(meds_l2_reg_loss) + self.lstmFlowWDRateL2 * sum(flowsheet_l2_reg_loss)

        return outcome_pred, reg_wd

class TS_lstm_Med_index(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.lstm_hid = kwargs["lstm_hid"]
        self.lstm_flow_hid = kwargs["lstm_flow_hid"]
        self.preops_init_med = kwargs["preops_init_med"]
        self.bilstm_med = kwargs["bilstm_med"]
        self.Att_MedAgg = kwargs["Att_MedAgg"]
        self.AttMedAgg_Heads = kwargs["AttMedAgg_Heads"]
        self.linear_out = kwargs["linear_out"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]
        self.drp_rate = kwargs["p_rows"]
        self.drp_rate_time = kwargs["p_time"]
        self.drp_rate_time_flow = kwargs["p_flow"]
        self.drp_final= kwargs["p_final"]
        self.lstm_num_layers = kwargs["lstm_num_layers"]
        self.lstm_flow_num_layers = kwargs["lstm_flow_num_layers"]
        self.preops_init_flow = kwargs["preops_init_flow"]
        self.bilstm_flow = kwargs["bilstm_flow"]
        self.binary_outcome = kwargs["binary"]
        self.hidden = kwargs["hidden_units"]
        self.hidden_final = kwargs["hidden_units_final"]
        self.hidden_depth = kwargs["hidden_depth"]
        self.input_size = kwargs["input_shape"]
        self.hidden_bow = kwargs["hidden_units_bow"]
        self.hidden_final_bow = kwargs["hidden_units_final_bow"]
        self.hidden_depth_bow = kwargs["hidden_depth_bow"]
        self.input_size_bow = kwargs["input_shape_bow"]
        self.Att_HM_Agg = kwargs['Att_HM_Agg']
        self.hidden_hm = kwargs["hidden_units_hm"]
        self.hidden_final_hm = kwargs["hidden_units_final_hm"]
        self.hidden_depth_hm = kwargs["hidden_depth_hm"]
        self.input_size_hm = kwargs["input_shape_hm"]
        self.input_flow_size = kwargs['num_flowsheet_feat']
        self.finalBN = kwargs["finalBN"]
        self.preopsWDRateL2 = kwargs["weight_decay_preopsL2"]
        self.preopsWDRateL1 = kwargs["weight_decay_preopsL1"]
        self.bowWDRateL2 = kwargs["weight_decay_bowL2"]
        self.bowWDRateL1 = kwargs["weight_decay_bowL1"]
        self.hmWDRateL2 = kwargs["weight_decay_hmL2"]
        self.hmWDRateL1 = kwargs["weight_decay_hmL1"]
        self.lstmMedWDRateL2 = kwargs["weight_decay_LSTMmedL2"]
        self.lstmFlowWDRateL2 = kwargs["weight_decay_LSTMflowL2"]
        self.weightInt = kwargs["weightInt"]
        self.group_start = kwargs['group_start_list']
        self.group_end = kwargs['group_end_list']


        ## NOTE: for the multiply structure, need to be conformable
        if self.e_dim_units is True:
            self.e_dim_units = self.e_dim_med_ids
        else:
            self.e_dim_units = 1

        def group_sum(multipliers, group_start, group_end):
            multipliers2 = torch.zeros_like(multipliers)
            for i in range(len(group_start)):
                multipliers2[group_start[i]: group_end[i] + 1] = multipliers[group_start[i]: group_end[i] + 1].clamp(0,
                                                                                                                     1).sum()
            return (multipliers2)

        class CustomEmbedding_ratio(torch.nn.Module):
            def __init__(self, num_embeddings, embedding_dim, group_start, group_end,p_idx_units):
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=p_idx_units)
                self.embedding_ratio = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=p_idx_units)
                self.embedding_ratio.weight.requires_grad = False
                self.group_start = group_start
                self.group_end = group_end

            def forward(self, x):
                self.embedding_ratio.weight = torch.nn.Parameter(
                    group_sum(torch.sigmoid(self.embedding.weight), self.group_start, self.group_end),
                    requires_grad=False)
                return torch.sigmoid(self.embedding(x)) / self.embedding_ratio(x)


        """ static preops bit """
        self.hidden_layers = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
        ## sizes for subsequent layers
        hidensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype(
            'int64')
        for thisindex in range(len(hidensizes) - 1):
            self.hidden_layers.append(
                nn.Linear(in_features=hidensizes[thisindex], out_features=hidensizes[thisindex + 1]))

        """ static preops (BOW) bit """
        self.hidden_layers_bow = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_bow.append(nn.Linear(in_features=self.input_size_bow, out_features=self.hidden_bow))
        ## sizes for subsequent layers
        hidensizes_bow = np.ceil(np.linspace(start=self.hidden_bow, stop=self.hidden_final_bow, num=self.hidden_depth_bow)).astype(
            'int64')
        for thisindex in range(len(hidensizes_bow) - 1):
            self.hidden_layers_bow.append(
                nn.Linear(in_features=hidensizes_bow[thisindex], out_features=hidensizes_bow[thisindex + 1]))

        if self.Att_HM_Agg == True:
            # attention layer to aggregate the home medication for each patient; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.input_size_hm ))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.input_size_hm  % self.AttMedAgg_Heads == 0:
                self.attention_hm = nn.MultiheadAttention(self.input_size_hm, self.AttMedAgg_Heads,
                                                                   batch_first=True)
            else:
                raise Exception("embedding dim needs to be divisible by number of attention heads")

        self.hidden_layers_hm = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_hm.append(nn.Linear(in_features=self.input_size_hm, out_features=self.hidden_hm))
        ## sizes for subsequent layers
        hiddensizes_hm = np.ceil(
            np.linspace(start=self.hidden_hm, stop=self.hidden_final_hm, num=self.hidden_depth_hm)).astype(
            'int64')
        for thisindex in range(len(hiddensizes_hm) - 1):
            self.hidden_layers_hm.append(
                nn.Linear(in_features=hiddensizes_hm[thisindex], out_features=hiddensizes_hm[thisindex + 1]))

        """ Med TS bit """
        self.embed_layer_med_ids = nn.Embedding(
            num_embeddings=self.v_med_ids + 1,
            embedding_dim=self.e_dim_med_ids,
            padding_idx=self.p_idx_med_ids
        )  # there is NA already in the dictionary that can be used a padding token

        self.embed_unit_med_comb = CustomEmbedding_ratio(self.v_units+1, self.e_dim_units, self.group_start, self.group_end, self.p_idx_units)


        if self.Att_MedAgg == True:
            # attention layer to aggregate the medication at any given time; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.e_dim_med_ids % self.AttMedAgg_Heads == 0:
                self.attention_med_pertime = nn.MultiheadAttention(self.e_dim_med_ids, self.AttMedAgg_Heads, batch_first=True)
            else:
                raise Exception("embedding ddim needs to be divisible by number of attention heads")


        # self.name_lstm = nn.LSTM(input_size=self.e_dim_words, hidden_size=self.lstm_hid, num_layers=1, batch_first=True)
        self.drop_layer1 = nn.Dropout(p=self.drp_rate)  ## this is element-wise drop in doses
        self.drop_layer2 = nn.Dropout(p=self.drp_final)  ## this is element-wise drop in doses

        if self.preops_init_med == True:
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_hid (med) is too restrictive
            self.preop_init_layer_med = nn.Linear(in_features=self.hidden_final, out_features=self.lstm_hid)

        if self.lstm_num_layers > 1:
            self.pre_final_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.e_dim_med_ids,
                hidden_size=self.lstm_hid,
                num_layers=self.lstm_num_layers,
                dropout=self.drp_rate_time,
                bidirectional=self.bilstm_med)
        else:
            self.pre_final_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.e_dim_med_ids,
                hidden_size=self.lstm_hid,
                num_layers=self.lstm_num_layers,
                bidirectional=self.bilstm_med
            )
        self.pre_final_lstm = self.pre_final_lstm.float()  # this was required to solve the inconsistencies between the datatypes TODO this suggests something wrong - it should already be floats (its inputs are floats)
        if self.finalBN == 1:
            self.bnlayer1 = nn.BatchNorm1d(self.hidden_final)


        """ TS flowsheet bit """

        if self.preops_init_flow == True:
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_flow_hid is too restrictive
            self.preop_init_layer_flow = nn.Linear(in_features=self.hidden_final, out_features=self.lstm_flow_hid)

        if self.lstm_flow_num_layers > 1:
            self.flowsheet_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.input_flow_size,
                hidden_size=self.lstm_flow_hid,
                num_layers=self.lstm_flow_num_layers,
                dropout=self.drp_rate_time_flow,
            bidirectional = self.bilstm_flow)
        else:
            self.flowsheet_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.input_flow_size,
                hidden_size=self.lstm_flow_hid,
                num_layers=self.lstm_flow_num_layers,
            bidirectional = self.bilstm_flow)

        """ Static + TS """
        self.pre_final_linear = nn.Linear(in_features=self.lstm_hid + self.hidden_final + self.hidden_final_bow  +self.hidden_final_hm + self.lstm_flow_hid , out_features=self.hidden_final)
        self.final_linear = nn.Linear(in_features=self.hidden_final, out_features=self.linear_out)

        if(self.weightInt):
            self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif ('hidden' in name) or ('linear' in name):
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, preops, seq_len, bow_data, home_meds, med_ids, dose, units, flowsheets,  ):
        #preops = preops.to(device)
        #bow_data = bow_data.to(device)
        #med_ids = med_ids.to(device)
        #dose = dose.to(device)
        #units = units.to(device)
        #flowsheets = flowsheets.to(device)

        """ preops MLP"""
        my_device = next(self.hidden_layers[0].parameters()).device
        preop_path = torch.nn.ReLU()(self.hidden_layers[0](preops) )
        if(len(self.hidden_layers) > 1) :
          for thisindex in range(len(self.hidden_layers)-1 ):
            preop_path = torch.nn.ReLU()(self.hidden_layers[thisindex+1](preop_path) )

        preops_l1_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=1) for i in range(self.hidden_depth)]
        preops_l2_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=2) for i in range(self.hidden_depth)]

        """ preops (BOW) MLP"""
        bow_path = torch.nn.ReLU()(self.hidden_layers_bow[0](bow_data) )
        if(len(self.hidden_layers_bow) > 1) :
          for thisindex in range(len(self.hidden_layers_bow)-1 ):
            bow_path = torch.nn.ReLU()(self.hidden_layers_bow[thisindex+1](bow_path) )

        bow_l1_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=1) for i in range(self.hidden_depth_bow)]
        bow_l2_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=2) for i in range(self.hidden_depth_bow)]

        """ preops (HOME MEDS) MLP"""

        if self.Att_HM_Agg == True:
            bs = home_meds.size(0)
            home_meds_att = torch.cat([self.class_token.expand(bs, -1, -1), home_meds],
                                                   dim=1)
            encodings, _ = self.attention_hm(home_meds_att, home_meds_att,
                                                      home_meds_att)
            home_meds_att0 = encodings[:, 0, :]
            hm_path = torch.nn.ReLU()(self.hidden_layers_hm[0](home_meds_att0))
        else:
            hm_path = torch.nn.ReLU()(self.hidden_layers_hm[0](home_meds))
        if (len(self.hidden_layers_hm) > 1):
            for thisindex in range(len(self.hidden_layers_hm) - 1):
                hm_path = torch.nn.ReLU()(self.hidden_layers_hm[thisindex + 1](hm_path))

        hm_l1_reg_loss = [torch.norm(self.hidden_layers_hm[i].weight.data, p=1) for i in range(self.hidden_depth_hm)]
        hm_l2_reg_loss = [torch.norm(self.hidden_layers_hm[i].weight.data, p=2) for i in range(self.hidden_depth_hm)]


        """ meds """

        # dropout before embedding layer. Using doses since we multiply the doses with units annd embedding. It circumvents the problem of  passing dropped out  values to the embedding layer
        # this is okay even when the there is dropout is included in the lstm with more than one layer because the dropout is applied to the outpput of lstm layer.
        dose = self.drop_layer1(dose)

        # embedding the names and units ; updated it based on med and unti combination
        # units_embedding = self.embed_layer_units(units.long())
        units_embedding = self.embed_unit_med_comb(units.long())

        med_ids_temp_embedding = self.embed_layer_med_ids(med_ids.long())

        if True:
            med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        else:
            med_combined_embed = torch.cat((dose.unsqueeze(-1), units_embedding, med_ids_temp_embedding), 3)  # was used as an alternative

        if self.Att_MedAgg == True:  # attention for aggregation instead of summing over the medication dimension
            idxname = med_combined_embed.shape
            med_combined_embed_pertime = med_combined_embed.contiguous().view(idxname[0] * idxname[1], *idxname[2:])
            bs = med_combined_embed_pertime.size(0)
            med_combined_embed_pertime = torch.cat([self.class_token.expand(bs, -1, -1), med_combined_embed_pertime], dim=1)
            encodings, _ = self.attention_med_pertime(med_combined_embed_pertime,med_combined_embed_pertime,med_combined_embed_pertime)
            med_combined_embed0 = encodings[:,0,:]
            med_combined_embed = med_combined_embed0.contiguous().view(idxname[0], idxname[1], idxname[-1])
        else: # summing the values across the medication dimension
            med_combined_embed = torch.sum(med_combined_embed, 2)

        # initializing the hidden state of med lstm with preop mlp's output
        if self.bilstm_med == True:
            D_med = 2  # D is the notation that is used by official docs too for emphasizing on bidirectional
        else:
            D_med = 1

        if self.preops_init_med == True:
            hidden_state_lstm_med = self.preop_init_layer_med(preop_path)
            hidden_state_lstm_med = torch.unsqueeze(hidden_state_lstm_med, 0)
            hidden_state_lstm_med = hidden_state_lstm_med.expand(D_med * self.lstm_num_layers, preop_path.size(0),
                                                                 self.lstm_hid)
        else:
            # Creation of cell state and hidden state for lstm
            hidden_state_lstm_med = torch.zeros(D_med * self.lstm_num_layers, med_combined_embed.size(0),
                                                self.lstm_hid).to( my_device)
            cell_state_lstm_med = torch.zeros(D_med * self.lstm_num_layers, med_combined_embed.size(0),
                                              self.lstm_hid).to(my_device)

            ## TODO: add an MLP of preops initializing the hidden state; done
            # Weights initialization
            torch.nn.init.xavier_normal_(hidden_state_lstm_med)
            torch.nn.init.xavier_normal_(cell_state_lstm_med)

        # breakpoint()
        # packing-unpacking for the final lstm too; here we will use the padding value as 0 (from the dose) because we padded over the time dimension
        med_combined_embed_packed = torch.nn.utils.rnn.pack_padded_sequence(med_combined_embed, seq_len,
                                                                            batch_first=True, enforce_sorted=False)

        if self.preops_init_med == True:
            # print("Init is working")
            _, (final_embed_comb_ht, final_embed_comb_ct) = self.pre_final_lstm(
                med_combined_embed_packed,
                (hidden_state_lstm_med.contiguous(), hidden_state_lstm_med.contiguous())
            )
        else:
            # print("Init is not working")
            _, (final_embed_comb_ht, final_embed_comb_ct) = self.pre_final_lstm(
                med_combined_embed_packed,
                (hidden_state_lstm_med.contiguous(), cell_state_lstm_med.contiguous())
            )
        # final linear layer with batch normalization before the mlp
        if self.bilstm_med == True:
            prefinal_for_mlp = final_embed_comb_ht[-2, :, :] + final_embed_comb_ht[-1, :, :]
        else:
            prefinal_for_mlp = final_embed_comb_ht[-1]

        meds_l2_reg_loss = [torch.norm(self.pre_final_lstm.all_weights[i][0], p=2) +torch.norm(self.pre_final_lstm.all_weights[i][1], p=2) for i in range(self.lstm_num_layers)]


        """  time series flowsheet """

        if self.bilstm_flow ==True:
            D_flow = 2 # D is the notation that is used by official docs too for emphasizing on bidirectional
        else:
            D_flow = 1

        # initializing the hidden state with preop mlp's output
        if self.preops_init_flow == True:
            hidden_state_lstm_flow = self.preop_init_layer_flow(preop_path)
            hidden_state_lstm_flow = torch.unsqueeze(hidden_state_lstm_flow,0)
            hidden_state_lstm_flow = hidden_state_lstm_flow.expand(D_flow*self.lstm_flow_num_layers, preop_path.size(0), self.lstm_flow_hid)

        # packing unpacking the flowsheet data too for lstm
        flowsheet_packed = torch.nn.utils.rnn.pack_padded_sequence(flowsheets, seq_len, batch_first=True, enforce_sorted=False)

        if self.preops_init_flow == True:
            _, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed , (hidden_state_lstm_flow.contiguous(), hidden_state_lstm_flow.contiguous()))
        else:
            _, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed )

        # final linear layer with batch normalization before the mlp
        if self.bilstm_flow == True:
            prefinal_for_mlp_fl = final_embed_comb_ht_f[-2, :, :] + final_embed_comb_ht_f[-1, :, :]
        else:
            prefinal_for_mlp_fl = final_embed_comb_ht_f[-1]

        flowsheet_l2_reg_loss = [torch.norm(self.flowsheet_lstm.all_weights[i][0], p=2) +torch.norm(self.flowsheet_lstm.all_weights[i][1], p=2) for i in range(self.lstm_flow_num_layers)]


        """ concatenate the hidden from mlp and lstm """
        # final_for_mlp = torch.cat((preop_path, seq_len.to(device).unsqueeze(-1), prefinal_for_mlp_fl), 1)
        final_for_mlp = torch.cat((prefinal_for_mlp, bow_path, hm_path, preop_path, prefinal_for_mlp_fl), 1)  #
        final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

        if self.finalBN == 1:
            final_for_mlp = self.bnlayer1(final_for_mlp)

        outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp) )
        if self.binary_outcome == 1:
          outcome_pred = torch.sigmoid(outcome_pred)

        # total weight decay regularization loss
        reg_wd = self.lstmMedWDRateL2* sum(meds_l2_reg_loss) + self.lstmFlowWDRateL2 * sum(flowsheet_l2_reg_loss) + \
                 self.preopsWDRateL1* sum(preops_l1_reg_loss) + self.preopsWDRateL2* sum(preops_l2_reg_loss) + \
                 self.bowWDRateL1* sum(bow_l1_reg_loss) + self.bowWDRateL2* sum(bow_l2_reg_loss) +\
                 self.hmWDRateL1 * sum(hm_l1_reg_loss) + self.hmWDRateL2 * sum(hm_l2_reg_loss)

        return outcome_pred, reg_wd

class TS_lstm_Med_index_Sum_AndORFirst_Last(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.lstm_hid = kwargs["lstm_hid"]
        self.lstm_flow_hid = kwargs["lstm_flow_hid"]
        self.preops_init_med = kwargs["preops_init_med"]
        self.bilstm_med = kwargs["bilstm_med"]
        self.Att_MedAgg = kwargs["Att_MedAgg"]
        self.AttMedAgg_Heads = kwargs["AttMedAgg_Heads"]
        self.linear_out = kwargs["linear_out"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]
        self.drp_rate = kwargs["p_rows"]
        self.drp_rate_time = kwargs["p_time"]
        self.drp_rate_time_flow = kwargs["p_flow"]
        self.drp_final = kwargs["p_final"]
        self.lstm_num_layers = kwargs["lstm_num_layers"]
        self.lstm_flow_num_layers = kwargs["lstm_flow_num_layers"]
        self.preops_init_flow = kwargs["preops_init_flow"]
        self.bilstm_flow = kwargs["bilstm_flow"]
        self.binary_outcome = kwargs["binary"]
        self.hidden = kwargs["hidden_units"]
        self.hidden_final = kwargs["hidden_units_final"]
        self.hidden_depth = kwargs["hidden_depth"]
        self.input_size = kwargs["input_shape"]
        self.hidden_bow = kwargs["hidden_units_bow"]
        self.hidden_final_bow = kwargs["hidden_units_final_bow"]
        self.hidden_depth_bow = kwargs["hidden_depth_bow"]
        self.input_size_bow = kwargs["input_shape_bow"]
        self.Att_HM_Agg = kwargs['Att_HM_Agg']
        self.hidden_hm = kwargs["hidden_units_hm"]
        self.hidden_final_hm = kwargs["hidden_units_final_hm"]
        self.hidden_depth_hm = kwargs["hidden_depth_hm"]
        self.input_size_hm = kwargs["input_shape_hm"]
        # self.hm_embed_size = kwargs['hm_embed_size']
        self.input_flow_size = kwargs['num_flowsheet_feat']
        self.finalBN = kwargs["finalBN"]
        self.preopsWDRateL2 = kwargs["weight_decay_preopsL2"]
        self.preopsWDRateL1 = kwargs["weight_decay_preopsL1"]
        self.bowWDRateL2 = kwargs["weight_decay_bowL2"]
        self.bowWDRateL1 = kwargs["weight_decay_bowL1"]
        self.hmWDRateL2 = kwargs["weight_decay_hmL2"]
        self.hmWDRateL1 = kwargs["weight_decay_hmL1"]
        self.lstmMedWDRateL2 = kwargs["weight_decay_LSTMmedL2"]
        self.lstmFlowWDRateL2 = kwargs["weight_decay_LSTMflowL2"]
        self.weightInt = kwargs["weightInt"]
        self.AttTS_depth = kwargs['AttTS_depth']
        self.Att_TSComb = kwargs["Att_TSComb"]
        self.AttTS_Heads = kwargs["AttTS_Heads"]
        self.MedSumAfterProd = kwargs['MedSumAfterProd']
        self.group_start = kwargs['group_start_list']
        self.group_end = kwargs['group_end_list']



        ## NOTE: for the multiply structure, need to be conformable
        if self.e_dim_units is True:
            self.e_dim_units = self.e_dim_med_ids
        else:
            self.e_dim_units = 1

        def group_sum(multipliers, group_start, group_end):
            multipliers2 = torch.zeros_like(multipliers)
            for i in range(len(group_start)):
                multipliers2[group_start[i]: group_end[i] + 1] = multipliers[group_start[i]: group_end[i] + 1].clamp(0,
                                                                                                                     1).sum()
            return (multipliers2)

        class CustomEmbedding_ratio(torch.nn.Module):
            def __init__(self, num_embeddings, embedding_dim, group_start, group_end,p_idx_units):
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=p_idx_units)
                self.embedding_ratio = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=p_idx_units)
                self.embedding_ratio.weight.requires_grad = False
                self.group_start = group_start
                self.group_end = group_end

            def forward(self, x):
                self.embedding_ratio.weight = torch.nn.Parameter(
                    group_sum(torch.sigmoid(self.embedding.weight), self.group_start, self.group_end),
                    requires_grad=False)
                return torch.sigmoid(self.embedding(x)) / self.embedding_ratio(x)

        self.embed_unit_med_comb = CustomEmbedding_ratio(self.v_units+1, self.e_dim_units, self.group_start, self.group_end, self.p_idx_units)

        """ static preops bit """
        self.hidden_layers = torch.nn.ModuleList()
        ## always have at least 1 layer
        if (self.MedSumAfterProd == True) or (self.Att_MedAgg == True):
            self.hidden_layers.append(nn.Linear(in_features=self.input_size + self.e_dim_med_ids, out_features=self.hidden))
        else:
            self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
        ## sizes for subsequent layers
        hidensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype(
            'int64')
        for thisindex in range(len(hidensizes) - 1):
            self.hidden_layers.append(
                nn.Linear(in_features=hidensizes[thisindex], out_features=hidensizes[thisindex + 1]))

        """ static preops (BOW) bit """
        self.hidden_layers_bow = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_bow.append(nn.Linear(in_features=self.input_size_bow, out_features=self.hidden_bow))
        ## sizes for subsequent layers
        hidensizes_bow = np.ceil(
            np.linspace(start=self.hidden_bow, stop=self.hidden_final_bow, num=self.hidden_depth_bow)).astype(
            'int64')
        for thisindex in range(len(hidensizes_bow) - 1):
            self.hidden_layers_bow.append(
                nn.Linear(in_features=hidensizes_bow[thisindex], out_features=hidensizes_bow[thisindex + 1]))

        """ static preops (HOME MEDS) bit """

        if self.Att_HM_Agg == True:
            # attention layer to aggregate the home medication for each patient; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.input_size_hm ))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.input_size_hm  % self.AttMedAgg_Heads == 0:
                self.attention_hm = nn.MultiheadAttention(self.input_size_hm, self.AttMedAgg_Heads,
                                                                   batch_first=True)
            else:
                raise Exception("embedding dim needs to be divisible by number of attention heads")

        self.hidden_layers_hm = torch.nn.ModuleList()
        ## always have at least 1 layer
        self.hidden_layers_hm.append(nn.Linear(in_features=self.input_size_hm, out_features=self.hidden_hm))
        ## sizes for subsequent layers
        hiddensizes_hm = np.ceil(
            np.linspace(start=self.hidden_hm, stop=self.hidden_final_hm, num=self.hidden_depth_hm)).astype(
            'int64')
        for thisindex in range(len(hiddensizes_hm) - 1):
            self.hidden_layers_hm.append(
                nn.Linear(in_features=hiddensizes_hm[thisindex], out_features=hiddensizes_hm[thisindex + 1]))


        """ Med TS bit """
        self.embed_layer_med_ids = nn.Embedding(
            num_embeddings=self.v_med_ids + 1,
            embedding_dim=self.e_dim_med_ids,
            padding_idx=self.p_idx_med_ids
        )  # there is NA already in the dictionary that can be used a padding token


        self.embed_layer_units = nn.Embedding(
            num_embeddings=self.v_units + 1,
            embedding_dim=self.e_dim_units,
            padding_idx=self.p_idx_units
        )

        if self.Att_MedAgg == True:
            # attention layer to aggregate the medication at any given time; class_token to get all the information
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.e_dim_med_ids))  # "global information"
            torch.nn.init.normal_(self.class_token, std=0.02)
            if self.e_dim_med_ids % self.AttMedAgg_Heads == 0:
                self.attention_med_pertime = nn.MultiheadAttention(self.e_dim_med_ids, self.AttMedAgg_Heads, batch_first=True)
            else:
                raise Exception("embedding dim needs to be divisible by number of attention heads")


        # self.name_lstm = nn.LSTM(input_size=self.e_dim_words, hidden_size=self.lstm_hid, num_layers=1, batch_first=True)
        self.drop_layer1 = nn.Dropout(p=self.drp_rate)  ## this is element-wise drop in doses
        self.drop_layer2 = nn.Dropout(p=self.drp_final)  ## this is element-wise drop in doses

        if self.preops_init_med == True:
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_hid (med) is too restrictive
            self.preop_init_layer_med = nn.Linear(in_features=self.hidden_final, out_features=self.lstm_hid)

        if self.lstm_num_layers > 1:
            self.pre_final_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.e_dim_med_ids,
                hidden_size=self.lstm_hid,
                num_layers=self.lstm_num_layers,
                dropout=self.drp_rate_time,
                bidirectional=self.bilstm_med)
        else:
            self.pre_final_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.e_dim_med_ids,
                hidden_size=self.lstm_hid,
                num_layers=self.lstm_num_layers,
                bidirectional=self.bilstm_med
            )
        self.pre_final_lstm = self.pre_final_lstm.float()  # this was required to solve the inconsistencies between the datatypes TODO this suggests something wrong - it should already be floats (its inputs are floats)
        if self.finalBN == 1:
            self.bnlayer1 = nn.BatchNorm1d(self.hidden_final)


        """ TS flowsheet bit """

        if self.preops_init_flow == True:
            # this additional layer being added because use of unsqueeze or expand or reshape is not suitable and fixing hidden_final = lstm_flow_hid is too restrictive
            self.preop_init_layer_flow = nn.Linear(in_features=self.hidden_final, out_features=self.lstm_flow_hid)

        if self.lstm_flow_num_layers > 1:
            self.flowsheet_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.input_flow_size,
                hidden_size=self.lstm_flow_hid,
                num_layers=self.lstm_flow_num_layers,
                dropout=self.drp_rate_time_flow,
                bidirectional = self.bilstm_flow)
        else:
            self.flowsheet_lstm = nn.LSTM(
                batch_first=True,
                input_size=self.input_flow_size,
                hidden_size=self.lstm_flow_hid,
                num_layers=self.lstm_flow_num_layers,
                bidirectional=self.bilstm_flow)


        """ Static + TS """
        if self.Att_TSComb ==True:
            """ Attention over Meds and flowsheets """

            self.class_token_TS = torch.nn.Parameter(torch.randn(1, 1, self.lstm_hid + self.lstm_flow_hid))  # "global information"
            torch.nn.init.normal_(self.class_token_TS, std=0.08)
            if (self.lstm_hid + self.lstm_flow_hid) % self.AttTS_Heads == 0:
                self.attention_TS_layers = torch.nn.ModuleList()
                for i  in range(self.AttTS_depth):
                    self.attention_TS_layers.append(nn.MultiheadAttention(self.lstm_hid + self.lstm_flow_hid, self.AttTS_Heads,
                                                                   batch_first=True))
            else:
                raise Exception("model dim needs to be divisible by number of attention heads")

        if (self.MedSumAfterProd == True) or (self.Att_MedAgg == True):
            self.pre_final_linear = nn.Linear(
                in_features= self.hidden_final + self.hidden_final_bow + self.hidden_final_hm,
                out_features=self.hidden_final)
        else:
            self.pre_final_linear = nn.Linear(in_features=self.lstm_hid + self.hidden_final + self.hidden_final_bow +self.hidden_final_hm + self.lstm_flow_hid , out_features=self.hidden_final)
        self.final_linear = nn.Linear(in_features=self.hidden_final, out_features=self.linear_out)

        if(self.weightInt):
            self._reinitialize()


    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif ('hidden' in name) or ('linear' in name):
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, preops, seq_len, bow_data, home_meds, med_ids, dose, units, flowsheets,  ):
        #preops = preops.to(device)
        #bow_data = bow_data.to(device)
        #med_ids = med_ids.to(device)
        #dose = dose.to(device)
        #units = units.to(device)
        #flowsheets = flowsheets.to(device)

        # print("testing")

        """ preops (BOW) MLP"""
        bow_path = torch.nn.ReLU()(self.hidden_layers_bow[0](bow_data))
        if (len(self.hidden_layers_bow) > 1):
            for thisindex in range(len(self.hidden_layers_bow) - 1):
                bow_path = torch.nn.ReLU()(self.hidden_layers_bow[thisindex + 1](bow_path))

        bow_l1_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=1) for i in range(self.hidden_depth_bow)]
        bow_l2_reg_loss = [torch.norm(self.hidden_layers_bow[i].weight.data, p=2) for i in range(self.hidden_depth_bow)]

        """ preops (HOME MEDS) MLP"""

        if self.Att_HM_Agg == True:
            bs = home_meds.size(0)
            home_meds_att = torch.cat([self.class_token.expand(bs, -1, -1), home_meds],
                                                   dim=1)
            encodings, _ = self.attention_hm(home_meds_att, home_meds_att,
                                                      home_meds_att)
            home_meds_att0 = encodings[:, 0, :]
            hm_path = torch.nn.ReLU()(self.hidden_layers_hm[0](home_meds_att0))
        else:
            hm_path = torch.nn.ReLU()(self.hidden_layers_hm[0](home_meds))
        if (len(self.hidden_layers_hm) > 1):
            for thisindex in range(len(self.hidden_layers_hm) - 1):
                hm_path = torch.nn.ReLU()(self.hidden_layers_hm[thisindex + 1](hm_path))

        hm_l1_reg_loss = [torch.norm(self.hidden_layers_hm[i].weight.data, p=1) for i in range(self.hidden_depth_hm)]
        hm_l2_reg_loss = [torch.norm(self.hidden_layers_hm[i].weight.data, p=2) for i in range(self.hidden_depth_hm)]

        """ initial meds """

        # dropout before embedding layer. Using doses since we multiply the doses with units annd embedding. It circumvents the problem of  passing dropped out  values to the embedding layer
        # this is okay even when the there is dropout is included in the lstm with more than one layer because the dropout is applied to the outpput of lstm layer.
        dose = self.drop_layer1(dose)

        # embedding the names and units
        # units_embedding = self.embed_layer_units(units.long())
        units_embedding = self.embed_unit_med_comb(units.long())

        med_ids_temp_embedding = self.embed_layer_med_ids(med_ids.long())

        if True:
            med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        else:
            med_combined_embed = torch.cat((dose.unsqueeze(-1), units_embedding, med_ids_temp_embedding), 3)  # was used as an alternative


        """ preops MLP """

        if self.MedSumAfterProd == True:
            # breakpoint()
            med_combined_embed = torch.sum(med_combined_embed, 1) # summing over the med dimension for summarizing part
            preops = torch.concat((preops, med_combined_embed), 1)

        if self.Att_MedAgg == True:  # attention for aggregation instead of summing over the medication dimension and then summing over the time dimension, finally appending the meds to preops
            idxname = med_combined_embed.shape
            med_combined_embed_pertime = med_combined_embed.contiguous().view(idxname[0] * idxname[1], *idxname[2:])
            bs = med_combined_embed_pertime.size(0)
            med_combined_embed_pertime = torch.cat([self.class_token.expand(bs, -1, -1), med_combined_embed_pertime],
                                                   dim=1)
            encodings, _ = self.attention_med_pertime(med_combined_embed_pertime, med_combined_embed_pertime,
                                                      med_combined_embed_pertime)
            med_combined_embed0 = encodings[:, 0, :]
            med_combined_embed = med_combined_embed0.contiguous().view(idxname[0], idxname[1], idxname[-1])

            med_combined_embed = torch.sum(med_combined_embed, 1)  # summing over the med dimension for summarizing part
            preops = torch.concat((preops, med_combined_embed), 1)

        my_device = next(self.hidden_layers[0].parameters()).device
        preop_path = torch.nn.ReLU()(self.hidden_layers[0](preops))
        if (len(self.hidden_layers) > 1):
            for thisindex in range(len(self.hidden_layers) - 1):
                preop_path = torch.nn.ReLU()(self.hidden_layers[thisindex + 1](preop_path))

        preops_l1_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=1) for i in range(self.hidden_depth)]
        preops_l2_reg_loss = [torch.norm(self.hidden_layers[i].weight.data, p=2) for i in range(self.hidden_depth)]

        if self.MedSumAfterProd == True:
            final_for_mlp = torch.cat((bow_path, hm_path, preop_path), 1)  # only preops (appended with flowsheets and meds), bow, home meds
            final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

            if self.finalBN == 1:
                final_for_mlp = self.bnlayer1(final_for_mlp)

            outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp))
            if self.binary_outcome == 1:
                outcome_pred = torch.sigmoid(outcome_pred)

            # total weight decay regularization loss
            reg_wd = self.preopsWDRateL1 * sum(preops_l1_reg_loss) + self.preopsWDRateL2 * sum(preops_l2_reg_loss) + \
                     self.bowWDRateL1 * sum(bow_l1_reg_loss) + self.bowWDRateL2 * sum(bow_l2_reg_loss) + \
                     self.hmWDRateL1 * sum(hm_l1_reg_loss) + self.hmWDRateL2 * sum(hm_l2_reg_loss)

            return outcome_pred, reg_wd



        if self.Att_MedAgg == True:  # attention for aggregation instead of summing over the medication dimension and then summing over the time dimension, finally appending the meds to preop
            final_for_mlp = torch.cat((bow_path, hm_path, preop_path), 1)  # only preops (appended with flowsheets and meds), bow, home meds
            final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

            if self.finalBN == 1:
                final_for_mlp = self.bnlayer1(final_for_mlp)

            outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp))
            if self.binary_outcome == 1:
                outcome_pred = torch.sigmoid(outcome_pred)

            # total weight decay regularization loss
            reg_wd = self.preopsWDRateL1 * sum(preops_l1_reg_loss) + self.preopsWDRateL2 * sum(preops_l2_reg_loss) + \
                     self.bowWDRateL1 * sum(bow_l1_reg_loss) + self.bowWDRateL2 * sum(bow_l2_reg_loss) + \
                     self.hmWDRateL1 * sum(hm_l1_reg_loss) + self.hmWDRateL2 * sum(hm_l2_reg_loss)

            return outcome_pred, reg_wd

        else: # summing the values across the medication dimension
            med_combined_embed = torch.sum(med_combined_embed, 2)

        """ Back to intra-op meds """

        # initializing the hidden state of med lstm with preop mlp's output
        if self.bilstm_med == True:
            D_med = 2  # D is the notation that is used by official docs too for emphasizing on bidirectional
        else:
            D_med = 1

        if self.preops_init_med == True:
            hidden_state_lstm_med = self.preop_init_layer_med(preop_path)
            hidden_state_lstm_med = torch.unsqueeze(hidden_state_lstm_med, 0)
            hidden_state_lstm_med = hidden_state_lstm_med.expand(D_med * self.lstm_num_layers, preop_path.size(0),
                                                                 self.lstm_hid)
        else:
            # Creation of cell state and hidden state for lstm
            hidden_state_lstm_med = torch.zeros(D_med * self.lstm_num_layers, med_combined_embed.size(0),
                                                self.lstm_hid).to( my_device)
            cell_state_lstm_med = torch.zeros(D_med * self.lstm_num_layers, med_combined_embed.size(0),
                                              self.lstm_hid).to(my_device)

            ## TODO: add an MLP of preops initializing the hidden state; done
            # Weights initialization
            torch.nn.init.xavier_normal_(hidden_state_lstm_med)
            torch.nn.init.xavier_normal_(cell_state_lstm_med)

        # breakpoint()
        # packing-unpacking for the final lstm too; here we will use the padding value as 0 (from the dose) because we padded over the time dimension
        med_combined_embed_packed = torch.nn.utils.rnn.pack_padded_sequence(med_combined_embed, seq_len,
                                                                            batch_first=True, enforce_sorted=False)

        if self.preops_init_med == True:
            # print("Init is working")
            lstm_output_med, (final_embed_comb_ht, final_embed_comb_ct) = self.pre_final_lstm(
                med_combined_embed_packed,
                (hidden_state_lstm_med.contiguous(), hidden_state_lstm_med.contiguous())
            )
        else:
            # print("Init is not working")
            lstm_output_med, (final_embed_comb_ht, final_embed_comb_ct) = self.pre_final_lstm(
                med_combined_embed_packed,
                (hidden_state_lstm_med.contiguous(), cell_state_lstm_med.contiguous())
            )

        # final linear layer with batch normalization before the mlp
        if self.bilstm_med == True:
            prefinal_for_mlp = final_embed_comb_ht[-2, :, :] + final_embed_comb_ht[-1, :, :]
        else:
            prefinal_for_mlp = final_embed_comb_ht[-1]



        meds_l2_reg_loss = [torch.norm(self.pre_final_lstm.all_weights[i][0], p=2) +torch.norm(self.pre_final_lstm.all_weights[i][1], p=2) for i in range(self.lstm_num_layers)]


        # for attention concatenation
        # torch.nn.utils.rnn.pad_packed_sequence(lstm_output_med, batch_first=True)[0]
        # breakpoint()
        """  time series flowsheet """

        if self.bilstm_flow ==True:
            D_flow = 2 # D is the notation that is used by official docs too for emphasizing on bidirectional
        else:
            D_flow = 1

        # initializing the hidden state with preop mlp's output
        if self.preops_init_flow == True:
            hidden_state_lstm_flow = self.preop_init_layer_flow(preop_path)
            hidden_state_lstm_flow = torch.unsqueeze(hidden_state_lstm_flow,0)
            hidden_state_lstm_flow = hidden_state_lstm_flow.expand(D_flow*self.lstm_flow_num_layers, preop_path.size(0), self.lstm_flow_hid)

        # packing unpacking the flowsheet data too for lstm
        flowsheet_packed = torch.nn.utils.rnn.pack_padded_sequence(flowsheets, seq_len, batch_first=True, enforce_sorted=False)

        if self.preops_init_flow == True:
            lstm_output_fl, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed , (hidden_state_lstm_flow.contiguous(), hidden_state_lstm_flow.contiguous()))
        else:
            lstm_output_fl, (final_embed_comb_ht_f, final_embed_comb_ct_f) = self.flowsheet_lstm(flowsheet_packed )

        # final linear layer with batch normalization before the mlp
        if self.bilstm_flow == True:
            prefinal_for_mlp_fl = final_embed_comb_ht_f[-2, :, :] + final_embed_comb_ht_f[-1, :, :]
        else:
            prefinal_for_mlp_fl = final_embed_comb_ht_f[-1]

        flowsheet_l2_reg_loss = [torch.norm(self.flowsheet_lstm.all_weights[i][0], p=2) +torch.norm(self.flowsheet_lstm.all_weights[i][1], p=2) for i in range(self.lstm_flow_num_layers)]

        # for attention concatenation
        # torch.nn.utils.rnn.pad_packed_sequence(lstm_output_fl, batch_first=True)[0]
        # breakpoint()

        """ concatenate the hidden from mlp and lstm """

        if self.Att_TSComb ==True:
            # attention based concatenation
            final_for_mlp0 = torch.cat((torch.nn.utils.rnn.pad_packed_sequence(lstm_output_med, batch_first=True)[0], torch.nn.utils.rnn.pad_packed_sequence(lstm_output_fl, batch_first=True)[0]), 2)
            bs = final_for_mlp0.size(0)
            # breakpoint()
            lstm_outout_TS_comb = torch.cat([self.class_token_TS.expand(bs, -1, -1), final_for_mlp0], dim=1)


            attTS_path, _  = self.attention_TS_layers[0](lstm_outout_TS_comb, lstm_outout_TS_comb,
                                                      lstm_outout_TS_comb)
            for i in range(self.AttTS_depth-1):
                attTS_path, _ = self.attention_TS_layers[i+1](attTS_path, attTS_path,
                                                      attTS_path)

            # encodings_TS1, _ = self.attention_TS1(lstm_outout_TS_comb, lstm_outout_TS_comb,
            #                                           lstm_outout_TS_comb)
            # encodings_TS2, _ = self.attention_TS2(encodings_TS1, encodings_TS1,
            #                                           encodings_TS1)
            # encodings_TS3, _ = self.attention_TS3(encodings_TS2, encodings_TS2,
            #                                           encodings_TS2)
            # final_for_mlp_TS = encodings_TS3[:, 0, :]
            final_for_mlp_TS = attTS_path[:, 0, :]
            final_for_mlp = torch.cat((final_for_mlp_TS, bow_path, preop_path), 1)  #
        else:
            # final_for_mlp = torch.cat((preop_path, seq_len.to(device).unsqueeze(-1), prefinal_for_mlp_fl), 1)
            final_for_mlp = torch.cat((prefinal_for_mlp, bow_path, hm_path, preop_path, prefinal_for_mlp_fl), 1)  #


        final_for_mlp = torch.nn.ReLU()(self.pre_final_linear(final_for_mlp))

        if self.finalBN == 1:
            final_for_mlp = self.bnlayer1(final_for_mlp)

        outcome_pred = self.final_linear(self.drop_layer2(final_for_mlp) )

        if torch.isnan(outcome_pred).any() == True:
            print("outcome has nan from somewhere")

        if self.binary_outcome == 1:
          outcome_pred = torch.sigmoid(outcome_pred)

        # total weight decay regularization loss
        reg_wd = self.lstmMedWDRateL2* sum(meds_l2_reg_loss) + self.lstmFlowWDRateL2 * sum(flowsheet_l2_reg_loss) + \
                 self.preopsWDRateL1* sum(preops_l1_reg_loss) + self.preopsWDRateL2* sum(preops_l2_reg_loss) + \
                 self.bowWDRateL1* sum(bow_l1_reg_loss) + self.bowWDRateL2* sum(bow_l2_reg_loss) + \
                 self.hmWDRateL1 * sum(hm_l1_reg_loss) + self.hmWDRateL2 * sum(hm_l2_reg_loss)

        return outcome_pred, reg_wd

## inputs are assumed to be: [preops, durations, bow, hm, med_index, med_dose, med_unit, flow_dense, flow_sparse (optional), labels (optional)]
## Completely created by Ryan
## provision for home meds as input
def collate_time_series(batch):
    durations = [example[1].int().item() for example in batch]
    sorted_batch = sorted(batch, key=lambda x: x[1].int().item(), reverse=True)
    num_objects = len(sorted_batch[0])
    has_labels = (num_objects > 8) and (len(sorted_batch[0][-1].shape) < 2)
    has_optional_flow = (num_objects > 8) and (len(sorted_batch[0][7].shape) == 2)
    has_missingness_mask = (num_objects > 11)
    preop_dim = sorted_batch[0][0].shape[0]
    dense_flow_dim = sorted_batch[0][7].shape[1]
    sparse_flow_dim = sorted_batch[0][8].shape[1]
    summary_meds = len(sorted_batch[0][4].shape)==1
    # breakpoint()
    ## truncate the time series and pad to a common length, then use the 2d pad if the achieved length does not match the target
    ## length WITHOUT sparse flowsheets -> 7, with sparse -> 8
    if (has_optional_flow):
        if (has_missingness_mask):
            if (summary_meds):
                med_index = torch.stack([p[4].int().to_dense() for p in sorted_batch])
                med_dose = torch.stack([p[5].float().to_dense() for p in sorted_batch])
                med_unit = torch.stack([p[6].int().to_dense() for p in sorted_batch])
            else:
                med_index =torch.nn.utils.rnn.pad_sequence([p[4].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True)                  ## this truncation could be done before creation of the tensor
                med_dose = torch.nn.utils.rnn.pad_sequence([p[5].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True)
                med_unit = torch.nn.utils.rnn.pad_sequence([p[6].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True)
            output = [
                torch.stack([torch.hstack([p[0].to_dense(), p[9].to_dense()]) for p in sorted_batch]),
                # appended the mask vector of preops data
                torch.stack([p[1].int().to_dense() for p in sorted_batch]),
                torch.stack([p[2].to_dense() for p in sorted_batch]),
                torch.stack([p[3].to_dense() for p in sorted_batch]),
                med_index,
                med_dose,
                med_unit,
                torch.nn.utils.rnn.pad_sequence([torch.hstack([torch.stack(
                    [p[7].to_dense()[0:p[1].int().item(), :], p[10].to_dense()[0:p[1].int().item(), :]], dim=2).reshape(
                    min(p[1].int().item(), p[7].shape[0]), dense_flow_dim * 2), torch.stack(
                    [torch.cumsum(p[8].to_dense()[0:p[1].int().item(), :], dim=0),
                     p[11].to_dense()[0:p[1].int().item(), :]], dim=2).reshape(min(p[1].int().item(), p[8].shape[0]),
                                                                               2 * sparse_flow_dim)])
                    for p in sorted_batch],
                    batch_first=True),
                # appended the mask vector of flowsheet data; the min is there because for the endofcase there are some durations that are more than 512
            ]
        else:
            output = [
                torch.stack([p[0].to_dense() for p in sorted_batch]),
                torch.stack([p[1].int().to_dense() for p in sorted_batch]),
                torch.stack([p[2].to_dense() for p in sorted_batch]),
                torch.stack([p[3].to_dense() for p in sorted_batch]),
                torch.nn.utils.rnn.pad_sequence([p[4].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True),
                ## this truncation could be done before creation of the tensor
                torch.nn.utils.rnn.pad_sequence([p[5].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True),
                torch.nn.utils.rnn.pad_sequence([p[6].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                                batch_first=True),
                torch.nn.utils.rnn.pad_sequence([torch.hstack([p[7].to_dense()[0:p[1].int().item(), :],
                                                               torch.cumsum(p[8].to_dense()[0:p[1].int().item(), :],
                                                                            dim=0)]) for p in sorted_batch],
                                                batch_first=True),
                ## note that a zero-dimension sparse hstacks as a no-op, so handles if it is included as a zero-column sparse
            ]
    else:
        output = [
            torch.stack([p[0].to_dense() for p in sorted_batch]),
            torch.stack([p[1].int().to_dense() for p in sorted_batch]),
            torch.stack([p[2].to_dense() for p in sorted_batch]),
            torch.stack([p[3].to_dense() for p in sorted_batch]),
            torch.nn.utils.rnn.pad_sequence([p[4].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                            batch_first=True),
            torch.nn.utils.rnn.pad_sequence([p[5].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                            batch_first=True),
            torch.nn.utils.rnn.pad_sequence([p[6].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                            batch_first=True),
            torch.nn.utils.rnn.pad_sequence([p[7].to_dense()[0:p[1].int().item(), :] for p in sorted_batch],
                                            batch_first=True),
        ]
    ## test and handle if the max achieved time is less than the inference target time
    if(summary_meds):
        for i in range(7, len(output)):
            augment = max(durations) - output[i].shape[1]
            if augment > 0:
                output[i] = torch.nn.functional.pad(output[i],
                                                    pad=[0, 0, 0, augment])  ## pad goes back to front along the shape
    else:
        for i in range(4, len(output)):
            augment = max(durations) - output[i].shape[1]
            if augment > 0:
                output[i] = torch.nn.functional.pad(output[i],
                                                    pad=[0, 0, 0, augment])  ## pad goes back to front along the shape
    if has_labels:
        labels = torch.stack([example[-1] for example in sorted_batch])
        return [output, labels]
    else:
        return output


