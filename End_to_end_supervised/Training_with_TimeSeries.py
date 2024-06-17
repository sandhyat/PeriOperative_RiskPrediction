""" MOST recent code running command on moose  """

# docker run --rm --gpus all --privileged -v '< /PATH TO THE INPUT DATA/ >:/input/' -v '< /PATH TO THE SCRIPTS/ >:/codes/' -v '< /PATH TO THE SAVING THE OUTPUT RESULTS/ >:/output/' docker121720/pytorch-for-ts:0.5 python /codes/testing-new_collate-for-docker.py --nameinfo="testing_Full_fixed_seed_withmaskOversamplingEarlyStoppingLR" --outputcsv="test_binary_outcomes.csv" --task='icu' --preopsDepth=6 --preopsWidth=20 --preopsWidthFinal=16 --bowDepth=5 --bowWidth=90 --bowWidthFinal=20 --lstmMedEmbDim=16 --lstmMedDepth=4 --lstmMedWidth=40 --lstmFlowDepth=4 --lstmFlowWidth=40 --LRPatience=3 --batchSize=120 --lstmMedDrop=0.1212 --lstmFlowDrop=0.0165 --finalDrop=0.3001 --learningRate=0.0002 --learningRateFactor=0.2482 --preopsL2=0.0004 --preopsL1=0.0029 --bowL2=0.0003 --bowL1=0.0042 --lstmMedL2=0.0003 --lstmFlowL2=0.0009 --randomSeed=350 --includeMissingnessMasks --overSampling

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
import preop_flow_med_bow_model 

# presetting the number of threads to be used
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
# torch.cuda.set_per_process_memory_fraction(1.0, device=None)

# TODO: pick better defaults
# TODO: make the same modifications to using the word-sequence
# TODO: MLP for final state (currently fixed at 2 layer)

# starting time of the script
start_time = datetime.now()

parser = argparse.ArgumentParser(description='HP for ML optimization')

## for the preops before concat to ts output
parser.add_argument("--preopsDepth",  default=3, type=int) #
parser.add_argument("--preopsWidth",  default=10, type=int) #
parser.add_argument("--preopsWidthFinal",  default=5, type=int) #
parser.add_argument("--preopsL2",  default=0.2, type=float)
parser.add_argument("--preopsL1",  default=0.1, type=float)
parser.add_argument("--preopsBN", default=False, action='store_true') ## not implemented

## for the bow before concat to ts output
parser.add_argument("--bowDepth",  default=3, type=int) #
parser.add_argument("--bowWidth",  default=300, type=int) #
parser.add_argument("--bowWidthFinal",  default=10, type=int) #
parser.add_argument("--bowL2",  default=0.2, type=float)
parser.add_argument("--bowL1",  default=0.1, type=float)

## for the homemeds
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum', 'embedding_attention'}
parser.add_argument("--AttentionHhomeMedsAgg", default=False, action='store_true') # this needs to be true when embedding_attention is active in the above line
parser.add_argument("--hmDepth", default=5, type=int)  #
parser.add_argument("--hmWidth", default=400, type=int)  #
parser.add_argument("--hmWidthFinal", default=10, type=int)  #
parser.add_argument("--hmL2", default=0.01, type=float)
parser.add_argument("--hmL1", default=0.01, type=float)

## for processing medication IDs (or the post-embedding words)
parser.add_argument("--lstmMedEmbDim",  default=5, type=int) #
parser.add_argument("--lstmMedDepth",  default=1, type=int)     #
parser.add_argument("--lstmMedWidth",  default=5, type=int) #
parser.add_argument("--lstmMedL2",  default =0.2 , type=float)
parser.add_argument("--lstmMedDrop",  default=0., type=float)  #
parser.add_argument("--preopInitLstmMed", default=True, action='store_true')
parser.add_argument("--BilstmMed", default=False, action='store_true')
parser.add_argument("--AttentionMedAgg", default=False, action='store_true')
parser.add_argument("--AttMedAggHeads", default=2, type=int)


## for processing words within a medication name
parser.add_argument("--lstmWordEmbDim",  default=5, type=int) # uses lstmMedEmbDim
parser.add_argument("--lstmWordDepth",  default=1, type=int)                 #
parser.add_argument("--lstmWordWidth",  default=5, type=int) ## not implemented
parser.add_argument("--lstmWordL2",  default =0. , type=float) ## not implemented
parser.add_argument("--lstmWordDrop",  default=0., type=float)  ## not implemented               

## generic dropout of med entry data
parser.add_argument("--lstmRowDrop",  default=0., type=float)    #              

## for processing medication units
## TODO: there is not proper support for units embed dim != med embeed dim or 1, you would have to add a fc layer to make the arrays conformable (or switch to concatenate instead of multiply)
parser.add_argument("--lstmUnitExpand", default=False, action='store_true') #

## for processing flowsheet data
## It's not clear that having 2 LSTMs is good or necessary instead of concatenate the inputs at each timestep
parser.add_argument("--lstmFlowDepth",  default=1, type=int)  #
parser.add_argument("--lstmFlowWidth", default=5 , type=int) #
parser.add_argument("--lstmFlowL2",  default=0.2, type=float)
parser.add_argument("--lstmFlowDrop",  default=0., type=float)
parser.add_argument("--preopInitLstmFlow", default=True, action='store_true')
parser.add_argument("--BilstmFlow", default=False, action='store_true')

## for the MLP combining preop and LSTM outputs
parser.add_argument("--finalDrop",  default=.4, type=float)  #               
parser.add_argument("--finalWidth",  default=10, type=int)   #              
parser.add_argument("--finalDepth",  default=3, type=int)    ## not implemented, fixed at   preopsWidthFinal           
parser.add_argument("--finalBN", default=False, action='store_true') #

## learning parametersq
parser.add_argument("--batchSize",  default=32, type=int) #
parser.add_argument("--learningRate",  default=1e-3, type=float) #
parser.add_argument("--learningRateFactor",  default=0.1, type=float) #
parser.add_argument("--LRPatience",  default=2, type=int) #
parser.add_argument("--epochs",  default=100, type=int) #
parser.add_argument("--XavOrthWeightInt", default=True, action='store_true')  # changes torch's weight initialization to xavier and orthogonal


## task and setup parameters
parser.add_argument("--task",  default="icu") #
parser.add_argument("--binaryEOC", default=True, action='store_true')  #
parser.add_argument("--drugNamesNo", default=True,  action='store_true') #
parser.add_argument("--skipPreops",  default=False, action='store_true') # True value of this will only use bow
parser.add_argument("--sepPreopsBow",  default=True, action='store_true') # True value of this variable would treat bow and preops as sep input and have different mlps for them. Also, both skipPreops and sepPreopsBow can't be True at the same time
parser.add_argument("--trainTime", default=True, action='store_true')
parser.add_argument("--testcondition", default='None') # options in  {None, preopOnly, MedOnly, FlowOnly, MedFlow }
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--includeMissingnessMasks", default=False, action='store_true')
parser.add_argument("--overSampling", default=True, action='store_true') # keep it as False when task is endofcase


## output parameters
parser.add_argument("--git",  default="") # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo",  default="") #
parser.add_argument("--outputcsv",  default="") #

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing


# reproducibility settings
# random_seed = 1 # or any of your favorite number
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)


data_dir = '/input/'

# reading the preop and outcome feather files
preops = feather.read_feather(data_dir + 'preops_reduced_for_training.feather')
preops = preops.drop(['MRN_encoded'], axis =1)
outcomes = pd.read_csv(data_dir + 'epic_outcomes.csv')
bow_input = pd.read_csv(data_dir+'cbow_proc_text.csv')

if False:
    preops.drop(columns='plannedDispo', inplace=True)

# home meds reading and processing
home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
Drg_pretrained_embedings = pd.read_csv('./Pain_outcome_models/df_cui_vec_2sourceMappedWODupl.csv')
# Drg_pretrained_embedings = pd.read_csv('/codes/Pain_outcome_models/df_cui_vec_2sourceMappedWODupl.csv')

# home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
# home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')
home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded','rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
home_meds_embedded = home_meds[['orlogid_encoded','rxcui']].merge(Drg_pretrained_embedings, how='left', on='rxcui')
home_meds_embedded.drop(columns=['code', 'description', 'source'], inplace= True)

# breakpoint()
# home meds basic processing
home_meds_freq = home_meds[['orlogid_encoded','rxcui','Frequency']].pivot_table(index='orlogid_encoded', columns='rxcui', values='Frequency')
rxcui_freq = home_meds["rxcui"].value_counts().reset_index()
#rxcui_freq = rxcui_freq.rename({'count':'rxcui_freq', 'rxcui':'rxcui'}, axis =1)
rxcui_freq = rxcui_freq.rename({'rxcui':'rxcui_freq', 'index':'rxcui'}, axis =1)
home_meds_small =  home_meds[home_meds['rxcui'].isin(list(rxcui_freq[rxcui_freq['rxcui_freq']>100]['rxcui']))]
home_meds_small['temp_const'] = 1
home_meds_ohe = home_meds_small[['orlogid_encoded', 'rxcui','temp_const']].pivot_table(index='orlogid_encoded', columns='rxcui', values='temp_const')
home_meds_ohe.fillna(0, inplace=True)
# breakpoint()

# end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
end_of_case_times = outcomes[['orlogid_encoded', 'endtime']]


regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration', 'n_glu_high',
                           'low_sbp_time','aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time', 'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1']
binary_outcome = task not in regression_outcome_list


outcomes = outcomes.dropna(subset=['ICU'])
outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

if task == 'endofcase':
    # updating the end_of_case_times targets for bigger distribution;
    """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 60] ## cases that are too short
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] < 25+511] ## cases that are too long
    end_of_case_times['true_test'] = end_of_case_times['endtime'] - 10
    end_of_case_times['t1'] = end_of_case_times['true_test'] -30
    end_of_case_times['t2'] = end_of_case_times['true_test'] -35 # temporary just to make sure nothing breaks; not being used
    end_of_case_times['t3'] = end_of_case_times['true_test'] -40 # temporary just to make sure nothing breaks; not being used
    overSampling = False  # TODO: there could be a better way to handle this.
else :
    end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 511)
    # end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 90)

# outcome
icu_outcome = outcomes[['orlogid_encoded', 'ICU']]
icu_outcome.loc[icu_outcome['ICU'] == True, 'ICU'] = 1
icu_outcome.loc[icu_outcome['ICU'] == False, 'ICU'] = 0
icu_outcome['ICU'] = icu_outcome['ICU'].astype(int)

mortality_outcome = outcomes[['orlogid_encoded', 'death_in_30']]
mortality_outcome.loc[mortality_outcome['death_in_30'] == True, 'death_in_30'] = 1
mortality_outcome.loc[mortality_outcome['death_in_30'] == False, 'death_in_30'] = 0
mortality_outcome['death_in_30'] = mortality_outcome['death_in_30'].astype(int)

aki_outcome = outcomes[['orlogid_encoded', 'post_aki_status']]
if task == 'aki1':
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
    aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
if task == 'aki2':
    aki_outcome.loc[aki_outcome[
                        'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
if task == 'aki3':
    aki_outcome.loc[aki_outcome[
                        'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
aki_outcome['post_aki_status'] = aki_outcome['post_aki_status'].astype(int)

dvt_pe_outcome = outcomes[['orlogid_encoded', 'DVT_PE']]

# flowsheet data
very_dense_flow = feather.read_feather(data_dir +"flow_ts/Imputed_very_dense_flow.feather")
very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace=True)
very_dense_flow = very_dense_flow.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint']]
very_dense_flow.drop(["endtime"], axis=1, inplace=True)

other_intra_flow_wlabs = feather.read_feather(data_dir +"flow_ts/Imputed_other_flow.feather")
other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index, inplace=True)
other_intra_flow_wlabs = other_intra_flow_wlabs.merge(end_of_case_times[['orlogid_encoded', 'endtime']],
                                                      on="orlogid_encoded")
other_intra_flow_wlabs = other_intra_flow_wlabs.loc[
    other_intra_flow_wlabs['endtime'] > other_intra_flow_wlabs['timepoint']]
other_intra_flow_wlabs.drop(["endtime"], axis=1, inplace=True)

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

a = pd.DataFrame(columns=['med_integer','unit_integer', 'med_unit_combo'])
a['med_integer'] = [ med_unit_unique_codes[i][0] for i in range(len(med_unit_unique_codes))]
a['unit_integer'] = [ med_unit_unique_codes[i][1] for i in range(len(med_unit_unique_codes))]
a['med_unit_combo'] = np.arange(len(med_unit_unique_codes))
a.sort_values(by=['med_integer','med_unit_combo'], inplace=True)


group_start = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']), 1)).nonzero().squeeze()  +1 # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.
group_end = (torch.tensor(a['med_integer']) != torch.roll(torch.tensor(a['med_integer']), -1)).nonzero().squeeze() +1 # this one is needed becasue otherwise there was some incompatibbility while the embeddginff for the combination are being created.

group_start = torch.cat((torch.tensor(0).reshape((1)), group_start)) # prepending 0 to make sure that it is treated as an empty slot
group_end = torch.cat((torch.tensor(0).reshape((1)), group_end)) # prepending 0 to make sure that it is treated as an empty slot


drug_med_ids = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_integer']]

if drugNamesNo == True:
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


if drugNamesNo == False:
    vocab_len_words = len(word_id_map)
else:
    vocab_len_med_ids = len(drug_med_id_map)

print("Passed all the data reading stage")

binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac', 'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']

if task in regression_outcome_list:
    outcomes['survival_time'] = np.minimum(outcomes['survival_time'], 90)
    outcomes['readmission_survival'] = np.minimum(outcomes['readmission_survival'], 30)
    # outcomes['n_glucose_high'] = outcomes['n_glucose_high'].fillna(0)  # this might not be needed as already taken of by the where statement
    outcomes['n_glu_high'] = np.where(outcomes['N_glu_measured'] > 0, outcomes['n_glu_high']/outcomes['N_glu_measured'], 0)
    outcomes['total_blood'] = outcomes['total_blood'].fillna(0)
    outcomes['low_sbp_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_sbp_time']/outcomes['total_t'], 0)
    outcomes['low_relmap_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_time']/outcomes['total_t'], 0)
    outcomes['low_map_time'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_time']/outcomes['total_t'], 0)
    outcomes['aoc_low_sbp'] = np.where(outcomes['total_t'] > 0, outcomes['aoc_low_sbp'], 0)
    outcomes['low_relmap_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_relmap_aoc'], 0)
    outcomes['low_map_aoc'] = np.where(outcomes['total_t'] > 0, outcomes['low_map_aoc'], 0)
    outcomes['postop_vent_duration'] = outcomes['postop_vent_duration'].fillna(0)
    outcomes['timew_pain_avg_0'] = outcomes['timew_pain_avg_0']/(outcomes['timew_pain_avg_0'].max() - outcomes['timew_pain_avg_0'].min())
    outcomes['median_pain_0'] = outcomes['median_pain_0']/(outcomes['median_pain_0'].max() - outcomes['median_pain_0'].min())
    outcomes['worst_pain_0'] = outcomes['worst_pain_0']/(outcomes['worst_pain_0'].max() - outcomes['worst_pain_0'].min())
    outcomes['worst_pain_1'] = outcomes['worst_pain_1']/(outcomes['worst_pain_1'].max() - outcomes['worst_pain_1'].min())


    outcome_df = outcomes[['orlogid_encoded', task]]
elif task in binary_outcome_list:
    if task == 'VTE':
        temp_outcome = outcomes[['orlogid_encoded']]
        temp_outcome[task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[task] == 2, task] = 1
    elif task == 'n_glucose_low':
        temp_outcome = outcomes[['orlogid_encoded', task]]
        temp_outcome[task] = temp_outcome[task].fillna(0)
        temp_outcome[task] = np.where(temp_outcome[task]>0, 1, 0)
    else:
        temp_outcome = outcomes[['orlogid_encoded', task]]
        temp_outcome.loc[temp_outcome[task] == True, task] = 1
        temp_outcome.loc[temp_outcome[task] == False, task] = 0
    temp_outcome[task] = temp_outcome[task].astype(int)
    outcome_df = temp_outcome
elif (task == 'dvt_pe'):
    outcome_df = dvt_pe_outcome
elif (task == 'icu'):
    outcome_df = icu_outcome
elif (task == 'mortality'):
    outcome_df = mortality_outcome
elif (task == 'aki1' or task == 'aki2' or task == 'aki3'):
    outcome_df = aki_outcome
elif (task == 'endofcase'):
    outcome_df = end_of_case_times[['orlogid_encoded', 'true_test']]
else:
    raise Exception("outcome not handled")


## intersect 3 mandatory data sources: preop, outcome, case end times
combined_case_set = list(set(outcome_df["orlogid_encoded"].values).intersection(
    set(end_of_case_times['orlogid_encoded'].values)).intersection(
    set(preops['orlogid_encoded'].values)))

if False:
    combined_case_set = np.random.choice(combined_case_set, 2500, replace=False)

outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(combined_case_set)]
preops = preops.loc[preops['orlogid_encoded'].isin(combined_case_set)]
end_of_case_times = end_of_case_times.loc[end_of_case_times['orlogid_encoded'].isin(combined_case_set)]

outcome_df = outcome_df.set_axis(["orlogid_encoded", "outcome"], axis=1)

# checking for NA and other filters
outcome_df = outcome_df.loc[outcome_df['orlogid_encoded'].isin(preops["orlogid_encoded"].unique())]
outcome_df = outcome_df.dropna(axis=0).sort_values(["orlogid_encoded"]).reset_index(drop=True)
new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)   # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

## drop missing data

drug_dose = drug_dose.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1)
preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                     axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

home_meds_ohe = home_meds_ohe.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(),preops.index.max()+1)),fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
home_meds_ohe.fillna(0, inplace =True)  # setting the value for the ones that were added later

home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(),preops.index.max()+1)),fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
home_meds_sum.fillna(0, inplace =True)  # setting the value for the ones that were added later

# breakpoint()

if home_medsform == 'ohe':
    home_meds_final = home_meds_ohe
if home_medsform == 'embedding_sum':
    # TODO: remove the rxcui number from the home_meds_sum dataframe
    home_meds_final = home_meds_sum
hm_embed_flag = 0  # not applicable case
if home_medsform == 'embedding_attention':
    hm_embed_flag =1
    col_names = [col for col in home_meds_embedded.columns if 'V' in col]
    home_meds_embedded.fillna(0, inplace=True)
    home_meds_embedded['med_pos'] = [item for idx in home_meds_embedded.groupby(by='orlogid_encoded')['rxcui'].count()
                                     for item in range(idx)]
    home_meds_embedded1 = new_index.merge(home_meds_embedded, on="orlogid_encoded", how="left").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
    home_meds_embedded1.fillna(0, inplace=True)  # setting the value for the ones that were added later


bow_input = bow_input.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(),preops.index.max()+1)),fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
bow_cols = [col for col in bow_input.columns if 'BOW' in col]
bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
bow_input.fillna(0, inplace=True)

very_dense_flow = very_dense_flow.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                          axis=1).rename(
    {"new_person": "person_integer"}, axis=1)
other_intra_flow_wlabs = other_intra_flow_wlabs.merge(new_index, on="orlogid_encoded", how="inner").drop(
    ["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)

if drug_words is not None:
    drug_words = drug_words.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                    axis=1).rename(
        {"new_person": "person_integer"}, axis=1)

if drug_med_ids is not None:
    drug_med_ids = drug_med_ids.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                        axis=1).rename(
        {"new_person": "person_integer"}, axis=1)

outcome_df.drop(["orlogid_encoded"], axis=1, inplace=True)
outcome_df.reset_index(inplace=True)
outcome_df.rename({"index": "person_integer"}, axis=1, inplace=True)

## I suppose these could have sorted differently
## TODO apparently, torch.from_numpy shares the memory buffer and inherits type
index_med_ids = torch.tensor(drug_med_ids[['person_integer', 'time', 'drug_position']].values, dtype=int)
index_med_dose = torch.tensor(drug_dose[['person_integer', 'time', 'drug_position']].values, dtype=int)
value_med_dose = torch.tensor(drug_dose['dose'].astype('float').values, dtype=float)
value_med_unit = torch.tensor(drug_dose['med_unit_comb'].values, dtype=int)

add_unit = 0 in value_med_unit.unique()
dense_med_units = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_unit + add_unit,
                                          dtype=torch.int32)
dense_med_dose = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_dose, dtype=torch.float32)

if drugNamesNo == True:
    value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
    add_med = 0 in value_med_ids.unique()
    dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0, 1), value_med_ids + add_med,
                                            dtype=torch.int32)
else:  ## not considered
    drug_words.dropna(axis=0, inplace=True)
    # convert name and unit+dose data seperately into the required format
    drug_words['time'] = drug_words['time'].astype('int64')
    drug_words['person_integer'] = drug_words['person_integer'].astype('int')
    index_med_names = torch.tensor(drug_words[['person_integer', 'time', 'drug_position', 'word_position']].values,
                                   dtype=int)
    value_med_name = torch.tensor(drug_words['word_integer'].values, dtype=int)
    add_name = 0 in value_med_name.unique()
    dense_med_names = torch.sparse_coo_tensor(torch.transpose(index_med_names, 0, 1),
                                              value_med_name + add_name, dtype=torch.int32).to_dense()


""" TS flowsheet proprocessing """

# need to convert the type of orlogid_encoded from object to int
other_intra_flow_wlabs['person_integer'] = other_intra_flow_wlabs['person_integer'].astype('int')
very_dense_flow['person_integer'] = very_dense_flow['person_integer'].astype('int')

index_med_other_flow = torch.tensor(other_intra_flow_wlabs[['person_integer', 'timepoint', 'measure_index']].values,
                                    dtype=int)
value_med_other_flow = torch.tensor(other_intra_flow_wlabs['VALUE'].values)
flowsheet_other_flow = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                               value_med_other_flow, dtype=torch.float32)

index_med_very_dense = torch.tensor(very_dense_flow[['person_integer', 'timepoint', 'measure_index']].values, dtype=int)
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


if (trainTime):
    # currently sacrificing 5 data points in the valid set and using the test set to finally compute the auroc etc
    preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = pps.preprocess_train(preops, skipPreops, task,
                                                                                                  y_outcome=outcome_df[
                                                                                                      "outcome"].values,
                                                                                                  binary_outcome=binary_outcome,
                                                                                                  valid_size=0.00005)
else:
    # reading metadata file generated during training time
    md_f = open('/output/preops_metadata.json')
    metadata = json.load(md_f)
    if task == 'postop_los' or task == 'endofcase':
        preops_te = pps.preprocess_inference(preops, skipPreops, metadata)
    else:
        preops_te = pps.preprocess_inference(preops, skipPreops, metadata)



# total_flowsheet_measures = 2 * total_flowsheet_measures
preop_mask_counter = 0
num_preop_features = preops_tr.shape[1]

if(includeMissingnessMasks):
    """  Masks for preops and very dense flowsheet seperation based on train test; will double the dimension of flowsheet """

    # mask for very dense
    mask_flowsheet_very_dense = torch.sparse_coo_tensor(flowsheet_very_dense_sparse_form._indices(), np.ones(len(flowsheet_very_dense_sparse_form._values())),
                                          flowsheet_very_dense_sparse_form.size()).to_dense()

    total_flowsheet_measures = 2*total_flowsheet_measures


    if task == 'endofcase':
        preops_tr_mask = pd.concat([preops_mask.iloc[train_index], preops_mask.iloc[train_index]])
        preops_te_mask = pd.concat([preops_mask.iloc[test_index], preops_mask.iloc[test_index]])
        very_dense_tr_mask = torch.vstack([mask_flowsheet_very_dense[train_index, :, :]] * 2)
        very_dense_te_mask = torch.vstack([mask_flowsheet_very_dense[test_index, :, :]] * 2)
    else:
        preops_tr_mask = preops_mask.iloc[train_index]
        preops_te_mask = preops_mask.iloc[test_index]
        very_dense_tr_mask = mask_flowsheet_very_dense[train_index, :, :]
        very_dense_te_mask = mask_flowsheet_very_dense[test_index, :, :]

    preop_mask_counter = 1

if home_medsform == 'embedding_attention':
    index_HM_med_ids = torch.tensor(home_meds_embedded1[['person_integer','med_pos']].values, dtype=int)
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
    hm_input_dim= len(home_meds_final.columns)

print("Passed all the data processing stage")


if task == 'endofcase':  ##I only included the first two timepoints; doing the rest requires either excluding cases so that all 4 are defined or more complex indexing
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
    data_tr = [
        torch.tensor(preops_tr.to_numpy(), dtype=torch.float32),
        torch.tensor(endtimes.iloc[train_index]["endtime"].values, dtype=int),
        torch.tensor(bow_input.iloc[train_index].to_numpy(), dtype=torch.float32),
        hm_tr,
        torch.index_select(dense_med_ids, 0, torch.tensor(train_index)).coalesce(),
        torch.index_select(dense_med_dose, 0, torch.tensor(train_index)).coalesce(),
        torch.index_select(dense_med_units, 0, torch.tensor(train_index)).coalesce(),
        flowsheet_very_dense[train_index, :, :],
        torch.index_select(flowsheet_other_flow, 0, torch.tensor(train_index)).coalesce(),
        torch.tensor(outcome_df.iloc[train_index]["outcome"].values)
    ]
    data_te = [
        torch.tensor(preops_te.to_numpy(), dtype=torch.float32),
        torch.tensor(endtimes.iloc[test_index]["endtime"].values, dtype=int),
        torch.tensor(bow_input.iloc[test_index].to_numpy(), dtype=torch.float32),
        hm_te,
        torch.index_select(dense_med_ids, 0, torch.tensor(test_index)).coalesce(),
        torch.index_select(dense_med_dose, 0, torch.tensor(test_index)).coalesce(),
        torch.index_select(dense_med_units, 0, torch.tensor(test_index)).coalesce(),
        flowsheet_very_dense[test_index, :, :],
        torch.index_select(flowsheet_other_flow, 0, torch.tensor(test_index)).coalesce(),
        torch.tensor(outcome_df.iloc[test_index]["outcome"].values)
    ]
    data_va = [
        torch.tensor(preops_val.to_numpy(), dtype=torch.float32),
        torch.tensor(endtimes.iloc[valid_index]["endtime"].values, dtype=int),
        torch.tensor(bow_input.iloc[valid_index].to_numpy(), dtype=torch.float32),
        hm_val,
        torch.index_select(dense_med_ids, 0, torch.tensor(valid_index)).coalesce(),
        torch.index_select(dense_med_dose, 0, torch.tensor(valid_index)).coalesce(),
        torch.index_select(dense_med_units, 0, torch.tensor(valid_index)).coalesce(),
        flowsheet_very_dense[valid_index, :, :],
        torch.index_select(flowsheet_other_flow, 0, torch.tensor(valid_index)).coalesce(),
        torch.tensor(outcome_df.iloc[valid_index]["outcome"].values)
    ]

device = torch.device('cuda')
# breakpoint()
model = preop_flow_med_bow_model.TS_lstm_Med_index(
  v_units=vocab_len_units,
  v_med_ids=vocab_len_med_ids,
  e_dim_med_ids=lstmMedEmbDim,
  e_dim_units=lstmUnitExpand,
  preops_init_med=preopInitLstmMed,
  preops_init_flow=preopInitLstmFlow,
  lstm_hid=lstmMedWidth,
  lstm_flow_hid=lstmFlowWidth,
  lstm_num_layers=lstmMedDepth,
  lstm_flow_num_layers=lstmFlowDepth,
  bilstm_med = BilstmMed,
Att_MedAgg = AttentionMedAgg,
AttMedAgg_Heads = AttMedAggHeads,
  bilstm_flow = BilstmFlow,
  linear_out=1,
  p_idx_med_ids=0,  # putting these 0 because the to dense sets everything not available as 0
  p_idx_units=0,
  p_time=lstmMedDrop,
  p_flow=lstmFlowDrop,
  p_rows=lstmRowDrop,
  p_final=finalDrop,
  binary= binary_outcome,
  hidden_units=preopsWidth,
  hidden_units_final=preopsWidthFinal,
  hidden_depth=preopsDepth,
  finalBN=finalBN,
  input_shape=data_tr[0].shape[1] + preop_mask_counter*num_preop_features, # this is done so that I dont have to write a seperate condition for endofcase where the current time is being appended to preops
  hidden_units_bow=bowWidth,
  hidden_units_final_bow=bowWidthFinal,
  hidden_depth_bow=bowDepth,
  input_shape_bow=len(bow_input.columns),
    Att_HM_Agg = AttentionHhomeMedsAgg,
    hidden_units_hm=hmWidth,
    hidden_units_final_hm=hmWidthFinal,
    hidden_depth_hm=hmDepth,
    input_shape_hm=hm_input_dim,
  num_flowsheet_feat=total_flowsheet_measures,
    weight_decay_preopsL2=preopsL2,
    weight_decay_preopsL1=preopsL1,
    weight_decay_bowL2=bowL2,
    weight_decay_bowL1=bowL1,
    weight_decay_hmL2=hmL2,
    weight_decay_hmL1=hmL1,
    weight_decay_LSTMmedL2=lstmMedL2,
    weight_decay_LSTMflowL2=lstmFlowL2,
    weightInt = XavOrthWeightInt,
    group_start_list=group_start,
    group_end_list=group_end
  ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

# lr scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LRPatience, verbose=True, factor=learningRateFactor)

# initializing the loss function
if not binary_outcome:
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
else:
  criterion = torch.nn.BCELoss()

total_train_loss = []
total_test_loss = []

model_cachename = hashlib.md5(json.dumps(vars(args)).encode())

PATH = "model_" + task + "_" + str(model_cachename.hexdigest()) + ".pth"

updating_lr = learningRate
best_metric = 1000 # some large number
lr_schedular_epoch_dict = {}
lr_schedular_epoch_dict[0] = updating_lr

for epoch in range(epochs):
  loss_tr = 0
  loss_tr_cls = 0
  model.train()
  ## the default __getitem__ is like 2 orders of magnitude slower
  shuffle_index = torch.randperm(n=data_tr[0].shape[0])
  if (overSampling == True) and (task != 'endofcase'):
      pos_idx = (data_tr[-1] == 1).nonzero()
      neg_idx = (data_tr[-1] == 0).nonzero()
      if batchSize % 2 == 0:  # this is done because it was creating a problem when the batchsize was an odd number
          nbatch = neg_idx.shape[0] // int(batchSize / 2)
      else:
          nbatch = neg_idx.shape[0] // math.ceil(batchSize / 2)

  else:
      nbatch = data_tr[0].shape[0] // batchSize
  for i in range(nbatch):
      # breakpoint()
      if (overSampling == True) and (task != 'endofcase'):
          if batchSize % 2 == 0:
              neg_indexbatch = neg_idx[range(i * int(batchSize / 2), (i + 1) * int(batchSize / 2))]
          else:
              neg_indexbatch = neg_idx[range(i * math.ceil(batchSize / 2), (i + 1) * math.ceil(batchSize / 2))]
          p = torch.from_numpy(np.repeat([1 / len(pos_idx)], len(pos_idx)))
          pos_indexbatch = pos_idx[p.multinomial(num_samples=int(batchSize / 2),
                                                 replacement=True)]  # this is sort of an equivalent of numpy.random.choice
          these_index = torch.vstack([neg_indexbatch, pos_indexbatch]).reshape([batchSize])
      else:
          these_index = shuffle_index[range(i * batchSize, (i + 1) * batchSize)]
      ## this collate method is pretty inefficent for this task but works with the generic DataLoader method
      local_data =[ torch.index_select(x,  0 ,  these_index )  for x in data_tr]
      if task == 'endofcase':
          local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
              len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
          local_data[0][:, -1] = local_data[
              1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
          local_data[-1] = torch.from_numpy(np.repeat([1, 0], [int(batchSize / 2), batchSize - int(batchSize / 2)]))
      if (includeMissingnessMasks):  # appending the missingness masks in training data
          labels = local_data[-1]
          local_data.pop(-1)
          local_data.append(torch.tensor(preops_tr_mask.iloc[these_index].to_numpy(), dtype=torch.float32))
          local_data.append(very_dense_tr_mask[these_index, :, :])
          sparse_mask = torch.sparse_coo_tensor(local_data[8]._indices(), np.ones(len(local_data[8]._values())),
                                                local_data[8].size())
          local_data.append(sparse_mask)
          local_data.append(labels)  # to keep the code general adding it again here
      data_train = preop_flow_med_bow_model.collate_time_series(
          [[x[i] for x in local_data] for i in range(local_data[0].shape[0])])
      # need to convert the flowsheets to float from double because in the model the hidden initialitzation are float
      data_train[0][7] = data_train[0][7].float()

      if True:  # temporary for ablation studies
          ## inputs are assumed to be: [preops, durations, bow, med_index, med_dose, med_unit, flow_sheetcombined]
          if testcondition == "preopOnly":
              data_train[0][4] = torch.zeros(data_train[0][4].shape)
              data_train[0][5] = torch.zeros(data_train[0][5].shape)
              data_train[0][6] = torch.zeros(data_train[0][6].shape)
              data_train[0][7] = torch.zeros(data_train[0][7].shape)
          if testcondition == 'MedOnly':
              data_train[0][0] = torch.zeros(data_train[0][0].shape)
              data_train[0][2] = torch.zeros(data_train[0][2].shape)
              data_train[0][3] = torch.zeros(data_train[0][3].shape)
              data_train[0][7] = torch.zeros(data_train[0][7].shape)
          if testcondition == 'FlowOnly':
              data_train[0][0] = torch.zeros(data_train[0][0].shape)
              data_train[0][2] = torch.zeros(data_train[0][2].shape)
              data_train[0][3] = torch.zeros(data_train[0][3].shape)
              data_train[0][4] = torch.zeros(data_train[0][4].shape)
              data_train[0][5] = torch.zeros(data_train[0][5].shape)
              data_train[0][6] = torch.zeros(data_train[0][6].shape)
          if testcondition == 'MedFlow':
              data_train[0][0] = torch.zeros(data_train[0][0].shape)
              data_train[0][2] = torch.zeros(data_train[0][2].shape)
              data_train[0][3] = torch.zeros(data_train[0][3].shape)

      for data_index in [0,2,3,4,5,6,7]:
        data_train[0][data_index] = data_train[0][data_index].to(device)
      ## TODO: this hurts me aesthetically; it would be nice to have the collate function have an option for this. the reason is to avoid passing device as an argument to forward pass. Currently I have the model detect what device the parameters are on to determine where to put the initialized LSTM parameters, so it is device agnostic. I will have to look into the function constructor type class with init and call
      # reset the gradients back to zero as PyTorch accumulates gradients on subsequent backward passes
      optimizer.zero_grad()
      if True:
        y_pred, reg_loss = model(*data_train[0])
        cls_loss_tr = criterion(y_pred.squeeze(-1), data_train[1].float().to(device)).float()
        train_loss = cls_loss_tr + reg_loss
        train_loss.backward()
        optimizer.step()
        loss_tr += train_loss.item()
        loss_tr_cls += cls_loss_tr.item()
  loss_tr = loss_tr / data_tr[0].shape[0]
  loss_tr_cls = loss_tr_cls/ data_tr[0].shape[0]

  loss_te = 0
  loss_te_cls = 0
  with torch.no_grad():
    model.eval()
    true_y_test = []
    pred_y_test = []
    nbatch = data_te[0].shape[0] // batchSize
    for i in range(nbatch):
        these_index = torch.tensor(list(range(i * batchSize, (i + 1) * batchSize)), dtype=int)
        local_data = [torch.index_select(x, 0, these_index) for x in data_te]
        if task == 'endofcase':
            local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
            local_data[0][:, -1] = local_data[
                1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
            local_data[-1] = torch.from_numpy(np.repeat([1, 0], [int(batchSize / 2), batchSize - int(batchSize / 2)]))
        if (includeMissingnessMasks):  # appending the missingness masks in test data
            labels = local_data[-1]
            local_data.pop(-1)
            local_data.append(torch.tensor(preops_te_mask.iloc[these_index].to_numpy(), dtype=torch.float32))
            local_data.append(very_dense_te_mask[these_index, :, :])
            sparse_mask = torch.sparse_coo_tensor(local_data[8]._indices(), np.ones(len(local_data[8]._values())),
                                                  local_data[8].size())
            local_data.append(sparse_mask)
            local_data.append(labels)  # to keep the code general adding it again here
        data_valid = preop_flow_med_bow_model.collate_time_series(
            [[x[i] for x in local_data] for i in range(local_data[0].shape[0])])
        data_valid[0][7] = data_valid[0][7].float()

        if True:  # temporary for ablation studies
            ## inputs are assumed to be: [preops, durations, bow, med_index, med_dose, med_unit, flow_dense, flow_sparse (optional), labels (optional)]
            if testcondition == "preopOnly":
                data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
                data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
                data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
                data_valid[0][7] = torch.zeros(data_valid[0][7].shape)
            if testcondition == 'MedOnly':
                data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
                data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
                data_valid[0][7] = torch.zeros(data_valid[0][7].shape)
            if testcondition == 'FlowOnly':
                data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
                data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
                data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
                data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
                data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
            if testcondition == 'MedFlow':
                data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
                data_valid[0][3] = torch.zeros(data_valid[0][3].shape)

        for data_index in [0,2,3,4,5,6,7]:
          data_valid[0][data_index] = data_valid[0][data_index].to(device)
        y_pred, reg_loss = model(*data_valid[0])
        cls_loss_te = criterion(y_pred.squeeze(-1),data_valid[1].float().to(device)).float()
        test_loss = cls_loss_te + reg_loss
        loss_te += test_loss.item()
        loss_te_cls += cls_loss_te.item()

        # values from the last epoch; it will get overwritten
        # using test data only instead of validation data for evaluation currently because the validation will be done on a seperate data
        true_y_test.append(data_valid[1].float().detach().numpy())
        pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())

    loss_te = loss_te / data_te[0].shape[0]
    loss_te_cls = loss_te_cls/ data_te[0].shape[0]

    if best_metric > loss_te_cls:
        best_metric = loss_te_cls
        # torch.save(model.state_dict(), PATH)
        pred_y_test_best = pred_y_test

    # display the epoch training and test loss
    print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, epochs, loss_tr_cls,loss_te_cls) )
    total_train_loss.append(loss_tr)
    total_test_loss.append(loss_te)

  scheduler.step(loss_te_cls)

  if optimizer.param_groups[0]['lr'] != updating_lr:
      updating_lr = optimizer.param_groups[0]['lr']
      lr_schedular_epoch_dict[epoch] = updating_lr

  # print("current lr ", optimizer.param_groups[0]['lr'])
  # 1e-8 is obtained by multiplying 1e-3 by (0.25)^5 so to make it general we can have initial_learning_rate * (learningRateFactor)^5
  if optimizer.param_groups[0]['lr'] <= learningRate * np.power(learningRateFactor, 5):  # hardcoding for now because our schedule is such that 10**-3 * (1, 1/4, 1/16, 1/64, 1/256, 1/1024, 0) with an initial rate of 10**-3 an learning rate factor of 0.25
      print("inside the early stopping loop")
      print("best validation loss ", best_metric)
      # epoch =epochs
      # true_y_test.append(data_valid[1].float().detach().numpy())
      # pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())
      break

model.eval()

true_y_test = np.concatenate(true_y_test)
pred_y_test = np.concatenate(pred_y_test_best)
if not binary_outcome:
    print(" Number of epochs that ran ", epoch)
    plt.scatter(np.array(true_y_test),np.array(pred_y_test))
    plt.title("True outcome value vs predictions on test set " + str(testcondition))
    plt.xlabel(" True " + str(task))
    plt.ylabel(" Predicted " + str(task))
    plt.savefig("/output/True_vs_Pred_"+str(task)+ "_"+ str(testcondition)+".png")
    plt.close()
    corr_value = np.round(np.corrcoef(np.array(true_y_test), np.array(pred_y_test))[1, 0], 3)
    print(str(task) + " prediction with correlation ", corr_value)
    r2value = r2_score(np.array(true_y_test), np.array(pred_y_test))
    temp_df = pd.DataFrame(columns=['true_value', 'pred_value'])
    temp_df['true_value'] = np.array(true_y_test)
    temp_df['pred_value'] = np.array(pred_y_test)
    temp_df['abs_diff'] = abs(temp_df['true_value'] - temp_df['pred_value'])
    mae_full = np.round(temp_df['abs_diff'].mean(), 3)
    print("MAE on the test set ", mae_full)
    q25, q7, q9 = temp_df['true_value'].quantile([0.25, 0.7, 0.9])

    firstP_data = temp_df.query('true_value<={high}'.format(high=q25))
    secondP_data = temp_df.query('{low}<true_value<={high}'.format(low=q25, high=q7))
    thirdP_data = temp_df.query('{low}<true_value<={high}'.format(low=q7, high=q9))
    fourthP_data = temp_df.query('{low}<true_value'.format(low=q9))

    mae_dict = {'<' + str(np.round(q25, decimals=1)): firstP_data['abs_diff'].mean(),
                str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): secondP_data['abs_diff'].mean(),
                str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): thirdP_data['abs_diff'].mean(),
                str(np.round(q9, decimals=1)) + "<": fourthP_data['abs_diff'].mean()}

    stratifying_point_dict = {'<' + str(np.round(q25, decimals=1)): '<' + str(np.round(q25, decimals=1)),
                              str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): str(
                                  np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)),
                              str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): str(
                                  np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)),
                              str(np.round(q9, decimals=1)) + "<": str(np.round(q9, decimals=1)) + "<"}

    end_time = datetime.now()  # only writing part is remaining in the code to time
    timetaken = end_time-start_time

    csvdata = {
        'hp': json.dumps(vars(args)),
        'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
        'corr': corr_value,
        'R2': r2value,
        'MAE': mae_full,
        'Stratifying_points': stratifying_point_dict,
        'Stratified_MAE': mae_dict,
        'git': args.git,
        'name': args.nameinfo,
        'target': args.task,
        'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
        'lr_change_epoch': json.dumps(lr_schedular_epoch_dict),
        'time': timetaken
    }

    csvdata = pd.DataFrame(csvdata)
    outputcsv = os.path.join('/output/', args.outputcsv)
    if (os.path.exists(outputcsv)):
        csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
    else:
        csvdata.to_csv(outputcsv, header=True, index=False)

    ## TODO: output saving to csv for non-binary
else:
    test_auroc = roc_auc_score(true_y_test, pred_y_test)
    test_auprc = average_precision_score(true_y_test, pred_y_test)
    print(" Number of epochs that ran ", epoch)
    print("Test AUROC and AUPRC values are ", np.round(test_auroc, 4), np.round(test_auprc, 4))
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(true_y_test, pred_y_test, drop_intermediate=False)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(true_y_test, pred_y_test)
    # interpolation in ROC
    mean_fpr = np.linspace(0, 1, 100)
    tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
    mean_fpr = np.round(mean_fpr, decimals=2)
    print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))

    if task=='endofcase':
        outcome_rate = 0.5  # this is hardcoded here
    else:
        outcome_rate = np.round(sum(outcome_df["outcome"].values)/len(outcome_df), decimals=4)

    end_time = datetime.now()  # only writing part is remaining in the code to time
    timetaken = end_time-start_time
    print("time taken to run the complete training script", timetaken)

    csvdata = {
        'hp': json.dumps(vars(args)),
        'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
        'outcome_rate': outcome_rate,
        'AUROC': test_auroc,
        'AUPRC': test_auprc,
        'Sensitivity': tpr_inter[np.where(mean_fpr == 0.10)],
        'git': args.git,
        'name': args.nameinfo,
        'target': args.task,
        'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S"),
        'lr_change_epoch': json.dumps(lr_schedular_epoch_dict),
        'time': timetaken
    }

    csvdata = pd.DataFrame(csvdata)
    outputcsv = os.path.join('/output/', args.outputcsv)
    if (os.path.exists(outputcsv)):
        csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
    else:
        csvdata.to_csv(outputcsv, header=True, index=False)

