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
from scipy.stats.stats import pearsonr
from datetime import datetime
import matplotlib.pyplot as plt
import Preops_processing as pps
import preop_flow_med_bow_model
import pickle

# presetting the number of threads to be used
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
# torch.cuda.set_per_process_memory_fraction(1.0, device=None)

# TODO: pick better defaults
# TODO: make the same modifications to using the word-sequence
# TODO: MLP for final state (currently fixed at 2 layer)

# starting time of the script
start_time = datetime.now()

parser = argparse.ArgumentParser(description='TS modular different model training')

## modalities to select
parser.add_argument('--preops', action="store_true",
                    help='Whether to add preops and bow to ts representation')
parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problem list representation to the lstm/transformer time series output')
parser.add_argument('--homemeds', action="store_true",
                    help='Whether to add homemeds to ts representation')
parser.add_argument('--meds', action="store_true",
                    help='Whether to add meds to ts representation')
parser.add_argument('--flow', action="store_true",
                    help='Whether to add flowsheets to ts representation')
parser.add_argument('--alerts', action="store_true",
                    help='Whether to add alerts to ts representation')

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

## for the past medical history before concat to ts output
parser.add_argument("--pmhform", default='embedding_sum') # options { 'embedding_sum', 'embedding_attention'}
parser.add_argument("--AttentionPmhAgg", default=False, action='store_true') # this needs to be true when embedding_attention is active in the above line
parser.add_argument("--AttPmhAggHeads", default=2, type=int)
parser.add_argument("--pmhDepth",  default=3, type=int) #
parser.add_argument("--pmhWidth",  default=300, type=int) #
parser.add_argument("--pmhWidthFinal",  default=10, type=int) #
parser.add_argument("--pmhL2",  default=0.2, type=float)
parser.add_argument("--pmhL1",  default=0.1, type=float)

## for the problem list before concat to ts output
parser.add_argument("--problistform", default='embedding_sum') # options {'embedding_sum', 'embedding_attention'}
parser.add_argument("--AttentionProblistAgg", default=False, action='store_true') # this needs to be true when embedding_attention is active in the above line
parser.add_argument("--problistDepth",  default=3, type=int) #
parser.add_argument("--problistWidth",  default=300, type=int) #
parser.add_argument("--problistWidthFinal",  default=10, type=int) #
parser.add_argument("--problistL2",  default=0.2, type=float)
parser.add_argument("--problistL1",  default=0.1, type=float)

## for the homemeds
parser.add_argument("--home_medsform", default='embedding_sum') # options {'ohe', 'embedding_sum', 'embedding_attention'}
parser.add_argument("--AttentionHhomeMedsAgg", default=False, action='store_true') # this needs to be true when embedding_attention is active in the above line
parser.add_argument("--AttentionHomeMedsAggHeads", default=2, type=int)
parser.add_argument("--hmDepth", default=5, type=int)  #
parser.add_argument("--hmWidth", default=400, type=int)  #
parser.add_argument("--hmWidthFinal", default=10, type=int)  #
parser.add_argument("--hmL2", default=0.01, type=float)
parser.add_argument("--hmL1", default=0.01, type=float)

## for processing medication IDs (or the post-embedding words)
parser.add_argument("--lstmMedEmbDim",  default=6, type=int) #
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

## for model type for time series
parser.add_argument("--modelType", default='lstm') # options {'lstm', 'transformer'}
## the following arguments are only relevant for the transformer model type
parser.add_argument("--e_dim_flow", default=10, type=int) # embedding id for flowsheet before passing them to the transformer
parser.add_argument("--AttTSDepth", default=4, type=int) # depth of the transformer that will take the time series input
parser.add_argument("--AttTSHeads", default=2, type=int) # number of heads in the transformer encoder layer
parser.add_argument("--cnn_before_Att", default=True, action='store_true')
parser.add_argument("--kernel_size_conv", default=5, type=int)
parser.add_argument("--stride_conv", default=2, type=int)
parser.add_argument("--ats_dropout", default=0.3, type=float)
## for the MLP combining preop and LSTM/Transformer based outputs
parser.add_argument("--finalDrop",  default=.4, type=float)  #
parser.add_argument("--finalBN", default=False, action='store_true') #


## learning parameters
parser.add_argument("--batchSize",  default=32, type=int) #
parser.add_argument("--learningRate",  default=1e-3, type=float) #
parser.add_argument("--learningRateFactor",  default=0.1, type=float) #
parser.add_argument("--LRPatience",  default=2, type=int) #
parser.add_argument("--epochs",  default=5, type=int) #
parser.add_argument("--XavOrthWeightInt", default=True, action='store_true')  # changes torch's weight initialization to xavier and orthogonal


## task and setup parameters
parser.add_argument("--task",  default="icu") #
parser.add_argument("--drugNamesNo", default=True,  action='store_true') #
parser.add_argument("--trainTime", default=True, action='store_true')
parser.add_argument("--randomSeed", default=100, type=int )
parser.add_argument("--includeMissingnessMasks", default=False, action='store_true')
parser.add_argument("--overSampling", default=True, action='store_true') # keep it as False when task is endofcase
parser.add_argument("--bestModel", default="False",
                    help='True when the best HP tuned settings are used on the train+valid setup')  #


## output parameters
parser.add_argument("--git",  default="") # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo",  default="") #
parser.add_argument("--outputcsv",  default="") #

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing

all_modality_list = ['flow', 'meds', 'alerts', 'pmh', 'problist', 'homemeds',  'preops', 'cbow']
modality_to_use = []
if eval('args.preops') == True:
    modality_to_use.append('preops')
    modality_to_use.append('cbow')

if eval('args.pmhProblist') == True:
    modality_to_use.append('pmh')
    # modality_to_use.append('problist')  ## this doesn't exist in MV

if eval('args.homemeds') == True:
    modality_to_use.append('homemeds')

if eval('args.flow') == True:
    modality_to_use.append('flow')

if eval('args.meds') == True:
    modality_to_use.append('meds')

# data_dir = '/mnt/ris/ActFastExports/v1.3.3/mv_data/'
data_dir = '/input/' +'mv_data/'

# out_dir = './'
out_dir = '/output/'

preops = pd.read_csv(data_dir + 'mv_preop.csv')
preops = preops.drop_duplicates(subset=['orlogid_encoded'])
outcomes = pd.read_csv(data_dir + 'outcomes_mv.csv')
outcomes = outcomes.dropna(subset=['orlogid_encoded'])
end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times_wave0.feather')

end_of_case_times = end_of_case_times[['orlogid_encoded', 'endtime']]


# end_of_case_times = feather.read_feather(data_dir + 'end_of_case_times.feather')
regression_outcome_list = ['postop_los', 'survival_time', 'readmission_survival', 'total_blood', 'postop_Vent_duration', 'n_glu_high',
                           'low_sbp_time','aoc_low_sbp', 'low_relmap_time', 'low_relmap_aoc', 'low_map_time',
                           'low_map_aoc', 'timew_pain_avg_0', 'median_pain_0', 'worst_pain_0', 'worst_pain_1',
                           'opioids_count_day0', 'opioids_count_day1']
binary_outcome = args.task not in regression_outcome_list

config = dict(
    linear_out=1
)
config['binary'] = binary_outcome

if args.task=='icu':
    outcomes = outcomes.dropna(subset=['ICU'])
outcomes = outcomes.sort_values(by='survival_time').drop_duplicates(subset=['orlogid_encoded'], keep='last')

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

if args.task == 'endofcase':
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

binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac', 'postop_trop_crit', 'postop_trop_high', 'post_dialysis', 'n_glucose_low']

if args.task in regression_outcome_list:
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

    outcome_df = outcomes[['orlogid_encoded', args.task]]
elif args.task in binary_outcome_list:
    if args.task == 'VTE':
        temp_outcome = outcomes[['orlogid_encoded']]
        temp_outcome[args.task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[args.task] == 2, args.task] = 1
    elif args.task == 'n_glucose_low':
        temp_outcome = outcomes[['orlogid_encoded', args.task]]
        temp_outcome[args.task] = temp_outcome[args.task].fillna(0)
        temp_outcome[args.task] = np.where(temp_outcome[args.task]>0, 1, 0)
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
new_index = outcome_df["orlogid_encoded"].copy().reset_index().rename({"index": "new_person"}, axis=1)   # this df basically reindexes everything so from now onwards orlogid_encoded is an integer

endtimes = end_of_case_times.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                     axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)

preops = preops.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename(
    {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)


if 'preops' in modality_to_use:
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
    preops = preops.drop(columns=to_drop_old_pmh_with_others) # "['pre_aki_status', 'preop_ICU', 'AnestStop']
    bow_input = pd.read_csv(data_dir + 'cbow_proc_text_mv.csv')

    bow_input = bow_input.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(
        ["orlogid_encoded"], axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    bow_cols = [col for col in bow_input.columns if 'BOW' in col]
    bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
    bow_input.fillna(0, inplace=True)

    config['hidden_units'] = args.preopsWidth
    config['hidden_depth'] = args.preopsDepth
    config['weight_decay_preopsL2'] = args.preopsL2
    config['weight_decay_preopsL1'] = args.preopsL1

    config['hidden_units_bow'] = args.bowWidth
    config['hidden_units_final_bow'] = args.bowWidthFinal
    config['hidden_depth_bow'] = args.bowDepth
    config['weight_decay_bowL2'] = args.bowL2
    config['weight_decay_bowL1'] = args.bowL1

if 'homemeds' in modality_to_use:
    # home meds reading and processing
    home_meds = pd.read_csv(data_dir + 'home_med_cui_mv.csv', low_memory=False)
    Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')

    # home_meds[["orlogid_encoded","rxcui"]].groupby("orlogid_encoded").agg(['count'])
    # home_med_dose = home_meds.pivot(index='orlogid_encoded', columns='rxcui', values='Dose')
    home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded',
                                                  'rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
    home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left', on='rxcui')
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

    home_meds_ohe = home_meds_ohe.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"],
                                                                                                  axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    home_meds_ohe.fillna(0, inplace=True)  # setting the value for the ones that were added later

    home_meds_sum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
    home_meds_sum = home_meds_sum.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(
        list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"],
                                                                                                  axis=1).rename(
        {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(
        ["person_integer"], axis=1)
    home_meds_sum.fillna(0, inplace=True)  # setting the value for the ones that were added later

    if args.home_medsform == 'ohe':
        home_meds_final = home_meds_ohe
    if args.home_medsform == 'embedding_sum':
        home_meds_sum = home_meds_sum.drop(["rxcui"], axis=1)
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
        hm_input_dim = len(col_names)
    else:
        hm_input_dim = len(home_meds_final.columns)

    config['hidden_units_hm'] = args.hmWidth
    config['hidden_units_final_hm'] = args.hmWidthFinal
    config['hidden_depth_hm'] = args.hmDepth
    config['weight_decay_hmL2'] = args.hmL2
    config['weight_decay_hmL1'] = args.hmL1
    config['input_shape_hm'] = hm_input_dim
    config['Att_HM_Agg'] = args.AttentionHhomeMedsAgg
    config['Att_HM_agg_Heads'] = args.AttentionHomeMedsAggHeads

if 'pmh' in modality_to_use:
    pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert_mv.csv')

    if args.pmhform == 'embedding_sum':
        pmh_emb_sb = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        pmh_emb_sb_final = pmh_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
            {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)

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
        pmh_input_dim = len(col_names)

    config['hidden_units_pmh'] = args.pmhWidth
    config['hidden_units_final_pmh'] = args.pmhWidthFinal
    config['hidden_depth_pmh'] = args.pmhDepth
    config['weight_decay_pmhL2'] = args.pmhL2
    config['weight_decay_pmhL1'] = args.pmhL1
    config['input_shape_pmh'] = pmh_input_dim
    config['Att_pmh_Agg'] = args.AttentionPmhAgg
    config['AttPmhAgg_Heads'] = args.AttPmhAggHeads

# if 'problist' in modality_to_use:
#     prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')
#
#     if args.problistform == 'embedding_sum':
#         prob_list_emb_sb = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
#         prob_list_emb_sb_final = prob_list_emb_sb.merge(new_index, on="orlogid_encoded", how="inner").set_index('new_person').reindex(list(range(preops.index.min(), preops.index.max() + 1)), fill_value=0).reset_index().drop(["orlogid_encoded"], axis=1).rename(
#             {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True).drop(["person_integer"], axis=1)
#
#         problist_input_dim = len(prob_list_emb_sb_final.columns)
#
#     if args.problistform == 'embedding_attention':
#         col_names = [col for col in prob_list_emb_sb.columns if 'sherbet' in col]
#         prob_list_emb_sb.fillna(0, inplace=True)
#         prob_list_emb_sb['pmh_pos'] = [item for idx in
#                                  prob_list_emb_sb.groupby(by='orlogid_encoded')['ICD_10_CODES'].count()
#                                  for item in range(idx)]
#         prob_list_emb_sb1 = new_index.merge(pmh_emb_sb, on="orlogid_encoded", how="left").drop(
#             ["orlogid_encoded"], axis=1).rename(
#             {"new_person": "person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
#         prob_list_emb_sb1.fillna(0, inplace=True)  # setting the value for the ones that were added later
#
#         index_problist_ids = torch.tensor(prob_list_emb_sb1[['person_integer', 'pmh_pos']].values, dtype=int)
#         value_problist_embed = torch.tensor(prob_list_emb_sb1[col_names].astype('float').values, dtype=float)
#         dense_problist_embedding = torch.sparse_coo_tensor(torch.transpose(index_problist_ids, 0, 1), value_problist_embed,
#                                                       dtype=torch.float32)
#         problist_input_dim = len(col_names)
#
#
#     config['hidden_units_problist'] = args.problistWidth
#     config['hidden_units_final_problist'] = args.problistWidthFinal
#     config['hidden_depth_problist'] = args.problistDepth
#     config['weight_decay_problistL2'] = args.problistL2
#     config['weight_decay_problistL1'] = args.problistL1
#     config['input_shape_problist'] = problist_input_dim
#     config['Att_problist_Agg'] = args.AttentionProblistAgg

if 'flow' in modality_to_use:
    # flowsheet data
    very_dense_flow = feather.read_feather(data_dir +"flow_ts/Imputed_very_dense_flow_wave0.feather")
    very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace=True)
    very_dense_flow = very_dense_flow.merge(end_of_case_times[['orlogid_encoded', 'endtime']], on="orlogid_encoded")
    very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint']]
    very_dense_flow.drop(["endtime"], axis=1, inplace=True)

    other_intra_flow_wlabs = feather.read_feather(data_dir +"flow_ts/Imputed_other_flow_wave0.feather")
    other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index, inplace=True)
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

    index_med_other_flow = torch.tensor(other_intra_flow_wlabs[['person_integer', 'timepoint', 'measure_index']].values, dtype=int)
    value_med_other_flow = torch.tensor(other_intra_flow_wlabs['VALUE'].values)
    flowsheet_other_flow = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                                   value_med_other_flow, dtype=torch.float32)

    index_med_very_dense = torch.tensor(very_dense_flow[['person_integer', 'timepoint', 'measure_index']].values,dtype=int)
    value_med_very_dense = torch.tensor(very_dense_flow['VALUE'].values)
    flowsheet_very_dense_sparse_form = torch.sparse_coo_tensor(torch.transpose(index_med_very_dense, 0, 1), value_med_very_dense, dtype=torch.float32)  ## this is memory heavy and could be skipped, only because it is making a copy not really because it is harder to store
    flowsheet_very_dense = flowsheet_very_dense_sparse_form.to_dense()
    flowsheet_very_dense = torch.cumsum(flowsheet_very_dense, dim=1)

    # trying to concatenate the two types of flowsheet tensors at the measure_index dimension
    # flowsheet_dense_comb = torch.cat((flowsheet_very_dense, flowsheet_other_flow), dim=2)
    total_flowsheet_measures = other_intra_flow_wlabs['measure_index'].unique().max() + 1 + very_dense_flow[
        'measure_index'].unique().max() + 1  # plus 1 because of the python indexing from 0

    config['preops_init_flow'] = args.preopInitLstmFlow
    config['lstm_flow_hid'] = args.lstmFlowWidth
    config['lstm_flow_num_layers'] = args.lstmFlowDepth
    config['bilstm_flow'] = args.BilstmFlow
    config['p_flow'] = args.lstmFlowDrop
    config['weight_decay_LSTMflowL2'] = args.lstmFlowL2

if 'meds' in modality_to_use:
    # reading the med files
    all_med_data = feather.read_feather(data_dir + 'med_ts/intraop_meds_filterd_wave0.feather')
    all_med_data.drop(all_med_data[all_med_data['time'] > 511].index, inplace=True)
    all_med_data.drop(all_med_data[all_med_data['time'] < 0].index, inplace=True)  # there are some negative time points  ## TODO: i think it had some meaning; check this
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

    if args.drugNamesNo == True:
        # drug_med_id_map = feather.read_feather(data_dir + 'med_ts/med_id_map.feather')
        drug_med_id_map = pd.read_csv(data_dir + 'med_ts/med_id_map.csv')
        drug_words = None
        word_id_map = None
    else:
        drug_words = feather.read_feather(data_dir + 'med_ts/drug_words.feather')
        drug_words.drop(drug_words[drug_words['timepoint'] > 511].index, inplace=True)
        word_id_map = feather.read_feather(data_dir + 'med_ts/word_id_map.feather')
        drug_med_id_map = None


    # drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'unit_integer',
    #                           'dose']]
    drug_dose = all_med_data[['orlogid_encoded', 'time', 'drug_position', 'med_unit_comb','dose']]  # replacing the unit_integer column by med_unit_comb column

    # unit_id_map = feather.read_feather(data_dir + 'med_ts/unit_id_map.feather')
    # vocab_len_units = len(unit_id_map)
    vocab_len_units = len(med_unit_unique_codes)  # replacing  len(unit_id_map) by len(med_unit_unique_codes)


    if args.drugNamesNo == False:
        vocab_len_words = len(word_id_map)
    else:
        vocab_len_med_ids = len(drug_med_id_map)  ## TODO: fix this once you know where the map file is; done
        # vocab_len_med_ids = 102


    drug_dose = drug_dose.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"], axis=1).rename({"new_person": "person_integer"}, axis=1)

    if drug_words is not None:
        drug_words = drug_words.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],
                                                                                         axis=1).rename(
            {"new_person": "person_integer"}, axis=1)

    if drug_med_ids is not None:
        drug_med_ids = drug_med_ids.merge(new_index, on="orlogid_encoded", how="inner").drop(["orlogid_encoded"],axis=1).rename( {"new_person": "person_integer"}, axis=1)

    ## I suppose these could have sorted differently
    ## TODO apparently, torch.from_numpy shares the memory buffer and inherits type
    index_med_ids = torch.tensor(drug_med_ids[['person_integer', 'time', 'drug_position']].values, dtype=int)
    index_med_dose = torch.tensor(drug_dose[['person_integer', 'time', 'drug_position']].values, dtype=int)
    value_med_dose = torch.tensor(drug_dose['dose'].astype('float').values, dtype=float)
    value_med_unit = torch.tensor(drug_dose['med_unit_comb'].values, dtype=int)

    add_unit = 0 in value_med_unit.unique()
    dense_med_units = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_unit + add_unit,dtype=torch.int32)
    dense_med_dose = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_dose, dtype=torch.float32)

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
        index_med_names = torch.tensor(drug_words[['person_integer', 'time', 'drug_position', 'word_position']].values,
                                       dtype=int)
        value_med_name = torch.tensor(drug_words['word_integer'].values, dtype=int)
        add_name = 0 in value_med_name.unique()
        dense_med_names = torch.sparse_coo_tensor(torch.transpose(index_med_names, 0, 1),
                                                  value_med_name + add_name, dtype=torch.int32).to_dense()


    config['v_units'] = vocab_len_units
    config['v_med_ids'] = vocab_len_med_ids
    config['e_dim_med_ids'] = args.lstmMedEmbDim
    config['e_dim_units'] = args.lstmUnitExpand
    config['preops_init_med'] = args.preopInitLstmMed
    config['lstm_hid'] = args.lstmMedWidth
    config['lstm_num_layers']= args.lstmMedDepth
    config['bilstm_med'] = args.BilstmMed
    config['Att_MedAgg'] = args.AttentionMedAgg
    config['AttMedAgg_Heads']= args.AttMedAggHeads
    config['p_idx_med_ids']=0 # putting these 0 because the to dense sets everything not available as 0
    config['p_idx_units']= 0
    config['p_time']=args.lstmMedDrop
    config['p_rows'] = args.lstmRowDrop
    config['weight_decay_LSTMmedL2'] = args.lstmMedL2
    config['group_start_list'] = group_start
    config['group_end_list'] = group_end

# outcome_df.drop(["orlogid_encoded"], axis=1, inplace=True)
# outcome_df.reset_index(inplace=True)
# outcome_df.rename({"index": "person_integer"}, axis=1, inplace=True)

print("Passed all the data processing stage")

config['modality_used'] = modality_to_use
device = torch.device('cuda')


# this is to add to the dir_name
modalities_to_add = '_modal'

for i in range(len(modality_to_use)):
    modalities_to_add = modalities_to_add + "_" + modality_to_use[i]

best_5_random_number = []  # this will take the args when directly run otherwise it will read the number from the file namee
if eval(args.bestModel) == True:
    # path_to_dir = '/home/trips/PeriOperative_RiskPrediction/HP_output/'
    # sav_dir = '/home/trips/PeriOperative_RiskPrediction/Best_results/Intraoperative/'

    # this is to be used when running the best setting results on RIS
    path_to_dir = out_dir + 'HP_output/'
    sav_dir = out_dir + 'Best_results/Intraoperative/'

    # Best_trial_resulticu_transformer_modal__preops_cbow_pmh_problist_homemeds_flow_meds_424_24-07-16-16:13:30.json
    file_names = os.listdir(path_to_dir)
    best_5_names = []

    best_5_initial_name = 'Best_trial_result' +args.task+"_" + str(args.modelType)+ "_modal_"

    modal_name = 'DataModal'
    for i in range(len(modality_to_use)):
        if (modality_to_use[i] != 'cbow'):
            modal_name = modal_name + "_" + modality_to_use[i]
        best_5_initial_name = best_5_initial_name + "_" + modality_to_use[i]


    dir_name = sav_dir + args.modelType + '/' + modal_name + "_" + str(args.task) + "/"

    for file_name in file_names:
        if (best_5_initial_name in file_name) and (file_name.split("_")[-3] in modality_to_use):
            print(file_name)
            best_5_names.append(file_name)
            best_5_random_number.append(int(file_name.split("_")[-2]))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print(f"The directory '{dir_name}' already exists.")

    best_metadata_dict = {}
else:
    best_5_random_number.append(args.randomSeed)
    config['p_final'] =args.finalDrop
    config['finalBN'] =args.finalBN
    # this is being added apriori because we are projecting the final representation to this dimension
    config['hidden_units_final'] =args.preopsWidthFinal
    config['weightInt'] =args.XavOrthWeightInt

if binary_outcome:
    perf_metric = np.zeros((len(best_5_random_number), 2)) # 2 is for the metrics auroc and auprc
    if args.overSampling == True:
        os_flag =True
else:
    perf_metric = np.zeros((len(best_5_random_number), 5)) # 5 is for the metrics corr, corr_p, R2, MAE, MSE
    os_flag =False

# not a priority right now; this would basically allow running multiple seeds at different time points by checking if already exists
# if eval(args.bestModel) ==True:
#     temp_filename = dir_name + '/Best_runs_metadata.pickle'
#     if os.path.exists(temp_filename):
#         with open(temp_filename, 'rb') as file: md_e = pickle.load(file)
#         seed_keys = md_e.keys()
#         done_seeds = [name.split("_")[-1] for name in seed_keys]
#         breakpoint()

for runNum in range(len(best_5_random_number)):
    # starting time of the run
    start_time = datetime.now()

    torch.manual_seed(int(best_5_random_number[runNum]))
    torch.cuda.manual_seed(int(best_5_random_number[runNum]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(int(best_5_random_number[runNum]))

    if eval(args.bestModel) ==True:
        best_file_name = path_to_dir + best_5_names[runNum]
        md_f = open(best_file_name)
        md = json.load(md_f)
        param_values = md['params']

        config['p_final'] = param_values['p_final']
        config['finalBN'] = param_values['finalBN']
        # this is being added apriori because we are projecting the final representation to this dimension
        config['hidden_units_final'] = param_values['hidden_units_final']
        config['weightInt'] = param_values['weightInt']
        best_dict_local = {}

    if 'preops' not in modality_to_use:
        test_size = 0.2
        valid_size = 0.05  # TODO: change back to 0.00005 for the full dataset
        y_outcome = outcome_df["outcome"].values
        preops.reset_index(drop=True, inplace=True)
        upto_test_idx = int(test_size * len(preops))
        test = preops.iloc[:upto_test_idx]
        train0 = preops.iloc[upto_test_idx:]
        if (binary_outcome == True) and (y_outcome.dtype != 'float64'):
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=int(best_5_random_number[runNum]),
                                            stratify=y_outcome[train0.index])
        else:
            train, valid = train_test_split(train0, test_size=valid_size / (1. - test_size),
                                            random_state=int(best_5_random_number[runNum]))

        train_index = train.index
        valid_index = valid.index
        test_index = test.index

        # if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set
        #     test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 'ICU'][
        #         'plannedDispo'].index

    if 'preops' in modality_to_use:
        # currently sacrificing 5 data points in the valid set and using the test set to finally compute the auroc etc
        preops_tr, preops_val, preops_te, train_index, valid_index, test_index, preops_mask = pps.preprocess_train(
            preops,
            args.task,
            y_outcome=
            outcome_df[
                "outcome"].values,
            binary_outcome=binary_outcome,
            valid_size=0.05, random_state=int(best_5_random_number[runNum]), input_dr=data_dir,
            output_dr=out_dir)  # TODO: change back to 0.00005

        # if args.task == 'icu':  # this part is basically dropping the planned icu cases from the evaluation set (value of plannedDispo are numeric after processing; the df has also been changed )
        #     test_index = preops.iloc[test_index][preops.iloc[test_index]['plannedDispo'] != 3]['plannedDispo'].index
        #     preops_te = preops_te.iloc[test_index]

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

        config['input_shape'] = num_preop_features + preop_mask_counter * num_preop_features,  # this is done so that I dont have to write a seperate condition for endofcase where the current time is being appended to preops
        config['input_shape_bow'] = len(bow_input.columns)


        if eval(args.bestModel) == True:
            config['hidden_units'] = param_values['hidden_units']
            config['hidden_depth'] = param_values['hidden_depth']
            config['weight_decay_preopsL2'] = param_values['weight_decay_preopsL2']
            config['weight_decay_preopsL1'] = param_values['weight_decay_preopsL1']

            config['hidden_units_bow'] = param_values['hidden_units_bow']
            config['hidden_units_final_bow'] = param_values['hidden_units_final_bow']
            config['hidden_depth_bow'] = param_values['hidden_depth_bow']
            config['weight_decay_bowL2'] = param_values['weight_decay_bowL2']
            config['weight_decay_bowL1'] = param_values['weight_decay_bowL1']

    if 'homemeds' in modality_to_use:
        if args.home_medsform == 'embedding_attention':
            hm_tr = torch.index_select(dense_HM_embedding, 0, torch.tensor(train_index)).coalesce()
            hm_val = torch.index_select(dense_HM_embedding, 0, torch.tensor(valid_index)).coalesce()
            hm_te = torch.index_select(dense_HM_embedding, 0, torch.tensor(test_index)).coalesce()
        else:
            hm_tr = torch.tensor(home_meds_final.iloc[train_index].to_numpy(), dtype=torch.float32)
            hm_te = torch.tensor(home_meds_final.iloc[test_index].to_numpy(), dtype=torch.float32)
            hm_val = torch.tensor(home_meds_final.iloc[valid_index].to_numpy(), dtype=torch.float32)
        if eval(args.bestModel) == True:
            config['hidden_units_hm'] = param_values['hidden_units_hm']
            config['hidden_units_final_hm'] = param_values['hidden_units_final_hm']
            config['hidden_depth_hm'] = param_values['hidden_depth_hm']
            config['weight_decay_hmL2'] = param_values['weight_decay_hmL2']
            config['weight_decay_hmL1'] = param_values['weight_decay_hmL1']
            if 'Att_HM_Agg' in param_values.keys():
                config['Att_HM_Agg'] = param_values['Att_HM_Agg']
                config['Att_HM_agg_Heads'] = param_values['Att_HM_agg_Heads']
    if 'pmh' in modality_to_use:

        if args.pmhform == 'embedding_sum':
            pmh_tr = torch.tensor(pmh_emb_sb_final.iloc[train_index].to_numpy(), dtype=torch.float32)
            pmh_te = torch.tensor(pmh_emb_sb_final.iloc[test_index].to_numpy(), dtype=torch.float32)
            pmh_val = torch.tensor(pmh_emb_sb_final.iloc[valid_index].to_numpy(), dtype=torch.float32)


        if args.pmhform == 'embedding_attention':
            pmh_tr = torch.index_select(dense_pmh_embedding, 0, torch.tensor(train_index)).coalesce()
            pmh_val = torch.index_select(dense_pmh_embedding, 0, torch.tensor(valid_index)).coalesce()
            pmh_te = torch.index_select(dense_pmh_embedding, 0, torch.tensor(test_index)).coalesce()


        if eval(args.bestModel) == True:
            config['hidden_units_pmh'] = param_values['hidden_units_pmh']
            config['hidden_units_final_pmh'] = param_values['hidden_units_final_pmh']
            config['hidden_depth_pmh'] = param_values['hidden_depth_pmh']
            config['weight_decay_pmhL2'] = param_values['weight_decay_pmhL2']
            config['weight_decay_pmhL1'] = param_values['weight_decay_pmhL1']
            if 'Att_pmh_Agg' in param_values.keys():
                config['Att_pmh_Agg'] = param_values['Att_pmh_Agg']
                config['AttPmhAgg_Heads'] = param_values['AttPmhAgg_Heads']

    # if 'problist' in modality_to_use:
    #
    #     if args.problistform == 'embedding_sum':
    #         problist_tr = torch.tensor(prob_list_emb_sb_final.iloc[train_index].to_numpy(), dtype=torch.float32)
    #         problist_te = torch.tensor(prob_list_emb_sb_final.iloc[test_index].to_numpy(), dtype=torch.float32)
    #         problist_val = torch.tensor(prob_list_emb_sb_final.iloc[valid_index].to_numpy(), dtype=torch.float32)
    #
    #
    #     if args.problistform == 'embedding_attention':
    #         problist_tr = torch.index_select(dense_problist_embedding, 0, torch.tensor(train_index)).coalesce()
    #         problist_val = torch.index_select(dense_problist_embedding, 0, torch.tensor(valid_index)).coalesce()
    #         problist_te = torch.index_select(dense_problist_embedding, 0, torch.tensor(test_index)).coalesce()
    #
    #
    #     if eval(args.bestModel) == True:
    #         config['hidden_units_problist'] = param_values['hidden_units_problist']
    #         config['hidden_units_final_problist'] = param_values['hidden_units_final_problist']
    #         config['hidden_depth_problist'] = param_values['hidden_depth_problist']
    #         config['weight_decay_problistL2'] = param_values['weight_decay_problistL2']
    #         config['weight_decay_problistL1'] = param_values['weight_decay_problistL1']
    #         if 'Att_problist_Agg' in param_values.keys():
    #             config['Att_problist_Agg'] = param_values['Att_problist_Agg']

    if 'flow' in modality_to_use:
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

        if eval(args.bestModel) == True:
            config['preops_init_flow'] = param_values['preops_init_flow']
            config['lstm_flow_hid'] = param_values['lstm_flow_hid']
            config['lstm_flow_num_layers'] = param_values['lstm_flow_num_layers']
            config['bilstm_flow'] = param_values['bilstm_flow']
            config['p_flow'] = param_values['p_flow']
            config['weight_decay_LSTMflowL2'] = param_values['weight_decay_LSTMflowL2']
        config['num_flowsheet_feat'] = total_flowsheet_measures

    if 'meds' in modality_to_use:
        if eval(args.bestModel) == True:
            config['e_dim_med_ids'] = param_values['e_dim_med_ids']
            config['preops_init_med'] = param_values['preops_init_med']
            config['lstm_hid'] = param_values['lstm_hid']
            config['lstm_num_layers'] = param_values['lstm_num_layers']
            config['bilstm_med'] = param_values['bilstm_med']
            if 'Att_problist_Agg' in param_values.keys():
                config['Att_MedAgg'] = param_values['Att_MedAgg']
                config['AttMedAgg_Heads'] = param_values['AttMedAgg_Heads']
            config['p_time'] = param_values['p_time']
            config['p_rows'] = param_values['p_rows']
            config['weight_decay_LSTMmedL2'] = param_values['weight_decay_LSTMmedL2']

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
            ## todo shuffle the homemeds here
            if True:
                data_tr['homemeds'] = hm_tr
                data_val['homemeds'] = hm_val
                data_te['homemeds'] = hm_te
            else:
                idx = torch.randperm(hm_tr.nelement())  #shuffling the indices here
                hm_tr = hm_tr.reshape(-1)[idx].view(hm_tr.size())
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
    if args.modelType == 'transformer':
        if eval(args.bestModel) == True:
            config['e_dim_flow'] = param_values['e_dim_flow']
            config['AttTS_depth'] = param_values['AttTS_depth']
            temp_list_heads = [i for i in param_values.keys() if 'AttTS_Heads' in i]
            if len(temp_list_heads) > 1:
                config['AttTS_Heads'] = param_values[temp_list_heads[-1]] # this assumes that the head names are ordered which has been the case
            else:
                config['AttTS_Heads'] = param_values['AttTS_Heads']
            config['cnn_before_Att'] = param_values['cnn_before_Att']
            config['kernel_size_conv'] = param_values['kernel_size_conv']
            config['stride_conv'] = param_values['stride_conv']
            config['ats_dropout'] = param_values['ats_dropout']
        else:
            config['e_dim_flow'] = args.e_dim_flow
            config['AttTS_depth'] = args.AttTSDepth
            config['AttTS_Heads'] = args.AttTSHeads
            config['cnn_before_Att'] = args.cnn_before_Att
            config['kernel_size_conv'] = args.kernel_size_conv
            config['stride_conv'] = args.stride_conv
            config['ats_dropout'] = args.ats_dropout

        model = preop_flow_med_bow_model.TS_Transformer_Med_index(**config).to(device)
    else:
        model = preop_flow_med_bow_model.TS_lstm_Med_index(**config).to(device)

    if eval(args.bestModel) == True:
        learn_rate = param_values['learningRate']
        lr_patience = param_values['LRPatience']
        lr_factor = param_values['learningRateFactor']
        batchsize = param_values['batchSize']
    else:
        learn_rate = args.learningRate
        lr_patience = args.LRPatience
        lr_factor = args.learningRateFactor
        batchsize = args.batchSize

    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, verbose=True, factor=lr_factor)

    # initializing the loss function
    if not binary_outcome:
        criterion = torch.nn.MSELoss()
        # criterion = torch.nn.L1Loss()
    else:
      criterion = torch.nn.BCELoss()

    total_train_loss = []
    total_test_loss = []


    updating_lr = learn_rate
    best_metric = 1000 # some large number
    lr_schedular_epoch_dict = {}
    lr_schedular_epoch_dict[0] = updating_lr
    for epoch in range(args.epochs):
      loss_tr = 0
      loss_tr_cls = 0
      model.train()
      ## the default __getitem__ is like 2 orders of magnitude slower
      shuffle_index = torch.randperm(n=data_tr['outcomes'].shape[0])
      if (os_flag == True) and (args.task != 'endofcase'):
          pos_idx = (data_tr['outcomes'] == 1).nonzero()
          neg_idx = (data_tr['outcomes'] == 0).nonzero()
          if batchsize % 2 == 0:  # this is done because it was creating a problem when the batchsize was an odd number
              nbatch = neg_idx.shape[0] // int(batchsize / 2)
          else:
              nbatch = neg_idx.shape[0] // math.ceil(batchsize / 2)
      else:
          nbatch = data_tr['outcomes'].shape[0] // batchsize
      for i in range(nbatch):
          if (os_flag == True) and (args.task != 'endofcase'):
              if batchsize % 2 == 0:
                  neg_indexbatch = neg_idx[range(i * int(batchsize / 2), (i + 1) * int(batchsize / 2))]
              else:
                  neg_indexbatch = neg_idx[range(i * math.ceil(batchsize / 2), (i + 1) * math.ceil(batchsize / 2))]
              p = torch.from_numpy(np.repeat([1 / len(pos_idx)], len(pos_idx)))
              pos_indexbatch = pos_idx[p.multinomial(num_samples=int(batchsize / 2),
                                                     replacement=True)]  # this is sort of an equivalent of numpy.random.choice
              these_index = torch.vstack([neg_indexbatch, pos_indexbatch]).reshape([batchsize])
          else:
              these_index = shuffle_index[range(i * batchsize, (i + 1) * batchsize)]

          ## this collate method is pretty inefficent for this task but works with the generic DataLoader method


          local_data={}
          for k in data_tr.keys():
              if type(data_tr[k]) != list:
                  local_data[k] = torch.index_select(data_tr[k],  0 ,  these_index )
              else:
                  local_data[k] = [torch.index_select(x,  0 ,  these_index )  for x in data_tr[k]]

          if args.task == 'endofcase':
              local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                  len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
              local_data[0][:, -1] = local_data[
                  1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
              local_data[-1] = torch.from_numpy(np.repeat([1, 0], [int(batchsize / 2), batchsize - int(batchsize / 2)]))
          if (args.includeMissingnessMasks):  # appending the missingness masks in training data
              if 'preops' in modality_to_use:
                  local_data['preops'] =[local_data['preops'], torch.tensor(preops_tr_mask.iloc[these_index].to_numpy(), dtype=torch.float32)]
              if 'flow' in modality_to_use:
                  local_data['flow'].append(very_dense_tr_mask[these_index, :, :])
                  sparse_mask = torch.sparse_coo_tensor(local_data['flow'][1]._indices(), np.ones(len(local_data['flow'][1]._values())), local_data['flow'][1].size())
                  local_data['flow'].append(sparse_mask)
          data_train, mod_order_dict = preop_flow_med_bow_model.collate_time_series(local_data, device)
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
      loss_tr_cls = loss_tr_cls/ data_tr['outcomes'].shape[0]

      loss_te = 0
      loss_te_cls = 0
      with torch.no_grad():
        model.eval()
        true_y_test = []
        pred_y_test = []

        # nbatch = data_te['outcomes'].shape[0] // batchsize
        nbatch, remain_batch = divmod(data_te['outcomes'].shape[0], batchsize)
        if remain_batch > 0:
            nbatch=nbatch+1  # this is being done to make sure all the test data is being used when the test set size is not a multiple of batchsize
        for i in range(nbatch):

            if (remain_batch>0) and (i==nbatch-1):
                these_index = torch.tensor(list(range(i * batchsize, (i * batchsize)+remain_batch)), dtype=int)
            else:
                these_index = torch.tensor(list(range(i * batchsize, (i + 1) * batchsize)), dtype=int)
            local_data = {}
            for k in data_te.keys():
                if type(data_te[k]) != list:
                    local_data[k] = torch.index_select(data_te[k], 0, these_index)
                else:
                    local_data[k] = [torch.index_select(x, 0, these_index) for x in data_te[k]]

            if args.task == 'endofcase':
                local_data[1] = torch.hstack([local_data[1][:int(len(these_index) / 2)], local_data[-1][int(
                    len(these_index) / 2):]])  # using hstack because vstack leads to two seperate tensors
                local_data[0][:, -1] = local_data[
                    1]  # this is being done because the last column has the current times which will be t1 timepoint for the second half of the batch
                local_data[-1] = torch.from_numpy(np.repeat([1, 0], [int(batchsize / 2), batchsize - int(batchsize / 2)]))
            if (args.includeMissingnessMasks):  # appending the missingness masks in test data
                if 'preops' in modality_to_use:
                    local_data['preops'] = [local_data['preops'],
                                            torch.tensor(preops_te_mask.iloc[these_index].to_numpy(), dtype=torch.float32)]
                if 'flow' in modality_to_use:
                    local_data['flow'].append(very_dense_te_mask[these_index, :, :])
                    sparse_mask = torch.sparse_coo_tensor(local_data['flow'][1]._indices(),
                                                          np.ones(len(local_data['flow'][1]._values())),
                                                          local_data['flow'][1].size())
                    local_data['flow'].append(sparse_mask)

            data_valid, mod_order_dict = preop_flow_med_bow_model.collate_time_series(local_data, device)

            y_pred, reg_loss = model(data_valid[0])
            cls_loss_te = criterion(y_pred.squeeze(-1),data_valid[1].float().to(device)).float()
            test_loss = cls_loss_te + reg_loss
            loss_te += test_loss.item()
            loss_te_cls += cls_loss_te.item()

            # values from the last epoch; it will get overwritten
            # using test data only instead of validation data for evaluation currently because the validation will be done on a seperate data
            true_y_test.append(data_valid[1].float().detach().numpy())
            pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())


        loss_te = loss_te / data_te['outcomes'].shape[0]
        loss_te_cls = loss_te_cls/ data_te['outcomes'].shape[0]

        if best_metric > loss_te_cls:
            best_metric = loss_te_cls
            pred_y_test_best = pred_y_test

        # display the epoch training and test loss
        print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, args.epochs, loss_tr_cls,loss_te_cls) )
        total_train_loss.append(loss_tr)
        total_test_loss.append(loss_te)

      scheduler.step(loss_te_cls)

      if optimizer.param_groups[0]['lr'] != updating_lr:
          updating_lr = optimizer.param_groups[0]['lr']
          lr_schedular_epoch_dict[epoch] = updating_lr

      # print("current lr ", optimizer.param_groups[0]['lr'])
      # 1e-8 is obtained by multiplying 1e-3 by (0.25)^5 so to make it general we can have initial_learning_rate * (learningRateFactor)^5
      if optimizer.param_groups[0]['lr'] <= learn_rate * np.power(lr_factor, 5):  # hardcoding for now because our schedule is such that 10**-3 * (1, 1/4, 1/16, 1/64, 1/256, 1/1024, 0) with an initial rate of 10**-3 an learning rate factor of 0.25
          print("inside the early stopping loop")
          print("best validation loss ", best_metric)
          # epoch =epochs
          # true_y_test.append(data_valid[1].float().detach().numpy())
          # pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())
          break

    model.eval()

    true_y_test = np.concatenate(true_y_test)
    pred_y_test = np.concatenate(pred_y_test_best)

    if binary_outcome:
        test_auroc = roc_auc_score(true_y_test, pred_y_test)
        test_auprc = average_precision_score(true_y_test, pred_y_test)

        perf_metric[runNum, 0] =test_auroc
        perf_metric[runNum, 1] = test_auprc
    else:
        corr_value = np.round(pearsonr(np.array(true_y_test), np.array(pred_y_test))[0], 3)
        cor_p_value = np.round(pearsonr(np.array(true_y_test), np.array(pred_y_test))[1], 3)
        print(str(args.task) + " prediction with correlation ", corr_value, ' and corr p value of ', cor_p_value)
        r2value = r2_score(np.array(true_y_test), np.array(pred_y_test))  # inbuilt function also exists for R2
        print(" Value of R2 ", r2value)
        temp_df = pd.DataFrame(columns=['true_value', 'pred_value'])
        temp_df['true_value'] = np.array(true_y_test)
        temp_df['pred_value'] = np.array(pred_y_test)
        temp_df['abs_diff'] = abs(temp_df['true_value'] - temp_df['pred_value'])
        temp_df['sqr_diff'] = (temp_df['true_value'] - temp_df['pred_value']) * (
                temp_df['true_value'] - temp_df['pred_value'])
        mae_full = np.round(temp_df['abs_diff'].mean(), 3)
        mse_full = np.round(temp_df['sqr_diff'].mean(), 3)
        print("MAE on the test set ", mae_full)
        print("MSE on the test set ", mse_full)

        perf_metric[runNum, 0] = corr_value
        perf_metric[runNum, 1] = cor_p_value
        perf_metric[runNum, 2] = r2value
        perf_metric[runNum, 3] = mae_full
        perf_metric[runNum, 4] = mse_full

    if eval(args.bestModel) == True:
        metadata_file = dir_name + 'BestModel_metadata' + str(best_5_random_number[runNum]) + '_' + args.task + '.pickle'  # had to use pickle instead of json because there is a tensor in config which is assigned to dict outside of loop so can't convert it to list
        with open(metadata_file, 'wb') as outfile: pickle.dump(config, outfile)
        model_cachename = hashlib.md5(json.dumps(vars(args)).encode())
        saving_path_name = dir_name + 'BestModel_' + str(int(best_5_random_number[runNum])) + "_" + modal_name + "_" + str(model_cachename.hexdigest()) + ".pth"
        torch.save(model.state_dict(), saving_path_name)

        best_dict_local['randomSeed'] = int(best_5_random_number[runNum])
        best_dict_local['outcome'] = str(args.task)
        best_dict_local['run_number'] = runNum
        best_dict_local['modalities_used'] = modality_to_use
        best_dict_local['model_params'] = config
        best_dict_local['model_file_path'] = saving_path_name
        best_dict_local['train_orlogids'] = outcome_df.iloc[train_index]["orlogid_encoded"].values.tolist()
        best_dict_local['test_orlogids'] =outcome_df.iloc[test_index]["orlogid_encoded"].values.tolist()
        if binary_outcome:
            best_dict_local['outcome_rate'] = np.round(outcome_df.iloc[test_index]["outcome"].mean(), decimals=4)
        # this is saving the true and predicted y for each run because the test set is the same
        if runNum == 0:
            outcome_with_pred_test = outcome_df.iloc[test_index]
            outcome_with_pred_test = outcome_with_pred_test.rename(columns={'outcome': 'y_true'})
            outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = pred_y_test
        else:
            outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = pred_y_test
        dict_key = 'run_randomSeed_' + str(int(best_5_random_number[runNum]))  # this is so dumb because it wont take the key dynamically
        best_metadata_dict[dict_key] = best_dict_local

    if binary_outcome:
        print(perf_metric)
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(true_y_test, pred_y_test, drop_intermediate=False)
        precision_prc, recall_prc, thresholds_prc = precision_recall_curve(true_y_test, pred_y_test)
        # interpolation in ROC
        mean_fpr = np.linspace(0, 1, 100)
        tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
        mean_fpr = np.round(mean_fpr, decimals=2)
        print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))
    end_time = datetime.now()  # only writing part is remaining in the code to time
    timetaken = end_time-start_time
    print("time taken to finish run number ", runNum, " is ", timetaken)

breakpoint()
print("Tranquila")
# saving metadata for all best runs in json; decided to save it also as pickle because the nested datatypes were not letting it be serializable
metadata_filename = dir_name + '/Best_runs_metadata.pickle'
with open(metadata_filename, 'wb') as outfile: pickle.dump(best_metadata_dict, outfile)

# saving the performance metrics from all best runs and all models in a pickle file
perf_filename = sav_dir + str(args.task) + '_Best_perf_metrics_combined_intraoperative.pickle'
if not os.path.exists(perf_filename):
    data = {}
    data[str(args.modelType)] = {modal_name: perf_metric}
    with open(perf_filename, 'wb') as file:
        pickle.dump(data, file)
else:
    with open(perf_filename, 'rb') as file: existing_data = pickle.load(file)

    try:
        existing_data[str(args.modelType)][modal_name] = perf_metric
    except(KeyError):  # this is to take care of the situation when a new model is added to the file
        existing_data[str(args.modelType)] = {}
        existing_data[str(args.modelType)][modal_name] = perf_metric

    # Save the updated dictionary back to the pickle file
    with open(perf_filename, 'wb') as file: pickle.dump(existing_data, file)

# saving the test set predictions for all models and all runs
pred_filename = sav_dir + str(args.task) + '_Best_pred_combined_intraoperative.pickle'
if not os.path.exists(pred_filename):
    data = {}
    data[str(args.modelType)] = {modal_name: outcome_with_pred_test.values}
    with open(pred_filename, 'wb') as file:
        pickle.dump(data, file)
else:
    with open(pred_filename, 'rb') as file: existing_data = pickle.load(file)

    try:
        existing_data[str(args.modelType)][modal_name] = outcome_with_pred_test.values
    except(KeyError):  # this is to take care of the situation when a new model is added to the file
        existing_data[str(args.modelType)] = {}
        existing_data[str(args.modelType)][modal_name] = outcome_with_pred_test.values

    # Save the updated dictionary back to the pickle file
    with open(pred_filename, 'wb') as file: pickle.dump(existing_data, file)

