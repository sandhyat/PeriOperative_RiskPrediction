This repository contains the source code for various approaches that can used for postoperative complication prediction.

## Overview

In brief, there are two approaches: 
1) prediction classifiers are learnt end to end in a supervised manner using different combination of data modalities, 
2) first a self supervised representation of a data modality such as time series is learnt followed by a supervised classifier on the learnt representation.

In clinical prediction setup, access to the data modalities can vary and need not be available for all the patients. 
Some of the common data modalities observed in Electronic Health Records are as follows: preoperative assessment, free text about surgical procedure, past medical history, home medications and problem list, intraoperative flowsheets, intraoperative medications, ICD codes (postoperatively).
Our task of interest (outcome) is predicting post-operative complications such as 30-day inhospital mortality, unplanned ICU admission, Acute Kidney Injury. These labels are retrospectively created by experts or extracted from the hospital billing data.

Based on varying algorithm complexity the approaches mentioned can be explained below:

1) A prediction classifier that uses the data that is available prior to the start of surgery. As this data exists in tabular form or can be converted in tabular (text embeddings), training a shallow classifier using XGBoost or Logistic Regression is the first and most straightforward approach. Recently, more complex and high capacity learners such as transformers are also available for tabular data. 

2) A prediction classifier that uses all the data that is available until the end of surgery. This method needs special techniques to learn from the intraoperative time series data. One can use LSTMs, attention mechanism or Transformer architecture along with feed forward networks for the tabular data and train them jointly.
![End to End Architecture](/Images/End-toEnd_Supervised.png)

3) A prediction classifier that uses all the data that is available retrospectively (including the outcomes) for self supervised learning (commonly known as pretraining) and using everything before the end of surgery for training the classifiers. One can use many of the techniques currently available for pretraining including contrastive learning for the first stage of representation learning. For the second stage of classifier learning, even training a shallow classifier suffices sometimes.
![MVCL Architecture](/Images/MVCL_SelfSupervised.png)

## Setups


#### 1) Data and Architecture adaptation format  

**For end-to-end supervised setup binary classification**
Code is available in `End_to_end_supervised/`

1) `Preops_processing.py`: Prepares the preoperative data including one hot encoding, missingness imputation, normalization and saves the metadata into json. Also, creates the train test partitioning on the dataset. 
2) `Preops_GBT_baseline.py`: Trains a shallow or basic feed forward model on data available before the surgery starts.
3) `Training_with_TimeSeries.py`: Jointly trains a deep model (LSTM/Transformer) using all the data modalities available at the end of surgery.
4) `preop_flow_med_bow_model.py`: Contains various deep learning architectures in the model class format that are callable in the `Training_with_TimeSeries.py` file.
5) `Training_with_ts_Optuna_hp_tuning.py`: Used for hyperparameter tuning using Optuna as an off the shelf method. Similar functionalities are `Training_with_TimeSeries.py` but in the context of hyper parameter tuning. Currently, the best trial is not saved and the inbuilt storage of optuna is not used (for future).
6) `ts_model_class_Optuna_hp_tune.py`: Mostly, same as `preop_flow_med_bow_model.py` but used during hp tuning. Could be removed in future.
7) `ts_optuna_hp_tuning_combined.sh`: Bash file that runs the hp tuning file `Training_with_ts_Optuna_hp_tuning.py` for different modality ablations and with different random seeds.
8) `Tabnet_tabular.py`: [Tabnet](https://arxiv.org/pdf/1908.07442) [implementation](https://github.com/dreamquark-ai/tabnet) on the preoperative stage modalities trained in a supervised manner. It has a `bestModel` argument which when selected will read the HP tuned result files and retrain the models on the train+validation set of dataset. It also saves the models and performance metrics and prediction values on the common test set across all ablations and all 5 repetitions.
9) `Tabnet_tabular_HP_tuning.py`: Used for Tabnet HP tuning using Optuna. 
10) `tabnet_optuna_hp_tuning.sh`: Bash file that runs Tabnet HP tuning file for different preoperative stage modalities with different random seeds.

**For two stage self-supervised setup binary classification**
Currently, for time series code for [TS2Vec](https://github.com/yuezhihan/ts2vec) is being used, for tabular preops and cbow [SCARF](https://github.com/clabrugere/pytorch-scarf/tree/master) method's code is being used, and for the outcomes there is only a projection head.

Code is available in `Two_stage_selfsupervised/`

1) `train_modular.py`: Calls `Multiview_CL_modular.py` for representation learning using all the modalities that are specified in the arguments. Once trained, calls `classification.py` to perform the downstream task or evaluate the learnt representation.
2) `Scarf_tabular.py`: [SCARF](https://arxiv.org/pdf/2106.15147) [implementation](https://github.com/clabrugere/pytorch-scarf) on the preoperative stage modalities to learn self supervised representation which is fed to an XGBT learner.
3) `Scarf_tabular_HP_tuning.py`: Used for Scarf HP tuning using Optuna.
4) `scarf_optuna_hp_tuning.sh`: Bash file that runs Scarf HP tuning file for different preoperative stage modalities with different random seeds. This one exposes the cuda device 1. Will need to be changed later to make it generalizable.

#### 2) Requirements and implementation

One can use the `requirements.txt` to install the dependencies. One can also run the codes inside a *docker container* using the docker121720/pytorch-for-ts:0.95 image. Optuna will be needed if you are running the HP tuning file `Training_with_ts_Optuna_hp_tuning.py` directly.

**Example for end-to-end supervised setup binary classification**
```
docker run --rm --gpus all --privileged -v '< /PATH TO THE INPUT DATA/ >:/input/' -v '< /PATH TO THE SCRIPTS/ >:/codes/' -v '< /PATH TO THE SAVING THE OUTPUT RESULTS/ >:/output/' docker121720/pytorch-for-ts:0.95 python /codes/Training_with_TimeSeries.py --nameinfo="testing_Full_fixed_seed_withmaskOversamplingEarlyStoppingLR" --outputcsv="test_binary_outcomes.csv" --task='icu' --preopsDepth=6 --preopsWidth=20 --preopsWidthFinal=16 --bowDepth=5 --bowWidth=90 --bowWidthFinal=20 --lstmMedEmbDim=16 --lstmMedDepth=4 --lstmMedWidth=40 --lstmFlowDepth=4 --lstmFlowWidth=40 --LRPatience=3 --batchSize=120 --lstmMedDrop=0.1212 --lstmFlowDrop=0.0165 --finalDrop=0.3001 --learningRate=0.0002 --learningRateFactor=0.2482 --preopsL2=0.0004 --preopsL1=0.0029 --bowL2=0.0003 --bowL1=0.0042 --lstmMedL2=0.0003 --lstmFlowL2=0.0009 --randomSeed=350 --includeMissingnessMasks --overSampling
```

**Example for two stage self-supervised setup binary classification**
```
docker run --rm --gpus all --privileged -v '< /PATH TO THE INPUT DATA/ >:/input/' -v '< /PATH TO THE SCRIPTS/ >:/codes/' -v '< /PATH TO THE SAVING THE OUTPUT RESULTS/ >:/output/' docker121720/pytorch-for-ts:0.95 python /codes/train_modular.py Flowsheets F_output --eval --outcome=icu
```
Based on the other available modalities, following can be added or removed ``` --preops --meds --alerts --pmh --problist --homemeds --postopcomp  ``` to the argument list. 

Further, if one is interested in using only the preops to train the model, i.e., no contrastive learning, one can add ``` --withoutCL ``` to the arguments.

By default, for the downstream classification task (when ``` --eval ``` is True), all modalitites are used in their original form except time series for which the learnt representation is used. 
If you are interested in using the learnt representation for preop modalities too, add ``` --all_rep ``` to the argument list. 


#### 3) Hyperparameter tuning

`Generating_HP_grid.py` is used to create a sobol grid and writes them in a format that can be submitted on a high performance computing server. It creates a file that can be launched using bash directly.