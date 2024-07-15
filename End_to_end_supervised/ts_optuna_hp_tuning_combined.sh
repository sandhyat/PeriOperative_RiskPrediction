# this file is going to take the input of random numbers and the ablation details
# this file is expected to run 30 trials (5 random initialization times 6 ablation cases)f or one outcome and one model

# to start the docker container inside which this file will run
#docker run -it --gpus all --privileged -v '/home/trips/PeriOperative_RiskPrediction/:/codes/' -v '/mnt/ris/ActFastExports/v1.3.2/:/input/' -v '/home/trips/PeriOperative_RiskPrediction/HP_output/:/output/' docker121720/pytorch-for-ts:0.95 /bin/bash


#!/usr/bin/env bash

pip install optuna
#pip uninstall pandas
pip install pandas==1.5.3

numbers=($(shuf -i 100-500 -n 5))

for number in "${numbers[@]}"
do
  echo $number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task='icu' --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task='icu' --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='icu' --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='icu' --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='icu' --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='icu' --randomSeed=$number
done