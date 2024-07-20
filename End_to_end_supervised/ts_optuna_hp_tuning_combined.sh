# this file is going to take the input of random numbers and the ablation details
# this file is expected to run 30 trials (5 random initialization times 6 ablation cases)f or one outcome and one model


# first positional argument will be the task second positional argument will be the model with options {'lstm', 'transformer'}

#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "No task provided."
    exit 1
fi

pip install optuna
#pip uninstall pandas
pip install pandas==1.5.3

numbers=($(shuf -i 100-500 -n 5))

for number in "${numbers[@]}"
do
  echo $number
  echo $1
  echo $2
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task=$1 --modelType=$2 --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task=$1 --modelType=$2 --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task=$1 --modelType=$2 --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task=$1 --modelType=$2 --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task=$1 --modelType=$2 --randomSeed=$number
  python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task=$1 --modelType=$2 --randomSeed=$number
done