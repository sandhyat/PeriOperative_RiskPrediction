# this file is going to take the input of random numbers and the ablation details
# this file is expected to run 30 trials (5 random initialization times 6 ablation cases)f or one outcome and one model

# to start the docker container inside which this file will run
#docker run -it --gpus all --privileged -v '/home/trips/PeriOperative_RiskPrediction/:/codes/' -v '/mnt/ris/ActFastExports/v1.3.2/:/input/' -v '/home/trips/PeriOperative_RiskPrediction/HP_output/:/output/' docker121720/pytorch-for-ts:0.95 /bin/bash

# to run inside the docker container on RIS
#LSF_DOCKER_VOLUMES='/storage1/fs1/christopherking/Active/ActFastExports/v1.3.2/:/input/ /storage1/fs1/christopherking/Active/sandhyat/Output-TS_docker_July2024/:/output/ /home/sandhyat/PeriOperative_RiskPrediction/:/codes/' bsub -G 'compute-christopherking' -g '/sandhyat/largeNjob15hpsearchgroup' -n 8 -q general -R 'gpuhost' -gpu 'num=1:gmodel=NVIDIAA100_SXM4_40GB' -M 256GB -R 'rusage[mem=256GB] span[hosts=1]' -a 'docker(docker121720/pytorch-for-ts:1.25)' 'bash /codes/End_to_end_supervised/ts_optuna_hp_tuning_combined.sh 'icu' 'transformer' > /output/logs/icu-transformer-Aug16.txt'
#LSF_DOCKER_VOLUMES='/storage1/fs1/christopherking/Active/ActFastExports/v1.3.2/:/input/ /storage1/fs1/christopherking/Active/sandhyat/Output-TS_docker_July2024/:/output/ /home/sandhyat/PeriOperative_RiskPrediction/:/codes/' bsub -G 'compute-christopherking' -g '/sandhyat/largeNjob15hpsearchgroup' -n 8 -q general -R 'gpuhost' -gpu 'num=1:gmodel=TeslaV100_SXM2_32GB' -M 256GB -R 'rusage[mem=256GB] span[hosts=1]' -a 'docker(docker121720/pytorch-for-ts:1.25)' 'bash /codes/End_to_end_supervised/ts_optuna_hp_tuning_combined.sh 'icu' 'lstm' > /output/logs/icu-lstm-Aug16.txt'

# first positional argument will be the task second positional argument will be the model with options {'lstm', 'transformer'}

#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "No task provided."
    exit 1
fi

#pip install optuna
#pip uninstall pandas
#pip install pandas==1.5.3

#export CUDA_VISIBLE_DEVICES=1

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

#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='mortality' --modelType=lstm --randomSeed=219 --numtrialsHP=5
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='mortality' --modelType=lstm --randomSeed=219 --numtrialsHP=5
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='mortality' --modelType=lstm --randomSeed=219 --numtrialsHP=5
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='mortality' --modelType=lstm --randomSeed=219 --numtrialsHP=5

#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='aki2' --modelType=lstm --randomSeed=219 --numtrialsHP=3

#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='postop_los' --modelType=lstm --randomSeed=219 --numtrialsHP=3

#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day0' --modelType=lstm --randomSeed=219 --numtrialsHP=3

#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --flow --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3
#python /codes/End_to_end_supervised/Training_with_ts_Optuna_hp_tuning.py --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day1' --modelType=lstm --randomSeed=219 --numtrialsHP=3