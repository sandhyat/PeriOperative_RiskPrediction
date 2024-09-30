# this file is going to take the input task and the ablation details.
# the random initializations will be read inside the file
# this file is expected to run 35 trials (5 random initialization times 7 ablation cases) for one outcome and one model
# since we are using the same hyper parameters from the best one for all ablations, this file will run 5 * 6 ablation cases where both stages are done and 5 runs
# (the last ablation with alerts and postops has already its best results available. We will read those files and put them in the appropriate format)

# to run inside the docker container on RIS
#LSF_DOCKER_VOLUMES='/storage1/fs1/christopherking/Active/ActFastExports/v1.3.2/:/input/ /storage1/fs1/christopherking/Active/sandhyat/Output-TS_docker_July2024/:/output/ /home/sandhyat/PeriOperative_RiskPrediction/:/codes/' bsub -G 'compute-christopherking' -g '/sandhyat/largeNjob15hpsearchgroup' -n 8 -q general -R 'gpuhost' -gpu 'num=1:gmodel=NVIDIAA100_SXM4_40GB' -M 256GB -R 'rusage[mem=256GB] span[hosts=1]' -a 'docker(docker121720/pytorch-for-ts:1.25)' 'bash /codes/Two_stage_selfsupervised/MVCL_best_setting_run.sh'


# the argument taken by this file is the outcome

python /codes/Two_stage_selfsupervised/train_modular.py --meds --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --flow --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --pmhProblist --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --pmhProblist --alerts --postopcomp --onlyEval --bestModel=True --outcome=$1