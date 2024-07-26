# this file is going to take the input task and the ablation details.
# the random initializations will be read inside the file
# this file is expected to run 35 trials (5 random initialization times 7 ablation cases) for one outcome and one model
# since we are using the same hyper parameters from the best one for all ablations, this file will run 5 * 6 ablation cases where both stages are done and 5 runs
# (the last ablation with alerts and postops has already its best results available. We will read those files and put them in the appropriate format)

# the argument taken by this file is the outcome

python /codes/Two_stage_selfsupervised/train_modular.py --meds --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --flow --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --pmhProblist --outcome=$1 --bestModel=True
python /codes/Two_stage_selfsupervised/train_modular.py --meds --flow --preops --homemeds --pmhProblist --alerts --postopcomp --onlyEval --bestModel=True --outcome=$1