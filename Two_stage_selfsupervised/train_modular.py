import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
# from datetime import datetime
import json
import time
import datetime
from Multiview_CL_modular import MVCL_f_m_sep
from tasks import eval_classification, eval_classification_sep, eval_classification_sep1, eval_classification_noCL, eval_regression_sep1
import datautils_modular
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:28"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{dir_name}/model_{n}.pkl')
    return callback

if __name__ == '__main__':

    # presetting the number of threads to be used
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.cuda.set_per_process_memory_fraction(1.0, device=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--withoutCL', action="store_true", help='does not use CL but instead directly trains XGBT based on the modalities given')
    parser.add_argument('--preops', action="store_true", help='Whether to add preops to ts representation in case of epic loader')
    parser.add_argument('--flow', action="store_true", help='Whether to add flowsheet to ts representation in case of epic loader')
    parser.add_argument('--meds', action="store_true", help='Whether to add meds to ts representation in case of epic loader')
    parser.add_argument('--alerts', action="store_true", help='Whether to add alerts to ts representation in case of epic loader')
    parser.add_argument('--pmhProblist', action="store_true", help='Whether to add pmh and problist to ts representation in case of epic loader')
    parser.add_argument('--homemeds', action="store_true", help='Whether to add homemeds to ts representation in case of epic loader')
    parser.add_argument('--postopcomp', action="store_true", help='Whether to add postop complications to ts representation in case of epic loader')
    parser.add_argument('--outcome', type=str, required=True, help='The postoperative outcome of interest')
    parser.add_argument('--all_rep', action='store_true', help='Whether to use the representation of all the modalities of only that of time series (flow and meds); to be used with very rare outcomes such as PE or pulm')
    parser.add_argument('--medid_embed_dim', type=int, default=5, help="Dimension to which medid is embedded to before final representations are learnt using ts2vec.")
    parser.add_argument('--alertid_embed_dim', type=int, default=50, help="Dimension to which alert id is embedded to before final representations are learnt using ts2vec.")
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims_f', type=int, default=320, help='The representation dimension for flowsheets (defaults to 320)')
    parser.add_argument('--repr_dims_m', type=int, default=None, help='The representation dimension for medications (defaults to 320)')
    parser.add_argument('--repr_dims_a', type=int, default=None, help='The representation dimension for alerts (defaults to 320)')
    parser.add_argument('--preops_rep_dim_o', type=int, default=80, help=' The representation dimension for preops (originally 101 dimension) ')
    parser.add_argument('--preops_rep_dim_l', type=int, default=84, help=' The representation dimension for labs (originally 110 dimension) ')
    parser.add_argument('--cbow_rep_dim', type=int, default=101, help=' The representation dimension for cbow (originally 101 dimension) ')
    parser.add_argument('--outcome_rep_dim', type=int, default=50, help=' The representation dimension for the outcomes view (originallly 57 + mask for 18) ')
    parser.add_argument('--homemeds_rep_dim', type=int, default=256, help=' The representation dimension for the homemeds view (currently 500) ')
    parser.add_argument('--pmh_rep_dim', type=int, default=85, help=' The representation dimension for the pmh view (currently 1024, 123 for sherbet version) ')
    parser.add_argument('--prob_list_rep_dim', type=int, default=85, help=' The representation dimension for the problem list view (currently 1024, 123 for sherbet version) ')
    parser.add_argument('--proj_dim', type=int, default=100, help=' Common dimension where all the views are projected to.')
    parser.add_argument('--proj_head_depth', type=int, default=2, help=' Depth of the projection head. Same across all the modalities.')
    parser.add_argument('--weight_preops', type=float, default=0.4, help=' Weight multiplier for the preop loss')
    parser.add_argument('--weight_ts_preops', type=float, default=0.2, help=' Weight multipler for the inter view loss')
    parser.add_argument('--weight_outcomes', type=float, default=0.3, help=' Weight multiplier for the outcome loss')
    parser.add_argument('--weight_std', type=float, default=0.3, help=' Weight multiplier for the std reg term')
    parser.add_argument('--weight_cov', type=float, default=0.3, help=' Weight multiplier for the covariance reg term')
    parser.add_argument('--weight_mse', type=float, default=0, help='Weight multiplier for the between modality mse loss')
    parser.add_argument('--weight_ts_cross', type=float, default=0.3, help='Weight multiplier for the between time series modality')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument("--modelType", default='MVCL')  # options {'MVCL'}
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=100, help='The random seed')
    parser.add_argument('--number_runs', type=int, default=5, help='Number of runs with different initial seeds')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', default=True, action='store_true', help='Whether to perform classification based evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument("--onlyEval",action="store_true", help='Whether to perform only classification on a already trained model and or read the existing result file')  #
    parser.add_argument("--bestModel", default="False",
                        help='True when the best HP tuned settings are used on the train+valid setup')  #

    args = parser.parse_args()
    
    print("Arguments:", str(args))

    # breakpoint()
    if (args.withoutCL == True) and (args.all_rep == True):
        print("Incompatible combination")
        exit()

    all_modality_list = ['flow', 'meds', 'alerts', 'pmh', 'problist', 'homemeds', 'postopcomp', 'preops_o', 'preops_l','cbow']
    modality_to_use = []
    if eval('args.preops') == True:
        modality_to_use.append('preops_o')
        modality_to_use.append('preops_l')
        modality_to_use.append('cbow')

    if eval('args.pmhProblist') == True:
        modality_to_use.append('pmh')
        modality_to_use.append('problist')

    if eval('args.homemeds') == True:
        modality_to_use.append('homemeds')

    if eval('args.flow') == True:
        modality_to_use.append('flow')

    if eval('args.meds') == True:
        modality_to_use.append('meds')

    if eval('args.postopcomp') == True:
        modality_to_use.append('postopcomp')

    if eval('args.alerts') == True:
        modality_to_use.append('alerts')

    if (args.withoutCL == True):
        if 'flow' in modality_to_use: modality_to_use.remove('flow')
        if 'meds' in modality_to_use: modality_to_use.remove('meds')
        if 'alerts' in modality_to_use: modality_to_use.remove('alerts')

    # enforcing representation size choices across the encoders
    if args.repr_dims_m == None or args.repr_dims_a == None:
        args.repr_dims_m = args.repr_dims_f
        args.repr_dims_a = args.repr_dims_f


    # input data directory
    # datadir = '/mnt/ris/ActFastExports/v1.3.2/'
    datadir = '/input/'

    # output_dir = './'
    output_dir = '/output/'

    # this is to add to the dir_name
    modalities_to_add = '_modal'

    for i in range(len(modality_to_use)):
        modalities_to_add = modalities_to_add + "_" + modality_to_use[i]


    best_5_random_number = []  # this will take the args when directly run otherwise it will read the number from the file namee
    if eval(args.bestModel) == True:
        # path_to_dir = '/home/trips/PeriOperative_RiskPrediction/HP_output/'
        # sav_dir = '/home/trips/PeriOperative_RiskPrediction/Best_results/Intraoperative/'

        #this is to be used when running the best setting results on RIS
        path_to_dir = output_dir + 'HP_output/'
        sav_dir = output_dir + 'Best_results/Intraoperative/'
        # Best_HPmodel_metadataicu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_6936_24-05-20.json
        file_names = os.listdir(path_to_dir)
        best_5_names = []

        best_5_initial_name = 'Best_HPmodel_metadata_' + args.outcome + "_" + args.modelType + "_modal_"

        modal_name = 'DataModal'
        if eval('args.preops') == True:
            modal_name = modal_name + "_" + 'preops'
        for i in range(len(modality_to_use)):
            if (modality_to_use[i] != 'preops_o') and (modality_to_use[i] != 'preops_l') and (
                    modality_to_use[i] != 'cbow'):
                modal_name = modal_name + "_" + modality_to_use[i]

        dir_name = sav_dir + args.modelType + '/' + modal_name + "_" + str(args.outcome) + "/"
        if eval('args.onlyEval') == True:
            best_5_names1 = {}
            best_5_initial_name1 = 'Best_HPmodel_' + args.outcome + "_" + args.modelType +'_Combined_Perf_metrics_all_modalities_'

            best_5_initial_name2 = 'Best_HPmodel_' + args.outcome + "_" + args.modelType +'_Pred_file_Classification_'
            import re
            labelTest_names = {}
            for file_name in file_names:
                if (best_5_initial_name2 in file_name) and (args.outcome in file_name):
                    if args.outcome not in ['aki2','opioids_count_day0','opioids_count_day1']:
                        match = re.search(r'\d+', file_name)  # this match picks the numbers and if the outcome has a numeric in it, there is a problem
                        if match:
                            number = int(match.group())
                            best_5_names1[int(number)] =file_name
                    else:
                            number = int(file_name.split("_")[-4])
                            best_5_names1[int(number)] =file_name
                elif best_5_initial_name1 in file_name:
                    combined_Result_filename = file_name
                elif ('Best_HPmodel_'+args.outcome in file_name) and (file_name.split("_")[-2] == 'label'):
                    labelTest_names[int(file_name.split("_")[-3])] = file_name
            best_5_names = {}
            for file_name in file_names:
                if (best_5_initial_name in file_name) and (all(elem in file_name for elem in all_modality_list)):  # using all modality list because we will read the all modality best result
                    print(file_name)
                    best_5_names[int(file_name.split("_")[-2])] = file_name
        else:
            for file_name in file_names:
                if (best_5_initial_name in file_name) and (all(elem in file_name for elem in all_modality_list)):  # using all modality list because we will read the all modality best result
                    print(file_name)
                    best_5_names.append(file_name)
                    best_5_random_number.append(int(file_name.split("_")[-2]))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            print(f"The directory '{dir_name}' already exists.")

        best_metadata_dict = {}
    else:
        best_5_random_number.append(args.seed)
        dir_name = './training/'+ args.outcome +'_' + name_with_datetime(modalities_to_add)
        os.makedirs(dir_name, exist_ok=True)

    ## could have derived it but the labels are coming later in the code so can't use the unique function; hence the hardcoding for now
    if args.outcome in ['postop_los', 'opioids_count_day0', 'opioids_count_day1']:
        binary_outcome=False
    else:
        binary_outcome=True

    if binary_outcome:
        perf_metric = np.zeros((len(best_5_random_number), 2))  # 2 is for the metrics auroc and auprc
    else:
        perf_metric = np.zeros((len(best_5_random_number), 5))  # 5 is for the metrics corr, corr_p, R2, MAE, MSE

    if eval('args.onlyEval')==True:
        with open(path_to_dir + combined_Result_filename, 'r') as file: metrics = file.read()

        rows = metrics.strip().split('\n')
        columns = [row.split() for row in rows]
        perf_metric = np.array(columns, dtype=float)

        # sorting this only in this setup to get the correct order of the already recorded performance metric
        myKeys = list(best_5_names.keys())
        myKeys.sort()
        best_5_names = {i: best_5_names[i] for i in myKeys}
        best_5_names1 = {i: best_5_names1[i] for i in myKeys}
        labelTest_names = {i: labelTest_names[i] for i in myKeys}

        count_num = 0
        for runNum in best_5_names.keys():

            best_dict_local = {}
            best_file_name1 = path_to_dir + best_5_names1[runNum]
            out_df = pd.read_csv(best_file_name1)

            best_file_name = path_to_dir + best_5_names[runNum]
            md_f = open(best_file_name)
            config = json.load(md_f)

            config['save_dir'] = dir_name

            label_file  = path_to_dir + labelTest_names[runNum]
            with open(label_file, "rb") as file:
                test_labels = pickle.load(file)

            best_dict_local['randomSeed'] = int(runNum)
            best_dict_local['outcome'] = str(args.outcome)
            best_dict_local['run_number'] = count_num
            best_dict_local['modalities_used'] = modality_to_use
            best_dict_local['model_params'] = config
            best_dict_local['train_orlogids'] = out_df[out_df['train_id_or_not']==1]['orlogid_encoded'].values.tolist()
            best_dict_local['test_orlogids'] = out_df[out_df['train_id_or_not']==0]['orlogid_encoded'].values.tolist()
            if binary_outcome:
                best_dict_local['outcome_rate'] = np.round(test_labels.mean(),decimals=4)

            # this is saving the true and predicted y for each run because the test set is the same
            if count_num == 0:
                outcome_with_pred_test = out_df[out_df['train_id_or_not']==0]
                outcome_with_pred_test['y_true'] = test_labels
                outcome_with_pred_test['y_pred_' + str(runNum)] = outcome_with_pred_test['pred_y']
                outcome_with_pred_test = outcome_with_pred_test.drop(columns=['train_id_or_not', 'pred_y'])
            else:
                outcome_with_pred_test['y_pred_' + str(runNum)] = out_df[out_df['train_id_or_not']==0]['pred_y']
            dict_key = 'run_randomSeed_' + str(runNum)  # this is so dumb because it wont take the key dynamically
            best_metadata_dict[dict_key] = best_dict_local

            count_num = count_num +1
    else:
        for runNum in range(len(best_5_random_number)):
            # starting time of the run
            t = time.time()

            if eval(args.bestModel) == True:
                best_file_name = path_to_dir + best_5_names[runNum]
                md_f = open(best_file_name)
                config = json.load(md_f)
                current_seed_val = int(best_5_random_number[runNum])
                best_dict_local = {}
            else:
                config = dict(
                    medid_embed_dim=args.medid_embed_dim,
                    alertID_embed_dim=args.alertid_embed_dim,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    output_dims_f=args.repr_dims_f,
                    output_dims_m=args.repr_dims_m,
                    output_dims_a=args.repr_dims_a,
                    preops_output_dims_o=args.preops_rep_dim_o,
                    preops_output_dims_l=args.preops_rep_dim_l,
                    cbow_output_dims=args.cbow_rep_dim,
                    homemeds_rep_dims=args.homemeds_rep_dim,
                    pmh_rep_dims=args.pmh_rep_dim,
                    prob_list_rep_dims=args.prob_list_rep_dim,
                    outcome_rep_dims=args.outcome_rep_dim,
                    max_train_length=args.max_train_length,
                    w_pr=args.weight_preops,
                    w_ts_pr=args.weight_ts_preops,
                    w_out=args.weight_outcomes,
                    w_std=args.weight_std,
                    w_cov=args.weight_cov,
                    w_mse=args.weight_mse,
                    w_ts_cross=args.weight_ts_cross,
                    proj_dim=args.proj_dim,
                    head_depth=args.proj_head_depth,
                    save_dir=dir_name
                )
                current_seed_val = int(best_5_random_number[runNum]) * (runNum+1)

            torch.manual_seed(current_seed_val)
            torch.cuda.manual_seed(current_seed_val)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(current_seed_val)

            print('Loading data... ', end='')
            proc_modality_dict_train, proc_modality_dict_test, train_labels, test_labels, outcome_with_orlogid, id_tuple = datautils_modular.load_epic(
                args.outcome, modality_to_use, current_seed_val, data_dir=datadir, out_dir=output_dir)
            if args.save_every is not None:
                unit = 'epoch' if args.epochs is not None else 'iter'
                config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

            if 'flow' in modality_to_use:
                config['input_dims_f'] = proc_modality_dict_train['flow'].shape[-1]

            if 'meds' in modality_to_use:
                config['input_dims_m'] = args.medid_embed_dim  # this is because we are embedding the meds

                total_meds = max(int(proc_modality_dict_train['meds'][:, :, :13].max()), int(
                    proc_modality_dict_test['meds'][:, :,
                    :13].max())) + 1  # total number of medications that are administered during the procedure across patients
                config['med_dim'] = total_meds

            if 'alerts' in modality_to_use:
                config['alert_dim'] = proc_modality_dict_train['alerts'].shape[-1]
                output_file_name = datadir + 'alertsCombID_map.json'
                with open(output_file_name) as outfile:  alert_idmap = json.load(outfile)
                config['alert_Ids'] = len(alert_idmap)

            if 'preops_l' in modality_to_use:
                proc_modality_dict_train['preops_l'] = torch.tensor(proc_modality_dict_train['preops_l'], dtype=torch.float)
                proc_modality_dict_test['preops_l'] = torch.tensor(proc_modality_dict_test['preops_l'], dtype=torch.float)
                config['preops_input_dims_l'] = proc_modality_dict_train['preops_l'].shape[1]

            if 'preops_o' in modality_to_use:
                proc_modality_dict_train['preops_o'] = torch.tensor(proc_modality_dict_train['preops_o'], dtype=torch.float)
                proc_modality_dict_test['preops_o'] = torch.tensor(proc_modality_dict_test['preops_o'], dtype=torch.float)
                config['preops_input_dims_o'] = proc_modality_dict_train['preops_o'].shape[1]

            if 'cbow' in modality_to_use:
                proc_modality_dict_train['cbow'] = torch.tensor(proc_modality_dict_train['cbow'], dtype=torch.float)
                proc_modality_dict_test['cbow'] = torch.tensor(proc_modality_dict_test['cbow'], dtype=torch.float)
                config['cbow_input_dims'] = proc_modality_dict_train['cbow'].shape[1]

            if 'homemeds' in modality_to_use:
                proc_modality_dict_train['homemeds'] = torch.tensor(proc_modality_dict_train['homemeds'], dtype=torch.float)
                proc_modality_dict_test['homemeds'] = torch.tensor(proc_modality_dict_test['homemeds'], dtype=torch.float)
                config['hm_input_dims'] = proc_modality_dict_train['homemeds'].shape[1]

            if 'pmh' in modality_to_use:
                proc_modality_dict_train['pmh'] = torch.tensor(proc_modality_dict_train['pmh'], dtype=torch.float)
                proc_modality_dict_test['pmh'] = torch.tensor(proc_modality_dict_test['pmh'], dtype=torch.float)
                config['pmh_input_dims'] = proc_modality_dict_train['pmh'].shape[1]

            if 'problist' in modality_to_use:
                proc_modality_dict_train['problist'] = torch.tensor(proc_modality_dict_train['problist'], dtype=torch.float)
                proc_modality_dict_test['problist'] = torch.tensor(proc_modality_dict_test['problist'], dtype=torch.float)
                config['prob_list_input_dims'] = proc_modality_dict_train['problist'].shape[1]

            if 'postopcomp' in modality_to_use:
                config['outcome_dim'] = proc_modality_dict_train['postopcomp'].shape[1]

            train_labels = train_labels.reshape(train_labels.shape[0])
            test_labels = test_labels.reshape(test_labels.shape[0])

            device = init_dl_program(args.gpu, seed=current_seed_val, max_threads=args.max_threads)

            # direct xgbt on the modalitites
            if (args.withoutCL == True):  ## this is not being used here
                out, out_tr, eval_res = eval_classification_noCL(proc_modality_dict_train, train_labels,
                                                                 proc_modality_dict_test, test_labels,
                                                                 args.outcome, current_seed_val)
            else:
                if eval(args.bestModel)==True:
                    config['save_dir'] = dir_name
                else:
                    config['seed_used']= current_seed_val
                model = MVCL_f_m_sep(device=device,**config)

                loss_log = model.fit(
                    proc_modality_dict_train,
                    n_epochs=args.epochs,
                    n_iters=args.iters,
                    verbose=True
                )
                # association_metrics_dictRel, association_metrics_dictInter = model.associationBTWalertsANDrestmodalities(proc_modality_dict_test)
                metadata_file = dir_name + '/BestModel_metadata' + str(config['seed_used']) +  '_'+args.outcome + '.json'
                with open(metadata_file, 'w') as outfile: json.dump(config, outfile)

                # not passing the alert data to the classification model because this modality would not be available at test time
                if binary_outcome:
                    preds_te, preds_tr, eval_res = eval_classification_sep1(model, proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels,args.all_rep, args.outcome, int(current_seed_val))
                else:
                    preds_te, preds_tr, eval_res = eval_regression_sep1(model, proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels,args.all_rep, args.outcome, int(current_seed_val))

            pkl_save(f'{dir_name}/{current_seed_val}_out_tr.pkl', preds_tr)  # test set labels and the predictions are already being saved so saving the tr only
            pkl_save(f'{dir_name}/{current_seed_val}_label_tr.pkl', train_labels)
            print('Evaluation result:', eval_res)

            if binary_outcome:
                perf_metric[runNum, 0] = eval_res['auroc']
                perf_metric[runNum, 1] = eval_res['auprc']
            else:
                perf_metric[runNum, 0] = eval_res['CorVal']
                perf_metric[runNum, 1] = eval_res['cor_p_value']
                perf_metric[runNum, 2] = eval_res['r2value']
                perf_metric[runNum, 3] = eval_res['mae']
                perf_metric[runNum, 4] = eval_res['mse']

            if eval(args.bestModel) == True:
                best_dict_local['randomSeed'] = int(best_5_random_number[runNum])
                best_dict_local['outcome'] = str(args.outcome)
                best_dict_local['run_number'] = runNum
                best_dict_local['modalities_used'] = modality_to_use
                best_dict_local['model_params'] = config
                best_dict_local['train_orlogids'] = outcome_with_orlogid.iloc[id_tuple[0]]["orlogid_encoded"].values.tolist()
                best_dict_local['test_orlogids'] = outcome_with_orlogid.iloc[id_tuple[1]]["orlogid_encoded"].values.tolist()
                if binary_outcome:
                    best_dict_local['outcome_rate'] = np.round(outcome_with_orlogid.iloc[id_tuple[1]]["outcome"].mean(), decimals=4)
                # this is saving the true and predicted y for each run because the test set is the same
                if runNum==0:
                    outcome_with_pred_test = outcome_with_orlogid.iloc[id_tuple[1]]
                    outcome_with_pred_test = outcome_with_pred_test.rename(columns={'outcome': 'y_true'})
                    if binary_outcome:
                        outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = preds_te[:, 1]
                    else:
                        outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = preds_te
                else:
                    if binary_outcome:
                        outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = preds_te[:, 1]
                    else:
                        outcome_with_pred_test['y_pred_' + str(int(best_5_random_number[runNum]))] = preds_te
                dict_key = 'run_randomSeed_' + str(int(best_5_random_number[runNum]))  # this is so dumb because it wont take the key dynamically
                best_metadata_dict[dict_key] = best_dict_local

            if binary_outcome:
                fpr_roc, tpr_roc, thresholds_roc = roc_curve(test_labels, preds_te[:, 1], drop_intermediate=False)
                precision_prc, recall_prc, thresholds_prc = precision_recall_curve(test_labels, preds_te[:, 1])
                # interpolation in ROC
                mean_fpr = np.linspace(0, 1, 100)
                tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
                mean_fpr = np.round(mean_fpr, decimals=2)
                print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))

            t = time.time() - t
            print(f"\ntime taken to finish run number: {datetime.timedelta(seconds=t)}\n")

    print("Tranquila")

    # saving metadata for all best runs in json; decided to save it also as pickle because the nested datatypes were not letting it be serializable
    metadata_filename = dir_name + '/Best_runs_metadata.pickle'
    with open(metadata_filename, 'wb') as outfile:
        pickle.dump(best_metadata_dict, outfile)

    # saving the performance metrics from all best runs and all models in a pickle file
    perf_filename = sav_dir + str(args.outcome) + '_Best_perf_metrics_combined_intraoperative.pickle'
    if not os.path.exists(perf_filename):
        data = {}
        data[str(args.modelType)] = {modal_name: perf_metric}
        with open(perf_filename, 'wb') as file:
            pickle.dump(data, file)
    else:
        with open(perf_filename, 'rb') as file:
            existing_data = pickle.load(file)

        try:
            existing_data[str(args.modelType)][modal_name] = perf_metric
        except(KeyError):  # this is to take care of the situation when a new model is added to the file
            existing_data[str(args.modelType)] = {}
            existing_data[str(args.modelType)][modal_name] = perf_metric

        # Save the updated dictionary back to the pickle file
        with open(perf_filename, 'wb') as file:
            pickle.dump(existing_data, file)

    # saving the test set predictions for all models and all runs
    pred_filename = sav_dir + str(args.outcome) + '_Best_pred_combined_intraoperative.pickle'
    if not os.path.exists(pred_filename):
        data = {}
        data[str(args.modelType)] = {modal_name: outcome_with_pred_test.values}
        with open(pred_filename, 'wb') as file:
            pickle.dump(data, file)
    else:
        with open(pred_filename, 'rb') as file:
            existing_data = pickle.load(file)

        try:
            existing_data[str(args.modelType)][modal_name] = outcome_with_pred_test.values
        except(KeyError):  # this is to take care of the situation when a new model is added to the file
            existing_data[str(args.modelType)] = {}
            existing_data[str(args.modelType)][modal_name] = outcome_with_pred_test.values

        # Save the updated dictionary back to the pickle file
        with open(pred_filename, 'wb') as file:
            pickle.dump(existing_data, file)
