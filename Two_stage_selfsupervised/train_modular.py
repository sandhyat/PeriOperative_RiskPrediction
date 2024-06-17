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
from tasks import eval_classification, eval_classification_sep, eval_classification_sep1, eval_classification_noCL
import datautils_modular
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

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
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':

    # presetting the number of threads to be used
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.cuda.set_per_process_memory_fraction(1.0, device=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--withoutCL', action="store_true", help='does not use CL but instead directly trains XGBT based on the modalities given')
    parser.add_argument('--preops', action="store_true", help='Whether to add preops to ts representation in case of epic loader')
    parser.add_argument('--meds', action="store_true", help='Whether to add meds to ts representation in case of epic loader')
    parser.add_argument('--alerts', action="store_true", help='Whether to add alerts to ts representation in case of epic loader')
    parser.add_argument('--pmh', action="store_true", help='Whether to add pmh to ts representation in case of epic loader')
    parser.add_argument('--problist', action="store_true", help='Whether to add problist to ts representation in case of epic loader')
    parser.add_argument('--homemeds', action="store_true", help='Whether to add homemeds to ts representation in case of epic loader')
    parser.add_argument('--postopcomp', action="store_true", help='Whether to add postop complications to ts representation in case of epic loader')
    parser.add_argument('--outcome', type=str, required=True, help='The postoperative outcome of interest')
    parser.add_argument('--all_rep', action='store_true', help='Whether to use the representation of all the modalities of only that of time series (flow and meds); to be used with very rare outcomes such as PE or pulm')
    parser.add_argument('--medid_embed_dim', type=int, default=5, help="Dimension to which medid is embedded to before final representations are learnt using ts2vec.")
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
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--number_runs', type=int, default=5, help='Number of runs with different initial seeds')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument("--outputcsv", default="")  #
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
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

    for i in all_modality_list:
        if i in ['flow']:
            modality_to_use.append(i)
        elif i in ['preops_o', 'preops_l','cbow']:
            print('for completeness') # should have taken care of in a different way
        elif eval('args.'+str(i)) == True:
            modality_to_use.append(i)

    if (args.withoutCL == True):
        modality_to_use.remove('flow')
        if 'meds' in modality_to_use: modality_to_use.remove('meds')
        if 'alerts' in modality_to_use: modality_to_use.remove('alerts')

    # enforcing representation size choices across the encoders
    if args.repr_dims_m == None or args.repr_dims_a == None:
        args.repr_dims_m = args.repr_dims_f
        args.repr_dims_a = args.repr_dims_f

    # this is to add to the dir_name
    modalities_to_add = '_modal_'
    for i in range(len(modality_to_use)):
        modalities_to_add = modalities_to_add + "_" + modality_to_use[i]

    # run_dir = '/output/training/WOCalib_' + args.outcome + modalities_to_add + '_' + name_with_datetime(args.run_name)
    run_dir = '/output/training/' + args.outcome + modalities_to_add + '_' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # input data directory
    datadir = '/input/'

    print('Loading data... ', end='')
    proc_modality_dict_train, proc_modality_dict_test, train_labels, test_labels, train_idx_df, test_idx_df = datautils_modular.load_epic(args.dataset, args.outcome, modality_to_use, data_dir=datadir)


    # run_dir = 'training/icu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_multipleCL_20240509_135453'
    #
    # metadata_file = run_dir + '/model_metadata.json'
    # with open(metadata_file) as f:
    #     config = json.load(f)

    config = dict(
        medid_embed_dim=args.medid_embed_dim,
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
        save_dir=run_dir
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    t = time.time()

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

    perf_metric = np.zeros((args.number_runs, 2)) # 2 is for the metrics
    for i in range(args.number_runs):
        device = init_dl_program(args.gpu, seed=(args.seed * (i+1)), max_threads=args.max_threads)
        # direct xgbt on the modalitites
        if (args.withoutCL == True):
            out, out_tr, eval_res = eval_classification_noCL(proc_modality_dict_train, train_labels,
                                                             proc_modality_dict_test, test_labels,
                                                             args.outcome, (args.seed * (i+1)))
        else:
            config['seed_used']= (args.seed * (i+1))
            model = MVCL_f_m_sep(device=device,**config)
            # association_metrics_dict = model.associationBTWalertsANDrestmodalities(proc_modality_dict_test)
            loss_log = model.fit(
                proc_modality_dict_train,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True
            )

            metadata_file = run_dir + '/' + str(config['seed_used']) + '_model_metadata.json'
            with open(metadata_file, 'w') as outfile:
                json.dump(config, outfile)

            t = time.time() - t
            print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")


            # not passing the alert data to the classification model because this modality would not be available at test time
            if args.eval:
                out, out_tr, eval_res = eval_classification_sep1(model, proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels,args.all_rep, args.outcome, (args.seed * (i+1)), train_idx_df, test_idx_df)

        pkl_save(f'{run_dir}/{args.seed * (i+1)}_out.pkl', out)
        pkl_save(f'{run_dir}/{args.seed * (i+1)}_out_tr.pkl', out_tr)
        pkl_save(f'{run_dir}/{args.seed * (i+1)}_label_te.pkl', test_labels)
        pkl_save(f'{run_dir}/{args.seed * (i+1)}_label_tr_tr.pkl', train_labels)
        pkl_save(f'{run_dir}/{args.seed * (i+1)}_eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)
        train_idx_df['pred_y']=out_tr[:,1]
        test_idx_df['pred_y']=out[:,1]

        perf_metric[i, 0] = eval_res['auroc']
        perf_metric[i, 1] = eval_res['auprc']


        file_to_save_df = pd.concat([train_idx_df, test_idx_df], axis=0)

        file_to_save_df.to_csv('/output/TrainedModels/' + modalities_to_add + '_' + 'WOCalib_Best_Pred_file_Classification_' + str((args.seed * (i+1))) + "_"  + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+".csv", index=False)
        # file_to_save_df.to_csv('/output/TrainedModels/' + modalities_to_add + '_' + 'Best_Pred_file_Classification_' + str((args.seed * (i+1))) + "_"  + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+".csv", index=False)
        # file_to_save_df.to_csv('./' + modalities_to_add + '_' + 'Best_Pred_file_Classification_' + str((args.seed * (i+1))) + "_"  + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+".csv", index=False)

        csvdata = {
            'hp': json.dumps(vars(args)),
            'outcome_rate': np.round(sum(train_labels) / len(train_labels), decimals=4),
            'AUROC': eval_res['auroc'],
            'AUPRC': eval_res['auprc'],
            'target': args.outcome,
            'random_seed':(args.seed * (i+1)),
            # 'random_seed': i*100,
            'evaltime': datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        }

        print(csvdata)
        csvdata = pd.DataFrame(csvdata, index=[0])
        outputcsv = os.path.join('/output/', args.outputcsv)
        if (os.path.exists(outputcsv)):
            csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
        else:
            csvdata.to_csv(outputcsv, header=True, index=False)

    np.savetxt('/output/TrainedModels/' + modalities_to_add + '_' + 'WOCalib_Combined_Perf_metrics' + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+'.txt', perf_metric)
    # np.savetxt('/output/TrainedModels/' + modalities_to_add + '_' + 'Combined_Perf_metrics' + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+'.txt', perf_metric)
    # np.savetxt('./' + modalities_to_add + '_' + 'Combined_Perf_metrics' + str(args.outcome) + "_all_modalities_" +str(datetime.datetime.now().strftime("%y-%m-%d"))+'.txt', perf_metric)
    print("Averaged AUROC with different seeds ", np.mean(perf_metric[:,0], axis=0))
    print("Averaged AUPRC with different seeds ", np.mean(perf_metric[:,1], axis=0))
    print("STD AUROC with different seeds ", np.std(perf_metric[:,0], axis=0))
    print("STD AUPRC with different seeds ", np.std(perf_metric[:,1], axis=0))
    print("Finished.")
