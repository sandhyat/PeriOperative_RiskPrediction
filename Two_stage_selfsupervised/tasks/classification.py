import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from . import scarf_model as preop_model
from . import loss as scarf_loss
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


def eval_classification_sep(model, train_data_f, train_data_m, train_pr, train_pr_l, train_bw, train_hm, train_pmh, train_pblist, train_labels, test_data_f, test_data_m, test_pr, test_pr_l, test_bw, test_hm, test_pmh, test_pblist, test_labels, includePreops,
                        outcome, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    # breakpoint()

    train_repr_f = model.encode(train_data_f, 'f', encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr_f = model.encode(test_data_f,  'f', encoding_window='full_series' if train_labels.ndim == 1 else None)

    # breakpoint()
    train_repr_m = model.encode(train_data_m, 'm', encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr_m = model.encode(test_data_m,  'm', encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'


    # breakpoint()

    # # data_dir = '/input/'
    # data_dir = 'datasets/Epic/'
    #
    # if includePreops:
    #     preops_train = np.load(data_dir + "preops_proc_train.npy")
    #     preops_test = np.load(data_dir + "preops_proc_test.npy")
    #     preops_valid = np.load(data_dir + "preops_proc_test.npy")
    #     #
    #     # train_X = np.load(data_dir + "preops_proc_train.npy")
    #     # test_X = np.load(data_dir + "preops_proc_test.npy")
    #     # preops_valid = np.load(data_dir + "preops_proc_test.npy")
    #
    #     cbow_train = np.load(data_dir + "cbow_proc_train.npy")
    #     cbow_test = np.load(data_dir + "cbow_proc_test.npy")
    #
    #     # train_X_cbow = np.load(data_dir + "cbow_proc_train.npy")
    #     # test_X_cbow = np.load(data_dir + "cbow_proc_test.npy")
    #
    #     if outcome=='icu': # evaluation only on the non preplanned ICU cases
    #         preops_raw = pd.read_csv(data_dir + "Raw_preops_used_in_ICU.csv")
    #         train_idx = pd.read_csv(data_dir + "train_test_id_orlogid_map.csv")
    #         test_index_orig = train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values
    #         test_index = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3][
    #             'plannedDispo'].index
    #
    #         # breakpoint()
    #
    #         preops_test = preops_test[test_index]
    #         cbow_test = cbow_test[test_index]
    #
    #         # test_X = test_X[test_index]
    #         # test_X_cbow = test_X_cbow[test_index]
    #
    #         del preops_raw, test_index, train_idx
    #
    #
    #     # is the scaling needed again as the preops have been processes already?
    #     scaler = StandardScaler()
    #     scaler.fit(preops_train)
    #     train_X = scaler.transform(preops_train)
    #     test_X = scaler.transform(preops_test)
    #
    #     scaler_cbow = StandardScaler()
    #     scaler_cbow.fit(cbow_train)
    #     train_X_cbow = scaler_cbow.transform(cbow_train)
    #     test_X_cbow = scaler_cbow.transform(cbow_test)
    #
    #     # breakpoint()
    #     # snippet to obtain the CL base preop representation
    #     train_ds = preop_model.ExampleDataset(train_X,train_labels)
    #     test_ds = preop_model.ExampleDataset(test_X,test_labels)
    #     # breakpoint()
    #     batch_size = 128
    #     epochs = 50
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #
    #     model_pr = preop_model.SCARF(
    #         input_dim=train_ds.shape[1],
    #         emb_dim=100,
    #         corruption_rate=0.6,
    #     ).to(device)
    #     optimizer = Adam(model_pr.parameters(), lr=0.001)
    #     ntxent_loss = scarf_loss.NTXent()
    #     # breakpoint()
    #     loss_history = []
    #
    #     for epoch in range(1, epochs + 1):
    #         epoch_loss = preop_model.train_epoch(model_pr, ntxent_loss, train_loader, optimizer, device, epoch)
    #         loss_history.append(epoch_loss)
    #
    #
    #     # final preop embeddings
    #
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    #     # breakpoint()
    #     # get embeddings for training and test set
    #     train_repr_pr = preop_model.dataset_embeddings(model_pr, train_loader, device)
    #     test_repr_pr = preop_model.dataset_embeddings(model_pr, test_loader, device)
    #
    #     # breakpoint()
    #     train_repr = np.concatenate((train_repr, train_repr_pr, train_X_cbow), axis=1)
    #     test_repr = np.concatenate((test_repr, test_repr_pr, test_X_cbow), axis=1)

    if False:  # this part is to obtain the preop representation after training and using the lower dim rep in the classifier training
        train_ds = preop_model.ExampleDataset(train_pr, train_labels)
        test_ds = preop_model.ExampleDataset(test_pr, test_labels)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        # breakpoint()
        # get embeddings for training and test set
        train_repr_pr = model.pr_dataset_embeddings(train_loader)
        test_repr_pr = model.pr_dataset_embeddings(test_loader)
    else:
        train_repr_pr = np.concatenate((train_pr, train_pr_l, train_bw, train_hm, train_pmh, train_pblist), axis=1)
        test_repr_pr = np.concatenate((test_pr, test_pr_l, test_bw, test_hm, test_pmh, test_pblist), axis=1)

    # breakpoint()
    train_repr = np.concatenate((train_repr_f, train_repr_m, train_repr_pr), axis=1)
    test_repr = np.concatenate((test_repr_f, test_repr_m, test_repr_pr), axis=1)

    clf = fit_clf(train_repr[:-3000, :], train_labels[:-3000])

    # calibration function
    from sklearn.calibration import CalibratedClassifierCV

    calibrated_clf = CalibratedClassifierCV(clf, cv = 'prefit', method="isotonic")
    calibrated_clf.fit(train_repr[-3000:, :], train_labels[-3000:])

    acc = calibrated_clf.score(test_repr, test_labels)

    # acc = clf.score(test_repr, test_labels)

    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = calibrated_clf.predict_proba(test_repr) # this is from the calibrated classifier
        y_score_tr = calibrated_clf.predict_proba(train_repr) # this is from the calibrated classifier
        auprc = average_precision_score(test_labels, y_score[:, 1])
        auroc = roc_auc_score(test_labels, y_score[:, 1])
    else:
        y_score = clf.decision_function(test_repr)
        y_score_tr = clf.decision_function(train_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)

    return y_score, y_score_tr, {'acc': acc, 'auprc': auprc, 'auroc': auroc}

def eval_classification(model, train_data, train_pr, train_labels, test_data, test_pr, test_labels, includePreops, outcome, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    # breakpoint()

    # # data_dir = '/input/'
    # data_dir = 'datasets/Epic/'
    #
    # if includePreops:
    #     preops_train = np.load(data_dir + "preops_proc_train.npy")
    #     preops_test = np.load(data_dir + "preops_proc_test.npy")
    #     preops_valid = np.load(data_dir + "preops_proc_test.npy")
    #     #
    #     # train_X = np.load(data_dir + "preops_proc_train.npy")
    #     # test_X = np.load(data_dir + "preops_proc_test.npy")
    #     # preops_valid = np.load(data_dir + "preops_proc_test.npy")
    #
    #     cbow_train = np.load(data_dir + "cbow_proc_train.npy")
    #     cbow_test = np.load(data_dir + "cbow_proc_test.npy")
    #
    #     # train_X_cbow = np.load(data_dir + "cbow_proc_train.npy")
    #     # test_X_cbow = np.load(data_dir + "cbow_proc_test.npy")
    #
    #     if outcome=='icu': # evaluation only on the non preplanned ICU cases
    #         preops_raw = pd.read_csv(data_dir + "Raw_preops_used_in_ICU.csv")
    #         train_idx = pd.read_csv(data_dir + "train_test_id_orlogid_map.csv")
    #         test_index_orig = train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values
    #         test_index = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3][
    #             'plannedDispo'].index
    #
    #         # breakpoint()
    #
    #         preops_test = preops_test[test_index]
    #         cbow_test = cbow_test[test_index]
    #
    #         # test_X = test_X[test_index]
    #         # test_X_cbow = test_X_cbow[test_index]
    #
    #         del preops_raw, test_index, train_idx
    #
    #
    #     # is the scaling needed again as the preops have been processes already?
    #     scaler = StandardScaler()
    #     scaler.fit(preops_train)
    #     train_X = scaler.transform(preops_train)
    #     test_X = scaler.transform(preops_test)
    #
    #     scaler_cbow = StandardScaler()
    #     scaler_cbow.fit(cbow_train)
    #     train_X_cbow = scaler_cbow.transform(cbow_train)
    #     test_X_cbow = scaler_cbow.transform(cbow_test)
    #
    #     # breakpoint()
    #     # snippet to obtain the CL base preop representation
    #     train_ds = preop_model.ExampleDataset(train_X,train_labels)
    #     test_ds = preop_model.ExampleDataset(test_X,test_labels)
    #     # breakpoint()
    #     batch_size = 128
    #     epochs = 50
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #
    #     model_pr = preop_model.SCARF(
    #         input_dim=train_ds.shape[1],
    #         emb_dim=100,
    #         corruption_rate=0.6,
    #     ).to(device)
    #     optimizer = Adam(model_pr.parameters(), lr=0.001)
    #     ntxent_loss = scarf_loss.NTXent()
    #     # breakpoint()
    #     loss_history = []
    #
    #     for epoch in range(1, epochs + 1):
    #         epoch_loss = preop_model.train_epoch(model_pr, ntxent_loss, train_loader, optimizer, device, epoch)
    #         loss_history.append(epoch_loss)
    #
    #
    #     # final preop embeddings
    #
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    #     # breakpoint()
    #     # get embeddings for training and test set
    #     train_repr_pr = preop_model.dataset_embeddings(model_pr, train_loader, device)
    #     test_repr_pr = preop_model.dataset_embeddings(model_pr, test_loader, device)
    #
    #     # breakpoint()
    #     train_repr = np.concatenate((train_repr, train_repr_pr, train_X_cbow), axis=1)
    #     test_repr = np.concatenate((test_repr, test_repr_pr, test_X_cbow), axis=1)

    if False: # this part is to obtain the preop representation after training and using the lower dim rep in the classifier training
        train_ds = preop_model.ExampleDataset(train_pr,train_labels)
        test_ds = preop_model.ExampleDataset(test_pr,test_labels)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        # breakpoint()
        # get embeddings for training and test set
        train_repr_pr = model.pr_dataset_embeddings(train_loader)
        test_repr_pr = model.pr_dataset_embeddings(test_loader)
    else:
        train_repr_pr = train_pr
        test_repr_pr = test_pr

    # breakpoint()
    train_repr = np.concatenate((train_repr, train_repr_pr), axis=1)
    test_repr = np.concatenate((test_repr, test_repr_pr), axis=1)


    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)


    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = clf.predict_proba(test_repr)
        auprc = average_precision_score(test_labels, y_score[:,1])
        auroc = roc_auc_score(test_labels, y_score[:,1])
    else:
        y_score = clf.decision_function(test_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc, 'auroc': auroc }


def eval_classification_sep1(model, proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels, all_rep,
                        outcome, randomSeed, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    modalities_selected = proc_modality_dict_train.keys()
    # breakpoint()
    if 'flow' in modalities_selected:
        train_data_f = proc_modality_dict_train['flow']
        test_data_f = proc_modality_dict_test['flow']
        train_repr_f = model.encode(train_data_f, 'f', encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr_f = model.encode(test_data_f,  'f', encoding_window='full_series' if train_labels.ndim == 1 else None)

    if 'meds' in modalities_selected:
        train_data_m = proc_modality_dict_train['meds']
        test_data_m = proc_modality_dict_test['meds']
        train_repr_m = model.encode(train_data_m, 'm', encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr_m = model.encode(test_data_m,  'm', encoding_window='full_series' if train_labels.ndim == 1 else None)

    if 'preops_o' in modalities_selected:
        train_pr = proc_modality_dict_train['preops_o']
        test_pr = proc_modality_dict_test['preops_o']
        train_pr_l = proc_modality_dict_train['preops_l']
        test_pr_l = proc_modality_dict_test['preops_l']
        train_bw = proc_modality_dict_train['cbow']
        test_bw = proc_modality_dict_test['cbow']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pr, train_labels)
            test_ds = preop_model.ExampleDataset(test_pr, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pr = model.pr_dataset_embeddings(train_loader)
            test_repr_pr = model.pr_dataset_embeddings(test_loader)

            train_ds = preop_model.ExampleDataset(train_pr_l, train_labels)
            test_ds = preop_model.ExampleDataset(test_pr_l, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pr_l = model.pr_l_dataset_embeddings(train_loader)
            test_repr_pr_l = model.pr_l_dataset_embeddings(test_loader)

            train_ds = preop_model.ExampleDataset(train_bw, train_labels)
            test_ds = preop_model.ExampleDataset(test_bw, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_bw = model.cbow_dataset_embeddings(train_loader)
            test_repr_bw = model.cbow_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pr_l, train_repr_bw), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pr_l, test_repr_bw), axis=1)

        else:
            train_repr_pr = np.concatenate((train_pr, train_pr_l, train_bw), axis=1)
            test_repr_pr = np.concatenate((test_pr, test_pr_l, test_bw), axis=1)

    if 'homemeds' in modalities_selected:
        train_hm = proc_modality_dict_train['homemeds']
        test_hm = proc_modality_dict_test['homemeds']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_hm, train_labels)
            test_ds = preop_model.ExampleDataset(test_hm, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_hm = model.hm_dataset_embeddings(train_loader)
            test_repr_hm = model.hm_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_hm), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_hm), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_hm), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_hm), axis=1)

    if 'pmh' in modalities_selected:
        train_pmh = proc_modality_dict_train['pmh']
        test_pmh = proc_modality_dict_test['pmh']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pmh, train_labels)
            test_ds = preop_model.ExampleDataset(test_pmh, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pmh = model.pmh_dataset_embeddings(train_loader)
            test_repr_pmh = model.pmh_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pmh), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pmh), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_pmh), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_pmh), axis=1)

    if 'problist' in modalities_selected:
        train_pblist = proc_modality_dict_train['problist']
        test_pblist = proc_modality_dict_test['problist']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pblist, train_labels)
            test_ds = preop_model.ExampleDataset(test_pblist, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pblist = model.problist_dataset_embeddings(train_loader)
            test_repr_pblist = model.problist_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pblist), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pblist), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_pblist), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_pblist), axis=1)

    # breakpoint()

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
        # fit_clf = eval_protocols.fit_xgbt_cv
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    # breakpoint()
    if ('flow' in modalities_selected) and ('meds' in modalities_selected) and ('preops_o' in modalities_selected):
        train_repr = np.concatenate((train_repr_f, train_repr_m, train_repr_pr), axis=1)
        test_repr = np.concatenate((test_repr_f, test_repr_m, test_repr_pr), axis=1)
    elif ('flow' in modalities_selected) and ('meds' in modalities_selected):
        train_repr = np.concatenate((train_repr_f, train_repr_m), axis=1)
        test_repr = np.concatenate((test_repr_f, test_repr_m), axis=1)
    elif ('flow' in modalities_selected):
        train_repr = train_repr_f
        test_repr = test_repr_f
    elif ('meds' in modalities_selected):
        train_repr = train_repr_m
        test_repr = test_repr_m

    clf = fit_clf(train_repr, train_labels, seed=randomSeed)
    if False:
        # breakpoint()
        clf = fit_clf(train_repr[:-3000, :], train_labels[:-3000], seed=randomSeed)

        # calibration function
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_clf = CalibratedClassifierCV(clf, cv='prefit', method="isotonic")
        calibrated_clf.fit(train_repr[-3000:, :], train_labels[-3000:])

        acc = calibrated_clf.score(test_repr, test_labels)

    acc = clf.score(test_repr, test_labels)

    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = clf.predict_proba(test_repr) # this is from the non-calibrated classifier
        y_score_tr = clf.predict_proba(train_repr) # this is from the non-calibrated classifier
        # y_score = calibrated_clf.predict_proba(test_repr) # this is from the calibrated classifier
        # y_score_tr = calibrated_clf.predict_proba(train_repr) # this is from the calibrated classifier
        auprc = average_precision_score(test_labels, y_score[:, 1])
        auroc = roc_auc_score(test_labels, y_score[:, 1])
    else:
        y_score = clf.decision_function(test_repr)
        y_score_tr = clf.decision_function(train_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)

    return y_score, y_score_tr, {'acc': acc, 'auprc': auprc, 'auroc': auroc}

def eval_classification_sep1_tUrl(model, proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels, all_rep,
                        outcome, randomSeed, train_idx, test_idx,rep_save=0, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    modalities_selected = proc_modality_dict_train.keys()
    if 'flow' in modalities_selected:
        train_data_f = proc_modality_dict_train['flow']
        test_data_f = proc_modality_dict_test['flow']
        train_repr_f = model.encode(train_data_f, 'f', encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr_f = model.encode(test_data_f,  'f', encoding_window='full_series' if train_labels.ndim == 1 else None)
        if rep_save==1:
            test_rep_idx_f = pd.concat([test_idx.reset_index(drop=True), pd.DataFrame(test_repr_f, columns=['Col'+str(i) for i in range(test_repr_f.shape[-1])]).reset_index(drop=True)], axis=1)
            train_rep_idx_f = pd.concat([train_idx.reset_index(drop=True), pd.DataFrame(train_repr_f, columns=['Col'+str(i) for i in range(train_repr_f.shape[-1])]).reset_index(drop=True)], axis=1)

    if 'meds' in modalities_selected:
        train_data_m = proc_modality_dict_train['meds']
        test_data_m = proc_modality_dict_test['meds']
        train_repr_m = model.encode(train_data_m, 'm', encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr_m = model.encode(test_data_m,  'm', encoding_window='full_series' if train_labels.ndim == 1 else None)
        if rep_save==1:
            test_rep_idx_m = pd.concat([test_idx.reset_index(drop=True), pd.DataFrame(test_repr_m, columns=['Col'+str(i) for i in range(test_repr_m.shape[-1])]).reset_index(drop=True)], axis=1)
            train_rep_idx_m = pd.concat([train_idx.reset_index(drop=True), pd.DataFrame(train_repr_m, columns=['Col'+str(i) for i in range(train_repr_m.shape[-1])]).reset_index(drop=True)], axis=1)


    if 'preops_o' in modalities_selected:
        train_pr = proc_modality_dict_train['preops_o']
        test_pr = proc_modality_dict_test['preops_o']
        train_pr_l = proc_modality_dict_train['preops_l']
        test_pr_l = proc_modality_dict_test['preops_l']
        train_bw = proc_modality_dict_train['cbow']
        test_bw = proc_modality_dict_test['cbow']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pr, train_labels)
            test_ds = preop_model.ExampleDataset(test_pr, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pr = model.pr_dataset_embeddings(train_loader)
            test_repr_pr = model.pr_dataset_embeddings(test_loader)

            train_ds = preop_model.ExampleDataset(train_pr_l, train_labels)
            test_ds = preop_model.ExampleDataset(test_pr_l, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pr_l = model.pr_l_dataset_embeddings(train_loader)
            test_repr_pr_l = model.pr_l_dataset_embeddings(test_loader)

            train_ds = preop_model.ExampleDataset(train_bw, train_labels)
            test_ds = preop_model.ExampleDataset(test_bw, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_bw = model.cbow_dataset_embeddings(train_loader)
            test_repr_bw = model.cbow_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pr_l, train_repr_bw), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pr_l, test_repr_bw), axis=1)

        else:
            train_repr_pr = np.concatenate((train_pr, train_pr_l, train_bw), axis=1)
            test_repr_pr = np.concatenate((test_pr, test_pr_l, test_bw), axis=1)

    if 'homemeds' in modalities_selected:
        train_hm = proc_modality_dict_train['homemeds']
        test_hm = proc_modality_dict_test['homemeds']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_hm, train_labels)
            test_ds = preop_model.ExampleDataset(test_hm, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_hm = model.hm_dataset_embeddings(train_loader)
            test_repr_hm = model.hm_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_hm), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_hm), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_hm), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_hm), axis=1)

    if 'pmh' in modalities_selected:
        train_pmh = proc_modality_dict_train['pmh']
        test_pmh = proc_modality_dict_test['pmh']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pmh, train_labels)
            test_ds = preop_model.ExampleDataset(test_pmh, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pmh = model.pmh_dataset_embeddings(train_loader)
            test_repr_pmh = model.pmh_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pmh), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pmh), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_pmh), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_pmh), axis=1)

    if 'problist' in modalities_selected:
        train_pblist = proc_modality_dict_train['problist']
        test_pblist = proc_modality_dict_test['problist']
        if all_rep == True:
            train_ds = preop_model.ExampleDataset(train_pblist, train_labels)
            test_ds = preop_model.ExampleDataset(test_pblist, test_labels)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            train_repr_pblist = model.problist_dataset_embeddings(train_loader)
            test_repr_pblist = model.problist_dataset_embeddings(test_loader)

            train_repr_pr = np.concatenate((train_repr_pr, train_repr_pblist), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_repr_pblist), axis=1)
        else:
            train_repr_pr = np.concatenate((train_repr_pr, train_pblist), axis=1)
            test_repr_pr = np.concatenate((test_repr_pr, test_pblist), axis=1)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
        # fit_clf = eval_protocols.fit_xgbt_cv
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    if ('flow' in modalities_selected) and ('meds' in modalities_selected) and ('preops_o' in modalities_selected):
        train_repr = np.concatenate((train_repr_f, train_repr_m, train_repr_pr), axis=1)
        test_repr = np.concatenate((test_repr_f, test_repr_m, test_repr_pr), axis=1)
    elif ('flow' in modalities_selected) and ('meds' in modalities_selected):
        train_repr = np.concatenate((train_repr_f, train_repr_m), axis=1)
        test_repr = np.concatenate((test_repr_f, test_repr_m), axis=1)
    else:
        train_repr = train_repr_f
        test_repr = test_repr_f

    clf = fit_clf(train_repr, train_labels, seed=randomSeed)
    if False:
        # breakpoint()
        clf = fit_clf(train_repr[:-3000, :], train_labels[:-3000], seed=randomSeed)

        # calibration function
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_clf = CalibratedClassifierCV(clf, cv='prefit', method="isotonic")
        calibrated_clf.fit(train_repr[-3000:, :], train_labels[-3000:])

        acc = calibrated_clf.score(test_repr, test_labels)

    acc = clf.score(test_repr, test_labels)

    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = clf.predict_proba(test_repr) # this is from the non-calibrated classifier
        y_score_tr = clf.predict_proba(train_repr) # this is from the non-calibrated classifier
        # y_score = calibrated_clf.predict_proba(test_repr) # this is from the calibrated classifier
        # y_score_tr = calibrated_clf.predict_proba(train_repr) # this is from the calibrated classifier
        auprc = average_precision_score(test_labels, y_score[:, 1])
        auroc = roc_auc_score(test_labels, y_score[:, 1])
    else:
        y_score = clf.decision_function(test_repr)
        y_score_tr = clf.decision_function(train_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)

    return y_score, y_score_tr, {'acc': acc, 'auprc': auprc, 'auroc': auroc}

def eval_classification_noCL(proc_modality_dict_train, train_labels, proc_modality_dict_test, test_labels,
                        outcomeGiven, randomSeed, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    modalities_selected = proc_modality_dict_train.keys()


    if 'preops_o' in modalities_selected:
        train_pr = proc_modality_dict_train['preops_o']
        test_pr = proc_modality_dict_test['preops_o']
        train_pr_l = proc_modality_dict_train['preops_l']
        test_pr_l = proc_modality_dict_test['preops_l']
        train_bw = proc_modality_dict_train['cbow']
        test_bw = proc_modality_dict_test['cbow']
        train_repr_pr = np.concatenate((train_pr, train_pr_l, train_bw), axis=1)
        test_repr_pr = np.concatenate((test_pr, test_pr_l, test_bw), axis=1)

    if 'homemeds' in modalities_selected:
        train_hm = proc_modality_dict_train['homemeds']
        test_hm = proc_modality_dict_test['homemeds']
        train_repr_pr = np.concatenate((train_repr_pr, train_hm), axis=1)
        test_repr_pr = np.concatenate((test_repr_pr, test_hm), axis=1)

    if 'pmh' in modalities_selected:
        train_pmh = proc_modality_dict_train['pmh']
        test_pmh = proc_modality_dict_test['pmh']
        train_repr_pr = np.concatenate((train_repr_pr, train_pmh), axis=1)
        test_repr_pr = np.concatenate((test_repr_pr, test_pmh), axis=1)

    if 'problist' in modalities_selected:
        train_pblist = proc_modality_dict_train['problist']
        test_pblist = proc_modality_dict_test['problist']
        train_repr_pr = np.concatenate((train_repr_pr, train_pblist), axis=1)
        test_repr_pr = np.concatenate((test_repr_pr, test_pblist), axis=1)

    # breakpoint()

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
        # fit_clf = eval_protocols.fit_xgbt_cv
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'


    train_repr = train_repr_pr
    test_repr = test_repr_pr

    # breakpoint()
    clf = fit_clf(train_repr, train_labels, seed=randomSeed, outcome=outcomeGiven)
    if False:
        # clf = fit_clf(train_repr[:-3000, :], train_labels[:-3000], seed=randomSeed, outcome=outcomeGiven)

        # calibration function
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_clf = CalibratedClassifierCV(clf, cv = 'prefit', method="isotonic")
        calibrated_clf.fit(train_repr[-3000:, :], train_labels[-3000:])

        acc = calibrated_clf.score(test_repr, test_labels)

    acc = clf.score(test_repr, test_labels)

    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = clf.predict_proba(test_repr) # this is from the non-calibrated classifier
        y_score_tr = clf.predict_proba(train_repr) # this is from the non-calibrated classifier
        # y_score = calibrated_clf.predict_proba(test_repr) # this is from the calibrated classifier
        # y_score_tr = calibrated_clf.predict_proba(train_repr) # this is from the calibrated classifier
        auprc = average_precision_score(test_labels, y_score[:, 1])
        auroc = roc_auc_score(test_labels, y_score[:, 1])
    else:
        y_score = clf.decision_function(test_repr)
        y_score_tr = clf.decision_function(train_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)

    return y_score, y_score_tr, {'acc': acc, 'auprc': auprc, 'auroc': auroc}