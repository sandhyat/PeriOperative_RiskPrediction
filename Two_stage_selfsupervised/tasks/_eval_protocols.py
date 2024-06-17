import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def fit_svm(features, y, seed = 0, MAX_SAMPLES=10000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

def fit_lr(features, y, seed = 0, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=seed,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe


def fit_xgbt(features, y, seed = 0, outcome=None, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    if outcome !=None:
        best_hp_dict = {'icu': [346, 8, 0.0091, 0.0036, 0.0054], 'mortality': [479, 7, 0.0071, 0.0066, 0.0002],
                        'aki1': [457, 8, 0.0087, 0.0024, 0.0035],
                        'aki2': [492, 4, 0.0093, 0.0012, 0.0017], 'aki3': [432, 7, 0.0091, 0.0009, 0.0008],
                        'dvt_pe': [403, 6, 0.0082, 0.0032, 0.0011],
                        'AF': [338, 6, 0.0092, 0.0009, 0.0015], 'cardiac': [409, 5, 0.0093, 0.0003, 0.0011],
                        'CVA': [322, 6, 0.0091, 0.0015, 0.0013],
                        'DVT': [293, 7, 0.0095, 0.0017, 0.004], 'GI': [483, 8, 0.0072, 0.0011, 0.0054],
                        'PNA': [405, 8, 0.0099, 0.0002, 0.0002],
                        'UTI': [498, 7, 0.0077, 0.0001, 0.0026], 'VTE': [435, 7, 0.0094, 0.003, 0.0006],
                        'postop_trop_crit': [339, 7, 0.0071, 0.0045, 0.0011],
                        'postop_trop_high': [494, 7, 0.0061, 0.0036, 0.0018],
                        'post_dialysis': [403, 6, 0.0072, 0.0099, 0.0093],
                        'postop_del': [301, 5, 0.0092, 0.0012, 0.0017],
                        'severe_present_1': [499, 4, 0.0081, 0.0007, 0.0047]}
        if outcome not in best_hp_dict.keys():
            xgb_model = XGBClassifier(random_state=seed)
            clf = GridSearchCV(
                xgb_model,
                {"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}, cv=3,
                verbose=1,
            )
            clf.fit(features, y)
            return clf.best_estimator_
        else:
            pipe = make_pipeline(
            StandardScaler(),
                XGBClassifier(n_estimators=best_hp_dict[outcome][0], max_depth=best_hp_dict[outcome][1],
                              learning_rate=best_hp_dict[outcome][2], reg_lambda=best_hp_dict[outcome][4],
                              reg_alpha=best_hp_dict[outcome][3], random_state=seed)
            )
    else:
        pipe = make_pipeline(
        StandardScaler(),
            XGBClassifier(n_estimators=300, random_state=seed)
        )

    pipe.fit(features, y)
    return pipe

def fit_xgbt_cv(features, y,  seed = 0, outcome=None, MAX_SAMPLES=100000):
    xgb_model = XGBClassifier(random_state=seed)
    clf = GridSearchCV(
        xgb_model,
        {"max_depth": [4, 6], "n_estimators": [50, 100, 200], "learning_rate":[0.01, 0.1,1.0]}, cv=3,
        verbose=1,
    )
    clf.fit(features,y)
    # breakpoint()
    return clf.best_estimator_

def fit_knn(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr
