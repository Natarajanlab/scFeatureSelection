
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging import config, debug

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV, ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler, StandardScaler


from sklearn.preprocessing import label_binarize


from sklearn.model_selection import cross_val_score


from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar
from scipy.integrate import quad

import sys
import os 
import yaml
from itertools import chain
from tqdm import tqdm
from joblib import Parallel, delayed

import threading
import time

# optional libs
try:
    from boruta import BorutaPy
except Exception:
    BorutaPy = None

try:
    from skrebate import ReliefF
except Exception:
    ReliefF = None

try:
    from hsic_lasso import HSICLasso
except Exception:
    HSICLasso = None

print("Python version: {}". format(sys.version))
print("pandas version: {}". format(pd.__version__))
print("NumPy version: {}". format(np.__version__))

print('-'*25)

def set_nature_style():
    plt.rcParams.update({
         # --- Figure & Font ---
    'figure.figsize': (6.8, 4.5),       # ~85mm x 55mm = 1 column in Nature
    'figure.dpi': 300,                  # High-resolution display
    'savefig.dpi': 600,                 # High-resolution export
    'font.size': 7,                     # Base font size (Nature uses small fonts)
    'font.family': 'sans-serif',       
    'font.sans-serif': ['Arial'],     
    'axes.titlesize': 7,               # Subplot titles
    'axes.labelsize': 7,               # Axis labels
    'xtick.labelsize': 6,              # Tick labels
    'ytick.labelsize': 6,
    'legend.fontsize': 6,              # Legend text
    'legend.frameon': True,         

    # --- Axes Style ---
    'axes.spines.top': False,          # Minimal style (Nature style)
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 2,
    'ytick.major.size': 2,

    # --- Save options ---
    'savefig.bbox': 'tight',           # Trim whitespace
    'savefig.transparent': True,       # Transparent background
    })

# plt.rcParams.update({
#     'font.size': 14,            # Base font size
#     'axes.titlesize': 16,       # Title of each subplot
#     'axes.labelsize': 16,       # X/Y axis labels
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14
# })

def load_config(path="config.yaml", validate=True, confirm=False):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if validate:
        _validate_config(config)

    print("loaded config:\n")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if confirm:    
        resp = input("\ncontinue with these settings? [y/n]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborting.")
            sys.exit(0)

    return config

def _validate_config(cfg):
    assert "dataset" in cfg and isinstance(cfg["dataset"], str), "dataset must be a string"
    assert "target" in cfg and isinstance(cfg["target"], str), "target must be a string"
    assert cfg.get("model_name") in {"Logistic Regression", "LR", "SVM", "Decision Tree", "DT", "Random Forest", "RF", "Neural Network", "NN", "MLP", "XGBoost", "XGB", "GBM", "HistGB", "Ridge", "SGD"}, "invalid model_name"
    assert isinstance(cfg.get("opt_model", 0), int), "opt_model must be int"
    assert isinstance(cfg.get("random_state", 42), int), "random_state must be int"
    assert isinstance(cfg.get("num_runs", 1), int) and cfg["num_runs"] > 0, "num_runs must be positive int"
    assert isinstance(cfg.get("remove_cols", False), bool), "remove_cols must be bool"
    assert cfg.get("plot_acc", 0) in {0,1}, "plot_acc must be 0 or 1"
    assert cfg.get("plot_auc", 1) in {0,1}, "plot_auc must be 0 or 1"

config = load_config()

np.random.seed(config['random_state'])

if config['train_test_sep']:
    save_fig_path = f"./Plots/{config['dataset']}/"+"test_"+config['test_dataset']+"/"+config['model_name']+"_train_test_sep/"
else:
    save_fig_path = "./Plots/"+config['dataset']+"/"+"/"+config['model_name']+"/"

# mapping of flags to folder suffixes
suffix_map = {
    'use_only_fs_list': 'fs_list',
    'use_only_hvgs': 'hvgs',
    'use_only_non_hvgs': 'non_hvgs',
    'use_only_mt_genes': 'mt_genes',
    'use_only_hk_genes': 'hk_genes',
    'use_union_of_cluster_genes': 'union_cluster_genes',
    'use_uc_plus': 'uc_plus',
}

# collect all active flags (order-independent)
active = [(k, v) for k, v in suffix_map.items() if config.get(k, 0) == 1]

if len(active) > 1:
    raise ValueError(
        f"multiple gene-selection flags active: {[k for k, _ in active]}; "
        "exactly one must be set"
    )

# exactly one active flag
try:
    active_flag, active_suffix = active[0]
except IndexError:
    active_flag, active_suffix = None, None


# combine
save_fig_path = save_fig_path + (active_suffix + "/" if active_suffix else "")

if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)  

print(f'\n\n All plots and CSVs will be saved in the directory: {save_fig_path}')

# set_nature_style() # Uncomment to set Nature style for plots

# plt.rcdefaults() # Uncomment to reset the plot parameters 

def assert_fraction(value):
    assert isinstance(value, (float, int)), "value must be a number"
    assert 0 < value < 1, "value must be a fraction between 0 and 1 (exclusive)"
    # print("value is valid")


def check_scaling(X):
    if isinstance(X, pd.DataFrame):
        X = X.values

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    medians = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)

    # print("For StandardScaler:")
    # print(f'mean per feature (should be ~0) :\n{means[:5]}')
    # print(f'std per feature (should be ~1) :\n{stds[:5]}')
    print('-'*25)
    print("For RobustScaler:")
    print(f'median per feature (should be ~0) :\n{medians[:5]}')
    print(f'iqr per feature (should be ~1) :\n{iqr[:5]}')
    print('-'*25)

def sparsity_summary_df(df, label_col):
    # extract X as numpy array
    X = df.drop(columns=[label_col]).values
    n_samples, n_features = X.shape

    # global sparsity
    global_zero_frac = (X == 0).sum() / X.size

    # per-feature sparsity
    feat_zero_frac = (X == 0).sum(axis=0) / n_samples
    median_feat_zero = np.median(feat_zero_frac)

    # per-sample sparsity
    samp_zero_frac = (X == 0).sum(axis=1) / n_features
    median_samp_zero = np.median(samp_zero_frac)

    return pd.Series({
        "n_samples": n_samples,
        "n_features": n_features,
        "global_zero_frac": global_zero_frac,
        "median_feature_zero_frac": median_feat_zero,
        "median_sample_zero_frac": median_samp_zero
    })

def _scale_if_needed(X):
    return StandardScaler().fit_transform(X)

def lasso_fs_fast(X, y, top_k=None, outdir="fs_prov", random_state=0):
    """
    fast lasso-like feature selector.

    strategy:
    - if p > 10000 and n < 1000 -> use sgd (fast, stochastic, scalable)
    - else use logisticregressioncv with a tiny Cs grid + loose tol

    returns: np.array of selected column indices (unique, sorted)
    """

    n_samples, n_features = X.shape

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    use_sgd = (n_features > 10000) and (n_samples < 2000)

    if use_sgd:
        # fast approximate l1 via sgd
        # note: sgd's 'l1' is not identical to full L1-CVX but is a pragmatic fast proxy
        clf = SGDClassifier(
            loss="log_loss",      # probabilistic logistic
            penalty="l1",
            alpha=1e-4,           # regularization strength (tunable)
            max_iter=2000,
            tol=1e-3,
            learning_rate="optimal",
            average=True,         # stabilizes weights
            random_state=random_state,
            verbose=0
        )
        clf.fit(Xs, y)
        coef = np.abs(clf.coef_).ravel()
        method = "sgd_l1"
        extra = {"alpha": clf.alpha, "max_iter": clf.max_iter, "tol": clf.tol}
    else:
        # compact logistic l1 cv with very small grid and loose tol for speed

        # Update to OneVsRestClassifier(LogisticRegressionCV(..)) if multi-class issues arise

        lr = LogisticRegressionCV(
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="accuracy",
            max_iter=2000,        # small but usually enough with scaled X
            tol=1e-3,
            Cs=[0.01, 0.1, 1.0],  # tiny grid -> faster
            n_jobs=1,
            random_state=random_state,
            # multi_class="ovr",
            refit=True
        )
        lr.fit(Xs, y)
        coef = np.abs(lr.coef_).ravel()
        method = "logistic_l1_cv"
        extra = {"Cs": [0.01,0.1,1.0], "max_iter": lr.max_iter, "n_iter_": getattr(lr, "n_iter_", None)}

    # pick features
    if top_k is None:
        sel = np.where(coef > 1e-6)[0]
        # if lasso ended up selecting nothing (possible in very sparse problems), fallback to top-100
        if sel.size == 0:
            kfb = min(100, n_features)
            order = np.argsort(-coef)
            sel = order[:kfb]
    else:
        order = np.argsort(-coef)
        ksel = min(top_k, n_features)
        sel = order[:ksel]

    sel = np.unique(sel)    # stable ordering

    return sel

def lasso_fs(X, y, top_k=None):
    # logistic l1 with cv. returns indices with largest abs(coef) if top_k provided, else nonzero indices
    Xs = _scale_if_needed(X)
    lr = LogisticRegressionCV(cv=5, penalty="l1", solver="saga", scoring="accuracy",
                              max_iter=5000, Cs=10, n_jobs=1, random_state=config['random_state'])
    lr.fit(Xs, y)
    coef = np.abs(lr.coef_).ravel()
    if top_k is None:
        sel = np.where(coef > 1e-6)[0]
    else:
        sel = np.argsort(-coef)[:top_k]
    return np.unique(sel)

def elasticnet_fs(X, y, top_k=100):

    # use ElasticNetCV on standardize; approximate by using LogisticRegressionCV with l1 then l2? simpler: use linear elastic net on stats
    # here we use sklearn's ElasticNetCV on standardized X and treat y as numeric (works for binary); fallback to lasso if multiclass
    Xs = _scale_if_needed(X)
    classes = np.unique(y)
    if len(classes) == 2:
        ynum = (y == classes[1]).astype(float)
        en = ElasticNetCV(cv=5, n_alphas=50, random_state=config['random_state'], max_iter=5000)
        en.fit(Xs, ynum)
        coef = np.abs(en.coef_)
        sel = np.argsort(-coef)[:top_k]
        return np.unique(sel)
    else:
        # multiclass fallback: use l1 logistic and return top_k by coef norm
        lr = LogisticRegressionCV(cv=5, penalty="l1", solver="saga", scoring="accuracy", max_iter=5000, Cs=10, n_jobs=1, random_state=config['random_state'])
        lr.fit(Xs, y)
        coef = np.linalg.norm(lr.coef_, axis=0)
        sel = np.argsort(-coef)[:top_k]
        return np.unique(sel)

def rf_importance_fs(X, y, top_k=100):
    rf = RandomForestClassifier(n_estimators=500, random_state=config['random_state'], n_jobs=1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    sel = np.argsort(-imp)[:top_k]
    return np.unique(sel)

def boruta_fs(X, y, top_k=None):
    if BorutaPy is None:
        raise ImportError("boruta_py not installed. pip install boruta-py")
    rf = RandomForestClassifier(n_jobs=1, n_estimators=500, random_state=config['random_state'])
    bor = BorutaPy(rf, n_estimators="auto", random_state=config['random_state'], verbose=0)
    bor.fit(X, y)
    sel = np.where(bor.support_)[0]
    if top_k is not None and len(sel) > top_k:
        # if boruta returns more than top_k, pick top_k by rf importances on selected features
        rf2 = RandomForestClassifier(n_estimators=200, random_state=config['random_state'], n_jobs=1)
        rf2.fit(X[:, sel], y)
        imp = rf2.feature_importances_
        order = np.argsort(-imp)[:top_k]
        sel = sel[order]
    return np.unique(sel)

def relieff_fs(X, y, top_k=100):
    if ReliefF is None:
        raise ImportError("skrebate not installed. pip install skrebate")
    Xs = _scale_if_needed(X)
    r = ReliefF(n_neighbors=100, n_features_to_select=top_k)
    r.fit(Xs, y)
    ranks = np.argsort(-r.feature_importances_)[:top_k]
    return np.unique(ranks)

def hsic_lasso_fs(X, y, top_k=100):
    if HSICLasso is None:
        raise ImportError("hsic_lasso package not installed. pip install hsic_lasso")
    # hsic_lasso expects features × samples maybe; follow package docs
    # convert to float
    Xf = X.astype(float)
    model = HSICLasso()
    model.input(Xf, y)
    model.regression()
    selected = model.get_index()  # returns list of selected indices (1-based maybe)
    # convert to zero-based
    sel = np.array(selected) - 1
    if top_k is not None:
        return sel[:top_k]
    return sel

# mapping
FS_METHODS = {
    "lasso": lasso_fs_fast,
    "elasticnet": elasticnet_fs,
    # "relieff": relieff_fs,
    # "hsic_lasso": hsic_lasso_fs,
    "rf_importance": rf_importance_fs,
    # "boruta": boruta_fs
}

# --- evaluation driver -- nested cv + compare top-k ---
def evaluate_topk_methods(df, label_col=config['target'], k_list=[10,50,100,200], outer_folds=5, inner_folds=3, n_jobs=8, classifier=None):
    """
    df: pandas dataframe, samples x features + label_col
    returns: summary dataframe with columns: method, k, median_acc, accs(list), n_selected
    """
    X = df.drop(columns=[label_col]).values
    feature_names = df.drop(columns=[label_col]).columns.to_numpy() 
    y = df[label_col].values

    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=500, random_state=config['random_state'], n_jobs=1)

    skf_outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=config['random_state'])
    results = []

    # outer loop: do selection inside train only
    outer_splits = list(skf_outer.split(X, y))

    def eval_method_on_k(method_name, k):
        accs = []
        aucs = []
        selected_counts = []
        for train_idx, test_idx in outer_splits:
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            # avoid leakage: train FS on Xtr,ytr
            fs_func = FS_METHODS[method_name]

            # some methods require top_k param, some interpret top_k as None
            try:
                sel = fs_func(Xtr, ytr, top_k=k)
            except TypeError:
                # method may not accept top_k, try without
                print(f"Note: method {method_name} does not accept top_k param, calling without it.")
                sel = fs_func(Xtr, ytr)

            if len(sel) == 0:
                print("**"*25)
                print(f'\t\t\t Warning: No features selected for method {method_name} with k={k}. Falling back to rf importance.')
                print("**"*25)
                # nothing selected: fallback to top-k by rf importance on train
                sel = rf_importance_fs(Xtr, ytr, top_k=k)

            selected_counts.append(len(sel))

            selected_feature_names = feature_names[sel]

            train_y_sel = ytr
            train_X_sel = Xtr[:, sel]

            test_y_sel = yte
            test_X_sel = Xte[:, sel]

            filtered_feature_train_df = pd.DataFrame(train_X_sel, columns=selected_feature_names)
            filtered_feature_train_df[label_col] = train_y_sel

            filtered_feature_test_df = pd.DataFrame(test_X_sel, columns=selected_feature_names)
            filtered_feature_test_df[label_col] = test_y_sel

            if debug:
                print("train shape:", filtered_feature_train_df.shape, "test shape:", filtered_feature_test_df.shape)
                print("n selected features:", len(selected_feature_names))

            acc_full_feature, roc_auc_ovo_test, roc_auc_ovr, pr_auc = check_model_acc_full_feature(df = filtered_feature_train_df, test_df=filtered_feature_test_df, train_test_sep=1, debug=0)
            print(f"Method: {method_name}, Full feature accuracy: {acc_full_feature}, AUC-OVO: {roc_auc_ovo_test}, AUC-OVR: {roc_auc_ovr}, PR-AUC: {pr_auc}")

            accs.append(acc_full_feature)
            aucs.append(roc_auc_ovr) # using AUC-OVR as the main AUC metric for simplicity

        return {
            "method": method_name,
            "k": k,
            "mean_acc": float(np.mean(accs)),
            "median_acc": float(np.median(accs)),
            "std_acc": float(np.std(accs)),
            "mean_auc": float(np.mean(aucs)),
            "median_auc": float(np.median(aucs)),
            "std_auc": float(np.std(aucs)),
            "accs": accs,
            "aucs": aucs,
            "n_selected_median": int(np.median(selected_counts))
        }

    # parallel run over methods & ks
    tasks = [(m, k) for m in FS_METHODS.keys() for k in k_list]
    out = Parallel(n_jobs=n_jobs)(delayed(lambda mm, kk: eval_method_on_k(mm, kk))(m, k) for (m, k) in tasks)

    # save & produce df
    rows = []
    for r in out:
        rows.append({
            "method": r["method"],
            "k": r["k"],
            "mean_acc": r["mean_acc"],
            "median_acc": r["median_acc"],
            "std_acc": r["std_acc"],
            "mean_auc": r["mean_auc"],
            "median_auc": r["median_auc"],
            "std_auc": r["std_auc"],
            "n_selected_median": r["n_selected_median"],
            "accs": r["accs"]
        })
    summary_df = pd.DataFrame(rows)
    summary_df[["method","k","mean_acc","median_acc","std_acc","mean_auc","median_auc","std_auc","n_selected_median"]].to_csv(os.path.join(save_fig_path, "fs_topk_summary.csv"), index=False)
    
    return summary_df

def get_optimized_rf_model(X_train, y_train, X_test, y_test, random_state=config['random_state']):
    # RandomForestClassifier(n_estimators=10, random_state=random_state),


    opt = RandomizedSearchCV(
        RandomForestClassifier(),
        {
            'n_estimators':[20,50, 100, 200, 500],
            'criterion':['log_loss', 'gini','entropy']
        },
        n_iter=100,
        cv=5,
        error_score='raise',
        n_jobs=-1,
        verbose=True, 
        scoring='accuracy'
    )

    opt.fit(X_train, y_train)


    print("Best model: %s" % opt.best_estimator_)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    return opt.best_estimator_

def get_model(model_name=config['model_name'], dataset=config['dataset'], random_state=config['random_state'], opt_model=config['opt_model']):
  
  if not(opt_model):
      model_dict = {
            'Logistic Regression':lambda: LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=random_state),
            'LR':lambda: LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=random_state),

            'SVM':lambda: LinearSVC(dual=True, class_weight='balanced', random_state=random_state),

            'Decision Tree':lambda: DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
            'DT':lambda: DecisionTreeClassifier(class_weight='balanced', random_state=random_state),

            'Random Forest': lambda: RandomForestClassifier(class_weight='balanced', random_state=random_state),
            'RF':lambda: RandomForestClassifier(class_weight='balanced', random_state=random_state),

            # for large datasets with many features and samples, use n_jobs (depending on the number of cores) to speed up training
            # 'Random Forest': lambda: RandomForestClassifier(n_estimators=200,
            #                                                 max_depth=20,
            #                                                 class_weight="balanced",
            #                                                 random_state=random_state,
            #                                                 n_jobs=56
            #                                                 ),
            # 'RF': lambda: RandomForestClassifier(n_estimators=200,
            #                                                 max_depth=20,
            #                                                 class_weight="balanced",
            #                                                 random_state=random_state,
            #                                                 n_jobs=56
            #                                                 ),
            'Extra Trees': lambda: ExtraTreesClassifier(n_estimators=300,
                                                        max_depth=20,
                                                        class_weight="balanced",
                                                        n_jobs=56,
                                                        random_state=random_state,
                                                        ),
            'ET': lambda: ExtraTreesClassifier(n_estimators=300,
                                                        max_depth=20,
                                                        class_weight="balanced",
                                                        n_jobs=56,
                                                        random_state=random_state,
                                                        ),

            "Neural Network":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
            "NN":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
            "MLP":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),

            "XGBoost": lambda: XGBClassifier(objective='binary:logistic', random_state=random_state),
            "XGB": lambda: XGBClassifier(objective='binary:logistic', random_state=random_state),

            'GBM': lambda: GradientBoostingClassifier(random_state=random_state),

            'HistGB': lambda: HistGradientBoostingClassifier(random_state=random_state), 

            'Ridge': lambda: RidgeClassifier(class_weight='balanced', random_state=random_state),

            'SGD': lambda: SGDClassifier(class_weight='balanced', loss="log_loss", random_state=random_state),
            }
  elif opt_model:
      if dataset == "GSE4115":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 4, min_samples_split = 2, n_estimators = 300, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 4, min_samples_split = 2, n_estimators = 300, random_state=random_state),
          }
      elif dataset == "ALL_AML":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 100, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 100, random_state=random_state),
            'GBM': lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=random_state)
          }
      elif dataset =="Colon":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 200, random_state=random_state),
          }
      elif dataset == "Arcene":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 200, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 200, random_state=random_state)
          }
      elif dataset == "Data_Bischoff2021_Lung":
            model_dict = {
                'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state),
                'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state)
            }

  model_name = model_name.strip()
  
  if 'model_dict' in locals():
    if model_name not in model_dict:
            print("Model name should be one of", list(model_dict.keys()))
            print("Please check the entered model name for spelling errors.")
            sys.exit()
    else:
        model = model_dict[model_name]()
        return model
  else:
      print("No model dictionary found for the current context.")
      sys.exit()

def get_scores_from_model(model, X):
    """Get usable scores for AUC, handling tricky classifiers."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba if proba.shape[1] > 2 else proba[:, 1]
    elif isinstance(model, LinearSVC) and hasattr(model, "_predict_proba_lr"):
        # Private API, but sometimes needed
        return model._predict_proba_lr(X)[:, 1]
    else:
        raise ValueError(
            f"Model {type(model).__name__} has no decision_function or predict_proba. "
            "Consider using a different loss/solver or wrapping in CalibratedClassifierCV."
        )

def get_auc_scores(model, X_test, y_test, average="macro"):
    """
    Compute per-class and average ROC AUC + PR AUC.
    Works with LinearSVC, RidgeClassifier, SGDClassifier, LogisticRegression, etc.
    """
    y_scores = get_scores_from_model(model, X_test)
    classes = np.unique(y_test)
    n_classes = len(classes)

    # --- Per-class ---
    if n_classes == 2:
        if y_scores.ndim > 1 and y_scores.shape[1] == 1:
            y_scores = y_scores.ravel()

        auc_val = roc_auc_score(y_test, y_scores)
        pr_val = average_precision_score(y_test, y_scores)

        per_class_roc = {int(classes[1]): auc_val}
        per_class_pr = {int(classes[1]): pr_val}

        avg_results = {"roc_auc_ovr": auc_val, "roc_auc_ovo": auc_val, "pr_auc": pr_val}

    else:
        if y_scores.ndim == 1:
            raise ValueError("Expected 2D scores for multiclass case.")

        y_test_bin = label_binarize(y_test, classes=classes)
        per_class_roc = {
            int(classes[i]): roc_auc_score(y_test_bin[:, i], y_scores[:, i])
            for i in range(n_classes)
        }
        per_class_pr = {
            int(classes[i]): average_precision_score(y_test_bin[:, i], y_scores[:, i])
            for i in range(n_classes)
        }

        avg_results = {
            "roc_auc_ovr": roc_auc_score(y_test, y_scores, multi_class="ovr", average=average),
            "roc_auc_ovo": roc_auc_score(y_test, y_scores, multi_class="ovo", average=average),
            "pr_auc": average_precision_score(y_test_bin, y_scores, average=average),
        }

    return {
        "per_class": {"roc_auc": per_class_roc, "pr_auc": per_class_pr},
        "average": avg_results,
    }

def check_model_acc_full_feature(df, debug=config['debug'], average='macro', opt_model=config['opt_model'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    ''' 
        Parameters:
                df (pandas DataFrame): A dataframe containing features and a target (in the last column)
                debug (bool): Whether to print debug information
                average (str): The averaging method to use for multi-class classification. Defaults to 'macro'.
                opt_model (bool): Whether to use an optimized model. Defaults to False. 
                train_test_sep (bool): Whether to use a separate test set for evaluation. Defaults to False.
                test_df (pandas DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                if train_test_sep is True:
                    Accuracy, ROC AUC (one-vs-one), ROC AUC (one-vs-rest), and PR AUC on the test set
                else:
                    Accuracy (float): Average accuracy for the 3/5-fold cross-validation  
                    ROC AUC (one-vs-one) (float): Average ROC AUC score using the one-vs-one strategy across folds
                    ROC AUC (one-vs-rest) (float): Average ROC AUC score using the one-vs-rest strategy across folds
                    PR AUC (float): Average Precision-Recall AUC score across folds

    '''   
    target = config['target']
    model_name=config['model_name']
    dataset=config['dataset']

    if debug:
        print(f"Running the experiment with the following parameters:\n"
            f"Model: {model_name}, Average: {average}, \n")
        if train_test_sep:
            print(f"Train dataset: {dataset}, samples: {df.shape[0]}, features: {df.shape[1]-1}")
            print(f"Test dataset: {config['test_dataset']}, samples: {test_df.shape[0]}, features: {test_df.shape[1]-1}")
        else:
            print(f"Dataset: {dataset}, samples: {df.shape[0]}, features: {df.shape[1]-1}")

    if not(train_test_sep):
        
        y = df[target]
        x = df.drop(columns=[target])

    elif train_test_sep:
        if debug:
            print(f'Using a separate test set for {dataset} dataset.')
        y = df[target]
        x = df.drop(columns=[target])

        y_test_sep = test_df[target]
        X_test_sep = test_df.drop(columns=[target])

    n_classes = len(np.unique(y))

    acc_list = []
    roc_auc_list_ovo = []
    roc_auc_list_ovr = []
    pr_auc_list = []
    b_acc_list = []
    f1_list = []
    prec_list = []
    rec_list = []
    mcc_list = []

    if train_test_sep == 0: 
        # If train and test are split of the same dataset, we do 3/5-fold cross-validation depending on the number of samples. 
        # For cases, where train and test sets are seprate, we use them as such without CV.
        if x.shape[0] > 100:
            if debug:
                print(f"Using StratifiedKFold with 5 folds for {dataset} dataset")
            fold_n = 5
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['random_state'])
        else:
            if debug:
                print(f"Using StratifiedKFold with 3 folds for {dataset} dataset")
            fold_n = 3
            skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=config['random_state'])

        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            if config['shuffle_cols']:
                if debug:
                    print("Shuffling columns")
                selected_columns = np.random.choice(x.columns, size=int(1* x.shape[1]), replace=False)
                x = x[selected_columns]

                if debug:
                    print(f"Fold {fold+1}: Selected {len(selected_columns)} columns for training and testing.")

                assert len(selected_columns) == df.shape[1]-1
                assert selected_columns.dtype == object  # should be column names
                assert not x[selected_columns].isnull().any().any()  # sanity check

            X_train_raw, X_test_raw = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            
            tree_models = ["RF", "DT", "XGB", "LGB", "GBM","HistGB", "Random Forest", "Decision Tree", "XGBoost", "Gradient Boosting", "HistGradientBoosting Classifier", "ET", "Extra Trees"]
            if config['model_name'] not in tree_models:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
                if debug:
                    print("For training data:")
                    check_scaling(X_train)
                    print("For test data:")
                    check_scaling(X_test)

            else:
                X_train = X_train_raw
                X_test = X_test_raw    
            

            if debug:
                print(f"fold {fold}: train={len(train_idx)}, test={len(test_idx)}")    
                print("\n\n******** Train ************\n\n")
                print(X_train.shape, y_train.shape)
                print("\n\n********* Test ***********\n\n")
                print(X_test.shape, y_test.shape)
                print("\n\n**************************\n\n")
            
            # fit model
            model = get_model(model_name=model_name, dataset=dataset, random_state=config['random_state'], opt_model=opt_model)
            if debug:
                print(f"Fitting model: {model_name} on fold {fold+1}")
            model.fit(X_train, y_train)
            
            # make predictions
            predictions = model.predict(X_test)

            # Get AUC scores
            auc_scores_results = get_auc_scores(model, X_test, y_test)
            if debug:
                print(auc_scores_results)

            roc_auc_ovo = auc_scores_results['average']['roc_auc_ovo']
            roc_auc_ovr = auc_scores_results['average']['roc_auc_ovr']
            pr_auc = auc_scores_results['average']['pr_auc']

            roc_auc_list_ovo.append(roc_auc_ovo)    
            roc_auc_list_ovr.append(roc_auc_ovr)
            pr_auc_list.append(pr_auc)
            
            if debug:
                print(classification_report(y_test, predictions, zero_division=0))
            
        
            accuracy = accuracy_score(y_test, predictions)
            acc_list.append(accuracy)

            b_accuracy = balanced_accuracy_score(y_test, predictions)
            b_acc_list.append(b_accuracy)

            f1score = f1_score(y_test, predictions, average='weighted', zero_division=0)
            f1_list.append(f1score)

            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            prec_list.append(precision)

            recall = recall_score(y_test, predictions, average='weighted')
            rec_list.append(recall)

            mcc = matthews_corrcoef(y_test, predictions)
            mcc_list.append(mcc)

            if debug:
                print(f'Accuracy: {accuracy}', end='\t')
                print(f'ROC AUC: {roc_auc_ovo}', end='\t')
                print(f'Balanced Accuracy: {b_accuracy}', end='\t')
                print(f'Macro F1 score: {f1score}', end='\t')
                print(f'Macro Precision: {precision}', end='\t')
                print(f'Macro Recall: {recall}')
                print(f'MCC: {mcc}')
                print("--"* 25)


        avg_acc = np.average(acc_list)
        avg_roc_auc_ovo = np.average(roc_auc_list_ovo)
        avg_roc_auc_ovr = np.average(roc_auc_list_ovr)
        avg_auprc = np.average(pr_auc_list)
    
        print("--"* 25)
        print(f"\n\n{fold_n}-fold cross-validation results:")
        print("Accuracy: ", avg_acc, end='\t')
        print("ROC AUC OVO: ", avg_roc_auc_ovo, end='\t')
        print("ROC AUC OVR: ", avg_roc_auc_ovr, end='\t')
        print("AUPRC: ", avg_auprc, end='\t')

        print("Balanced Accuracy: ", np.average(b_acc_list), end='\t')
        print("Macro F1 score: ", np.average(f1_list), end='\t')
        print("Macro Precision: ", np.average(prec_list), end='\t')
        print("Macro Recall: ", np.average(rec_list))
        print("MCC: ", np.average(mcc_list))
        print("--"* 25)
        
        return avg_acc, avg_roc_auc_ovo, avg_roc_auc_ovr, avg_auprc

    if train_test_sep:

        if debug:
            print(f'Train and test sets are different, evaluating the model on a separate test set with {x.shape[1]} features.')

        model_sep = get_model(model_name=model_name, dataset=dataset, random_state=config['random_state'], opt_model=opt_model)
        model_sep.fit(x, y)

        predictions_test = model_sep.predict(X_test_sep)

        if debug:
            print(classification_report(y_test_sep, predictions_test))
        accuracy_test = accuracy_score(y_test_sep, predictions_test)
        
        auc_scores_results = get_auc_scores(model_sep, X_test_sep, y_test_sep)
        
        if debug:
            print(auc_scores_results)
        
        # roc_auc = auc_scores_results['average']['roc_auc_ovo']

        roc_auc_ovo_test = auc_scores_results['average']['roc_auc_ovo']
        roc_auc_ovr_test = auc_scores_results['average']['roc_auc_ovr']
        pr_auc_test = auc_scores_results['average']['pr_auc']

        if debug:
            print("Accuracy on separate test set: ", accuracy_test, end='\t')
            print("ROC AUC OVO on separate test set: ", roc_auc_ovo_test, end='\t')
            print("ROC AUC OVR on separate test set: ", roc_auc_ovr_test, end='\t')
            print("PR AUC on separate test set: ", pr_auc_test)

        return accuracy_test, roc_auc_ovo_test, roc_auc_ovr_test, pr_auc_test

def run_random_train_test_split(df, test_size=0.2):
    '''
        Parameters:
                df (pandas.DataFrame): Original unmodified dataframe containing features and a target (in the last column)
                test_size (float): Proportion of the dataset to include in the test split
                random_state (int): Random seed for reproducibility

        Returns:
                train_df (pandas.DataFrame): Training dataframe
                test_df (pandas.DataFrame): Testing dataframe
    '''

    target=config['target']
    num_runs = config['num_runs']

    cm_norm_values = []
    acc_list = []
    b_acc_list = []
    roc_auc_ovo_list = []
    roc_auc_ovr_list = []    
    pr_auc_list = []
    f1score_list = []
    prec_list = []
    rec_list = []
    mcc_list = []

    for run in range(num_runs):
        print("\n")
        print("-"*50)
        print(f"Random train-test split run {run+1}/{num_runs}", flush=True)
        
        random_state = config['random_state'] + run    # Different random state for each run
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]

        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
        class_labels = np.unique(y_test)

        clf = get_model(model_name=config['model_name'], opt_model=config['opt_model'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm_norm = confusion_matrix(
                y_test,
                y_pred,
                labels=class_labels,
                normalize="true"        # row-wise %
            )

        if config['debug']:
            print("Confusion Matrix (normalized):")
            print(cm_norm)

        cm_norm_values.append(cm_norm)

        auc_scores_results = get_auc_scores(clf, X_test, y_test)
        roc_auc_ovo = auc_scores_results['average']['roc_auc_ovo']
        roc_auc_ovr = auc_scores_results['average']['roc_auc_ovr']
        pr_auc = auc_scores_results['average']['pr_auc']


        roc_auc_ovo_list.append(roc_auc_ovo)
        roc_auc_ovr_list.append(roc_auc_ovr)
        pr_auc_list.append(pr_auc)

        accuracy = accuracy_score(y_test, y_pred)
        acc_list.append(accuracy)

        b_acc = balanced_accuracy_score(y_test, y_pred)
        b_acc_list.append(b_acc)
        
        f1score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1score_list.append(f1score)

        prec_score = precision_score(y_test, y_pred, average='macro', zero_division=0)
        prec_list.append(prec_score)

        rec_score = recall_score(y_test, y_pred, average='macro')
        rec_list.append(rec_score)

        mcc = matthews_corrcoef(y_test, y_pred)
        mcc_list.append(mcc)

    print("\n")
    print("-"*50)
    # print(f"Saving confusion matrix mean ± std over {num_runs} random train-test split runs", flush=True)

    # Convert list of confusion matrices to a 3D numpy array for easier computation
    # Get mean and std deviation across runs
    cm_array = np.array(cm_norm_values)

    cm_mean = cm_array.mean(axis=0)
    cm_std  = cm_array.std(axis=0)

    # build annotation strings: mean ± std
    #  annot = np.full(cm_mean.shape, "", dtype=object)
    # for i in range(cm_mean.shape[0]):
    # annot[i, i] = f"{cm_mean[i,i]:.2f}\n± {cm_std[i,i]:.2f}"
    annot = np.empty_like(cm_mean, dtype=object)
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            annot[i, j] = f"{cm_mean[i,j]:.2f}\n ± {cm_std[i,j]:.2f}"

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm_mean,
        annot=annot,
        fmt="",
        # square=True,
        cmap = plt.cm.viridis,
        vmin=0, vmax=1,
        linewidths=0.6,
        linecolor="white",
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "mean recall"}
    )

    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title(f"confusion matrix (mean ± std over {num_runs} runs)")

    plt.savefig(os.path.join(save_fig_path, f"{config['dataset']}_{config['model_name']}_{num_runs}_train_test_split_confusion_matrix_mean_std.pdf"), dpi=600)
    plt.close()
    print(f"Saved confusion matrix (mean ± std over {num_runs} runs) to {save_fig_path}")

    random_train_test_result_df = pd.DataFrame({
        'Accuracy': acc_list,
        'Balanced Accuracy': b_acc_list,
        'ROC AUC OVO': roc_auc_ovo_list,
        'ROC AUC OVR': roc_auc_ovr_list,
        'PR AUC': pr_auc_list,
        'F1 Score': f1score_list,
        'Precision': prec_list,
        'Recall': rec_list,
        'MCC': mcc_list
    })

    print(f"\n\nRandom Train-Test Split Results over {num_runs} runs:"
          f"\nAverage Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}"
            f"\nAverage Balanced Accuracy: {np.mean(b_acc_list):.4f} ± {np.std(b_acc_list):.4f}"
            f"\nAverage ROC AUC OVO: {np.mean(roc_auc_ovo_list):.4f} ± {np.std(roc_auc_ovo_list):.4f}"
            f"\nAverage ROC AUC OVR: {np.mean(roc_auc_ovr_list):.4f} ± {np.std(roc_auc_ovr_list):.4f}"
            f"\nAverage PR AUC: {np.mean(pr_auc_list):.4f} ± {np.std(pr_auc_list):.4f}"
            f"\nAverage F1 Score: {np.mean(f1score_list):.4f} ± {np.std(f1score_list):.4f}"
            f"\nAverage Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}"
            f"\nAverage Recall: {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}"
            f"\nAverage MCC: {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}"
    )
    random_train_test_result_df.to_csv(os.path.join(save_fig_path, f'random_train_test_split_results_{config["dataset"]}.csv'), index=False)
    print(f"Random train-test split results for {num_runs} runs saved to 'random_train_test_split_results_{config['dataset']}.csv' inside {save_fig_path} directory.")


def random_features_model_accuracy(df, batch_size=20, debug=config['debug'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    '''
        Parameters:
                df (pandas.DataFrame): Original unmodified dataframe containing features and a target (in the last column)
                batch_size (int): Number of features to randomly select in each batch
                debug (bool): Whether to print debug information
                train_test_sep (bool): Whether to use a separate test set for evaluation. 
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                batch_numbers (list): A list of batch numbers
                accuracy_values (list): A list of accuracy values corresponding to each batch
                auc_values (list): A list of AUC values corresponding to each batch (if applicable).
                
    '''

    if batch_size < 1:
            print("Batch size should be at least 1.")
            return [], [], [], [], [], [], [], [], [], [], [], []
    
    target=config['target']
    remove_cols=config['remove_cols']
    stop_at = config['num_runs']   # Number of runs per random feature set

    save_cm_subset_selected_features = config['save_cm_subset_selected_features']   
    save_random_features = config['save_random_subset_selected_features']

    random_subset_features_list = [] 
    # Initialize list to store accuracy values
    batch_numbers = []

    accuracy_values = []
    auc_values = []
    pr_auc_values = []
    
    b_acc_values = []
    f1_values = []
    prec_values = []
    rec_values = []
    mcc_values = []

    cm_norm_values = []
    cm_counts_values = []

    # Initialize batch counter
    batch_number = 1

    y = df[target]
    class_labels = np.unique(y)
    n_classes = sorted(y.unique())
    
    n_cols = df.shape[1]-1

    # if n_cols > 20000:
    #     remove_cols = True
    # else:
    #     remove_cols = False

    if debug:
        print(f"Number of classes in target variable '{target}': {n_classes}")
        print(f"Type of the target variable 'n_classes': {type(n_classes)}", flush=True)
        

    df_rand_feat = df.drop(columns=[target]).copy()  # Drop the target column to get features only


    while len(df_rand_feat.columns) >= batch_size:
        # Randomly select columns
        
        selected_columns = np.random.choice(df_rand_feat.columns, size=batch_size, replace=False)

        if save_random_features:
            random_subset_features_list.append(selected_columns.tolist())

        # Extract the selected features
        X_batch = df_rand_feat[selected_columns]

        if not(train_test_sep):
        # Split the data into training and testing sets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_batch, y, stratify=y, test_size=0.2, random_state=config['random_state'])

        elif train_test_sep:

            # print(f'Using a separate test set for evaluation. Training with {batch_size} randomly selected features and evaluating on the separate test set with all features.')
            
            X_train_raw = X_batch
            y_train = y

            X_test_raw = test_df[selected_columns]
            y_test = test_df[target]

        tree_models = ["RF", "DT", "XGB", "LGB"]
        if config['model_name'] not in tree_models:
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            if debug:
                print("--"* 25)
                print("For training data:")
                check_scaling(X_train)
                print("--"* 25)
                print("For test data:")
                check_scaling(X_test)
                print("--"* 25)

        else:
            if debug:
                print("Using tree-based model, skipping scaling.")
            X_train = X_train_raw
            X_test = X_test_raw

        model = get_model(config['model_name'], opt_model=config['opt_model'])
        model.fit(X_train, y_train)
        
        # Predict on the test set and calculate accuracy
        y_pred = model.predict(X_test)

        if debug:
            print(f"Batch {batch_number}: Evaluated model with {len(selected_columns)} randomly selected features.", flush=True)
            print(f"samples and features in training set: {X_train.shape}")
            print(f"samples and features in test set: {X_test.shape}")

        if save_cm_subset_selected_features:
            cm_counts = confusion_matrix(
                y_test,
                y_pred,
                labels=class_labels,
                normalize=None          # raw counts
            )

            cm_norm = confusion_matrix(
                y_test,
                y_pred,
                labels=class_labels,
                normalize="true"        # row-wise %
            )
            if debug:
                print(f"\n\nConfusion Matrix by count for batch {batch_number} with {len(selected_columns)} selected features:")
                print(cm_counts)
                print(f"Confusion Matrix normalised for batch {batch_number} with {len(selected_columns)} selected features:")
                print(np.round(cm_norm, 3))
                print("\n\n")

            cm_counts_values.append(cm_counts)
            cm_norm_values.append(cm_norm)

        auc_scores_results = get_auc_scores(model, X_test, y_test)
        
        if debug:
            print(auc_scores_results)

        roc_auc = auc_scores_results['average']['roc_auc_ovo']
        pr_auc = auc_scores_results['average']['pr_auc']
        
        auc_values.append(roc_auc)
        pr_auc_values.append(pr_auc)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_values.append(accuracy)

        b_accuracy = balanced_accuracy_score(y_test, y_pred)
        b_acc_values.append(b_accuracy)

        f1score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_values.append(f1score)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        prec_values.append(precision)

        recall = recall_score(y_test, y_pred, average='macro')
        rec_values.append(recall)

        mcc = matthews_corrcoef(y_test, y_pred)
        mcc_values.append(mcc)

        batch_numbers.append(batch_number)

        if debug: 
            try:
                print(f"Batch {batch_number} - Accuracy: {accuracy:.4f}")
                print(f"Macro F1: {f1score:.4f}")
                print(f"AUC: {roc_auc:.4f}")
            except:
                pass

        # Remove the selected columns from the DataFrame
        if remove_cols:
            if debug:
                print("Removing columns")
            df_rand_feat = df_rand_feat.drop(columns=selected_columns).copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Increment the batch counter
        batch_number += 1

        if batch_number > stop_at:   # Stop after a certain number of runs
            break 
    
    if save_random_features:
        print(f"\n\nSaved random subsets of selected features for each batch.")
    
    print(f"=====Completed evaluation for random subset size {batch_size}.=====\n")

    return batch_numbers, accuracy_values, auc_values, pr_auc_values, b_acc_values, f1_values, prec_values, rec_values, mcc_values, cm_norm_values, cm_counts_values, random_subset_features_list


def get_result_random_sets(df, debug=config['debug'], use_active_suffix = 1, train_test_sep = config['train_test_sep'], test_df = pd.DataFrame()):

    dataset = config['dataset']
    model_name = config['model_name']
    num_runs = config['num_runs']

    remove_cols = config['remove_cols']
    n_cols = df.shape[1]-1

    # if n_cols > 20000:
    # if config['remove_cols']:
    #     remove_cols = True
    # else:
    #     remove_cols = False

    # remove_cols = config['remove_cols']

    if config['run_var_random_subsets_of_all_features'] or config['use_feature_ticks_ranges']:
        range_tuples = config["feature_ticks_ranges"]      # Sizes of random subsets
        length = sum(len(range(start, stop, step)) for start, stop, step in range_tuples)

    elif config['run_var_perc_random_subset_of_all_features'] or config['use_feature_ticks_perc_ranges']:
        range_tuples = config["feature_ticks_perc_ranges"]    # Percentages of random subsets
        length = sum(len(np.arange(start, stop, step)) for start, stop, step in range_tuples)

    if config['opt_model']:
        to_opt_model = "Optimized"
    else:
        to_opt_model = "Default"

    print(f"""Running the experiment with the following parameters:
          Dataset: {dataset}, 
          Are train and test sets different?: {bool(train_test_sep)},
          Model: {model_name}({to_opt_model}), 
          Number of runs per expr: {num_runs}       
          """)
    #   Remove columns?: {remove_cols}, 
    #   Print Debug Info: {bool(debug)}

    if config['interactive']:
        # If interactive mode is enabled, ask for confirmation before proceeding
        print("Interactive mode is enabled. Please confirm the settings before proceeding.")
        resp = input("\ncontinue with these settings? [y/n]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborting.")
            sys.exit(0)
    else:
        print("Interactive mode is disabled. Proceeding with the settings without confirmation.")

    batch_list = []

    batch_mean_acc_list = []
    batch_median_acc_list = []
    batch_std_acc_list = []

    batch_mean_auc_list = []
    batch_median_auc_list = []
    batch_std_auc_list = []

    batch_mean_pr_auc_list = []
    batch_median_pr_auc_list = []
    batch_std_pr_auc_list = []

    batch_mean_b_acc_list = []
    batch_std_b_acc_list = []
    
    batch_mean_f1_list = []
    batch_std_f1_list = []
    
    batch_mean_prec_list = []
    batch_std_prec_list = []
    
    batch_mean_rec_list = []
    batch_std_rec_list = []

    batch_mean_mcc_list = []
    batch_std_mcc_list = []


    if debug: 
        print(f"Mean and standard deviation are for {num_runs} runs.")
    
    pbar = tqdm(total=length, desc="Batching")

    if train_test_sep and test_df.empty:
        raise ValueError("Test set is empty, but 'train_test_sep' is True.")

    for batch in chain(*(np.arange(start, stop, step) for start, stop, step in range_tuples)):
        
        if debug:
            if config['run_var_random_subsets_of_all_features']:
                print(f"\n For random gene set size {batch}:")
            elif config['run_var_perc_random_subset_of_all_features']:
                print(f"\n For random gene set size {batch}% of all features:")

        if config['run_var_random_subsets_of_all_features'] or config['use_feature_ticks_ranges']:
            batch = batch
        elif config['run_var_perc_random_subset_of_all_features'] or config['use_feature_ticks_perc_ranges']:
            # print(f"Total number of features (excluding target): {n_cols}")
            batch = int(batch * n_cols / 100)

        if train_test_sep == 0:    
            batch_numbers, accuracy_random, auc_random, pr_auc_random, b_acc_random, f1_random, prec_random, rec_random, mcc_random, cm_norm_random, cm_count_random, random_subset_features_list = random_features_model_accuracy(df = df, 
                                                                                        batch_size = batch, 
                                                                                        debug = debug)
        elif train_test_sep == 1:
            # If train_test_sep is 1, we use a separate test set
            # We assume test_df is already provided and has the same structure as df
            batch_numbers, accuracy_random, auc_random, pr_auc_random, b_acc_random, f1_random, prec_random, rec_random, mcc_random, cm_norm_random, cm_count_random, random_subset_features_list = random_features_model_accuracy(df = df, 
                                                                                        batch_size = batch, 
                                                                                        debug = debug, 
                                                                                        train_test_sep = train_test_sep, 
                                                                                        test_df = test_df)


        mean_acc = round(np.mean(accuracy_random), 5)
        median_acc = round(np.median(accuracy_random), 5)
        std_acc = round(np.std(accuracy_random), 5)

        mean_b_acc = round(np.mean(b_acc_random), 5)
        std_b_acc = round(np.std(b_acc_random), 5)

        mean_f1 = round(np.mean(f1_random), 5)
        std_f1 = round(np.std(f1_random), 5)

        mean_prec = round(np.mean(prec_random), 5)
        std_prec = round(np.std(prec_random), 5)

        mean_rec = round(np.mean(rec_random), 5)
        std_rec = round(np.std(rec_random), 5)

        mean_mcc = round(np.mean(mcc_random), 5)
        std_mcc = round(np.std(mcc_random), 5)

        if debug:
            print(f"Accuracy Mean: {mean_acc}, STD: {std_acc}")
            print(f"Balanced Accuracy Mean: {mean_b_acc}, STD: {std_b_acc}")
            print(f"Macro F1 Mean: {mean_f1}, STD: {std_f1}")


        if len(auc_random) > 0:
            mean_auc = round(np.mean(auc_random), 5)
            std_auc = round(np.std(auc_random), 5)
            median_auc = round(np.median(auc_random), 5)
            if debug:
                print(f"AUC Mean: {mean_auc}, STD: {std_auc}")
        else:
            mean_auc = np.nan
            std_auc = np.nan
            median_auc = np.nan
            if debug:
                print(f"AUC Mean: {mean_auc}, STD: {std_auc}")

        if len(pr_auc_random) > 0:
            mean_pr_auc = round(np.mean(pr_auc_random), 5)
            std_pr_auc = round(np.std(pr_auc_random), 5)
            median_pr_auc = round(np.median(pr_auc_random), 5)
            if debug:
                print(f"PR AUC Mean: {mean_pr_auc}, STD: {std_pr_auc}")
        else:
            mean_pr_auc = np.nan
            std_pr_auc = np.nan
            median_pr_auc = np.nan
            if debug:
                print(f"PR AUC Mean: {mean_pr_auc}, STD: {std_pr_auc}")
        
        print("=="*25)

        batch_list.append(batch)
        batch_mean_acc_list.append(mean_acc)
        batch_median_acc_list.append(median_acc)
        batch_std_acc_list.append(std_acc)

        batch_mean_auc_list.append(mean_auc)  
        batch_median_auc_list.append(median_auc)
        batch_std_auc_list.append(std_auc)

        batch_mean_pr_auc_list.append(mean_pr_auc)
        batch_median_pr_auc_list.append(median_pr_auc)
        batch_std_pr_auc_list.append(std_pr_auc)

        batch_mean_b_acc_list.append(mean_b_acc)
        batch_std_b_acc_list.append(std_b_acc)

        batch_mean_f1_list.append(mean_f1)
        batch_std_f1_list.append(std_f1)

        batch_mean_prec_list.append(mean_prec)
        batch_std_prec_list.append(std_prec)

        batch_mean_rec_list.append(mean_rec)
        batch_std_rec_list.append(std_rec)

        batch_mean_mcc_list.append(mean_mcc)
        batch_std_mcc_list.append(std_mcc)
        

        pbar.update(1)
    pbar.close()

    if config['train_test_sep']:
        print(f"\t\tCompleted experiment with model: {model_name}, Train dataset: {dataset}, Test dataset: {config['test_dataset']}")
    else:
        print(f"\t\tCompleted experiment with model: {model_name}, Dataset: {dataset}")

    if train_test_sep:
        print(f'For this experiment, train and test sets are different.')

    if remove_cols:
        print(f"For each of the {num_runs} runs, genes were removed successively for model training.")
    else:
        pass

    result_df = pd.DataFrame({
        'Random Gene Set Size': batch_list,
        
        'Accuracy Mean': batch_mean_acc_list,
        'Accuracy Median': batch_median_acc_list,
        'Accuracy STD': batch_std_acc_list,

        'AUC Mean': batch_mean_auc_list,
        'AUC Median': batch_median_auc_list,
        'AUC STD': batch_std_auc_list,

        'PR AUC Mean': batch_mean_pr_auc_list,
        'PR AUC Median': batch_median_pr_auc_list,
        'PR AUC STD': batch_std_pr_auc_list,

        'Balanced Accuracy Mean': batch_mean_b_acc_list,
        'Balanced Accuracy STD': batch_std_b_acc_list,

        'Macro F1 Score Mean': batch_mean_f1_list,
        'Macro F1 Score STD': batch_std_f1_list,

        'Macro Precision Mean': batch_mean_prec_list,
        'Macro Precision STD': batch_std_prec_list,

        'Macro Recall Mean': batch_mean_rec_list,
        'Macro Recall STD': batch_std_rec_list,

        'MCC Mean': batch_mean_mcc_list,
        'MCC STD': batch_std_mcc_list,

    })

    if config['opt_model']:
        to_opt_model = "O"
    else:
        to_opt_model = "D"

    file_name = f'{dataset}_{model_name}_({to_opt_model})_random_features_{active_suffix if (use_active_suffix and active_suffix is not None) else "all"}_extended_results.csv'

    result_file = os.path.join(save_fig_path, file_name)

    result_df.to_csv(result_file, index=False)

    print(f"Saved extended results to CSV file at {result_file}")

    return result_file

def plot_random_features_acc_auc(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=config['plot_acc'], plot_auc=config['plot_auc'], debug=config['debug']):

    dataset = config['dataset']
    model_name = config['model_name']
    num_runs = config['num_runs']
    remove_cols = config['remove_cols']
    train_test_sep = config['train_test_sep']

    range_tuples = config["feature_ticks_ranges"]  # Sizes of random subsets

    # lengths for each range
    lengths = [len(range(start, stop, step)) for start, stop, step in range_tuples]

    # compute cumulative index boundaries
    boundaries = [0]
    for l in lengths:
        boundaries.append(boundaries[-1] + l)

    # make (start_idx, end_idx, step) pairs per range
    index_spans = [
        (boundaries[i], boundaries[i+1], range_tuples[i][2])
        for i in range(len(range_tuples))
        ]

    (x1_low, x1_high, x1_step), (x2_low, x2_high, x2_step), (x3_low, x3_high, x3_step) = index_spans

    if train_test_sep:
        print(f'For this experiment, train and test sets are different.')
        test_dataset = config['test_dataset']
        print(f'Train dataset: {dataset}')
        print(f'Test dataset: {test_dataset}')
    else:
        print(f'For this experiment, train and test sets are the same from {dataset} dataset.')

    if config['remove_cols']:
        set_selection_method = "w_o_replacement"
    else:
        set_selection_method =  "w_replacement"

    result_df = pd.read_csv(result_csv_file)
    if debug:
        print(f"Loaded results from {result_csv_file}")
        print(result_df.head())

    if plot_acc:
        metric = "Accuracy"
        reference_value = round(acc_full_feature,3)
        x1 = result_df['Random Gene Set Size'][x1_low:x1_high] 
        y1 = result_df['Accuracy Mean'][x1_low:x1_high] 
        y1_std = result_df['Accuracy STD'][x1_low:x1_high] 


        x2 = result_df['Random Gene Set Size'][x2_low:x2_high] 
        y2 = result_df['Accuracy Mean'][x2_low:x2_high]
        y2_std = result_df['Accuracy STD'][x2_low:x2_high]


        x3 = result_df['Random Gene Set Size'][x3_low:x3_high]
        y3 = result_df['Accuracy Mean'][x3_low:x3_high]
        y3_std = result_df['Accuracy STD'][x3_low:x3_high]


    elif plot_auc:
        metric = "AUC"
        reference_value = round(auc_full_feature,3)
        x1 = result_df['Random Gene Set Size'][x1_low:x1_high]
        y1 = result_df['AUC Mean'][x1_low:x1_high]
        y1_std = result_df['AUC STD'][x1_low:x1_high]


        x2 = result_df['Random Gene Set Size'][x2_low:x2_high]
        y2 = result_df['AUC Mean'][x2_low:x2_high]
        y2_std = result_df['AUC STD'][x2_low:x2_high]


        x3 = result_df['Random Gene Set Size'][x3_low:x3_high]
        y3 = result_df['AUC Mean'][x3_low:x3_high]
        y3_std = result_df['AUC STD'][x3_low:x3_high]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 8))

    # First axis for 1-50
    ax1.errorbar(x1, y1, yerr=y1_std, fmt='-o', capsize=5, color='blue')
    
    ax1.set_xlim(min(x1), max(x1))
    ax1.set_ylim(0, 1.01)
    ax1.set_xticks(range(min(x1), max(x1)+1, x1_step))    
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.set_yticks((np.arange(0, 1.01, step=0.1)))

    ax1.set_title(f'Range {min(x1)}-{max(x1)} with step size {x1.iloc[1] - x1.iloc[0]}', fontsize=12)
    ax1.set_ylabel(f'{metric} (mean ± std over {num_runs} Runs)', fontsize=16)

    # Second axis for 51-200

    ax2.errorbar(x2, y2, yerr=y2_std, fmt='-o', capsize=5, color='red')
    
    ax2.set_xlim(min(x2), max(x2))

    ax2.set_xticks(range(min(x2), max(x2)+1, x2_step))
    ax2.tick_params(axis='x', rotation=45, labelsize=11)

    ax2.set_title(f'Range {min(x2)}-{max(x2)} with step size {x2.iloc[1] - x2.iloc[0]}', fontsize=12)
    ax2.set_xlabel('Number of randomly selected features', fontsize=16)
    

    # Third axis for 200-2000
    ax3.errorbar(x3, y3, yerr=y3_std, fmt='-o', capsize=5, color='green')
    
    ax3.set_xlim(min(x3), max(x3))

    ax3.set_title(f'Range {min(x3)}-{max(x3)} with step size {x3.iloc[1] - x3.iloc[0]}', fontsize=12)
    ax3_x_ticks = list(range(min(x3), max(x3)+1, x3_step))
    ax3.set_xticks(ax3_x_ticks)
    ax3.tick_params(axis='x', rotation=45, labelsize=11)

    # Hide the spines between the two axes
    ax1.spines['right'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['left'].set_visible(False)

    ax1.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    ax2.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    ax3.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} \n with all features ({num_features}): {reference_value}')

    ax1.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02}')
    ax2.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    ax3.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    
    ax1.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    ax2.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    ax3.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    
    if plot_acc:
        if config['annotate_paper_results']:
            x_from_paper = config['x_from_paper']
            y_from_paper = config['y_from_paper']
            Paper = config['paper']
            print(f"Annotating paper results: {Paper}, x: {x_from_paper}, y: {y_from_paper}")
            ax1.annotate(f'Paper: {Paper} \nSelected Features: {x_from_paper} \nAccuracy: {y_from_paper:.2f}', 
                        xy=(x_from_paper, y_from_paper), 
                        #  xytext=(x_from_paper+10, y_from_paper+0.08),
                        xytext=(0.6, 0.3),
                        textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
            ax1.scatter(x_from_paper, y_from_paper, color='red', marker='x', s=100, label=f'{x_from_paper} (acc={y_from_paper:.2f})')

    if config['opt_model']:
        model_opt = "O"
    else:
        model_opt = "D"

    if train_test_sep:
        fig.suptitle(f'{metric} vs Random Features \n Train Dataset: {dataset} ({num_samples} samples, {num_classes} classes, {num_features} {active_suffix if active_suffix else ""} features) Test Dataset: {test_dataset}')   
    else:
        fig.suptitle(f'{metric} vs Random Features; {dataset} Dataset ({num_samples} samples, {num_classes} classes, {num_features} {active_suffix if active_suffix else ""} features)')   

    ax2.legend(loc='lower center', fontsize=11)


    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)  

    fname = dataset+"_"+model_name+"_("+model_opt+")_"+metric+"_"+set_selection_method.replace(" ","_")+".pdf"
    file_path = os.path.join(save_fig_path, fname)
    
    print(f"\n\nSaving {metric} figure to {file_path}")
    plt.savefig(file_path, dpi=600)
    # plt.show()


def plot_perc_random_features_acc_auc(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=config['plot_acc'], plot_auc=config['plot_auc'], debug=config['debug']):

    dataset = config['dataset']
    model_name = config['model_name']
    num_runs = config['num_runs']
    train_test_sep = config['train_test_sep']

    def count_to_pct(x):
        return (x / num_features) * 100

    def pct_to_count(x):
        return (x / 100) * num_features


    range_tuples = config["feature_ticks_perc_ranges"]    # Percentages of random subsets

    # # lengths for each range
    # lengths = [len(np.arange(start, stop, step)) for start, stop, step in range_tuples]

    # # compute cumulative index boundaries
    # boundaries = [0]
    # for l in lengths:
    #     boundaries.append(boundaries[-1] + l)

    # # make (start_idx, end_idx, step) pairs per range
    # index_spans = [
    #     (boundaries[i], boundaries[i+1], range_tuples[i][2])
    #     for i in range(len(range_tuples))
    #     ]

    (x1_low_orig, x1_high_orig, x1_step_orig), (x2_low_orig, x2_high_orig, x2_step_orig) = range_tuples

    x1 = int(x1_low_orig *num_features/100), int((x1_high_orig-x1_step_orig)*num_features/100)
    x2 = int(x2_low_orig *num_features/100), int((x2_high_orig-x2_step_orig)*num_features/100)

    if debug:
        print(f"Converted percentage ranges to feature counts:")
        print(f"Range 1: {x1_low_orig}% - {x1_high_orig}% --> {x1[0]} - {x1[1]} features")
        print(f"Range 2: {x2_low_orig}% - {x2_high_orig}% --> {x2[0]} - {x2[1]} features")

    x1_low, x1_high = x1
    x2_low, x2_high = x2

    if train_test_sep:
        print(f'For this experiment, train and test sets are different.')
        test_dataset = config['test_dataset']
        print(f'Train dataset: {dataset}')
        print(f'Test dataset: {test_dataset}')
    else:
        print(f'For this experiment, train and test sets are the same from {dataset} dataset.')

    if config['remove_cols']:
        set_selection_method = "w_o_replacement"
    else:
        set_selection_method =  "w_replacement"

    result_df = pd.read_csv(result_csv_file)
    if debug:
        print(f"Loaded results from {result_csv_file}")
        print(result_df.head())

    if plot_acc:
        metric = "Accuracy"
        reference_value = round(acc_full_feature,3)

        mask1 = (result_df['Random Gene Set Size'] >= x1_low) & \
        (result_df['Random Gene Set Size'] <= x1_high)

        mask2 = (result_df['Random Gene Set Size'] >= x2_low) & \
                (result_df['Random Gene Set Size'] <= x2_high)

        x1 = result_df.loc[mask1, 'Random Gene Set Size']
        # x1 = ((x1 / num_features) * 100).round().round(1)
        y1 = result_df.loc[mask1, 'Accuracy Mean']
        y1_std = result_df.loc[mask1, 'Accuracy STD']

        x2 = result_df.loc[mask2, 'Random Gene Set Size']
        # x2 = ((x2 / num_features) * 100).round().round(1)
        y2 = result_df.loc[mask2, 'Accuracy Mean']
        y2_std = result_df.loc[mask2, 'Accuracy STD']


        if debug:
            print(f"x1 values: {x1.tolist()}")
            print(f"y1 values: {y1.tolist()}")
            print(f"y1_std values: {y1_std.tolist()}")

            print(f"x2 values: {x2.tolist()}")
            print(f"y2 values: {y2.tolist()}")
            print(f"y2_std values: {y2_std.tolist()}")



    elif plot_auc:
        metric = "AUC"
        reference_value = round(auc_full_feature,3)

        mask1 = (result_df['Random Gene Set Size'] >= x1_low) & \
        (result_df['Random Gene Set Size'] <= x1_high)

        mask2 = (result_df['Random Gene Set Size'] >= x2_low) & \
                (result_df['Random Gene Set Size'] <= x2_high)

        x1 = result_df.loc[mask1, 'Random Gene Set Size']
        # x1 = ((x1 / num_features) * 100).round().round(1)
        y1 = result_df.loc[mask1, 'AUC Mean']
        y1_std = result_df.loc[mask1, 'AUC STD']

        x2 = result_df.loc[mask2, 'Random Gene Set Size']
        # x2 = ((x2 / num_features) * 100).round().round(1)
        y2 = result_df.loc[mask2, 'AUC Mean']
        y2_std = result_df.loc[mask2, 'AUC STD']


        if debug:
            print(f"x1 values: {x1.tolist()}")
            print(f"y1 values: {y1.tolist()}")
            print(f"y1_std values: {y1_std.tolist()}")
            
            print(f"x2 values: {x2.tolist()}")
            print(f"y2 values: {y2.tolist()}")
            print(f"y2_std values: {y2_std.tolist()}")
 

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

    # First axis
    ax1.errorbar(x1, y1, yerr=y1_std, fmt='-o', capsize=5, color='blue')
    
    # ax1.set_xlim(x1_low, x1_high)
    ax1.set_ylim(0, 1.01)

    # remove primary x ticks and labels
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    # optionally remove the primary x-axis spine
    ax1.spines['bottom'].set_visible(False)
    
    # ax1.set_xticks(x1)    
    # ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.set_yticks((np.arange(0, 1.01, step=0.1)))

    ax1.set_title(f'Range {x1_low_orig}% - {x1_high_orig-x1_step_orig}% with step size {x1_step_orig}', fontsize=12)
    ax1.set_ylabel(f'{metric} (mean ± std over {num_runs} Runs)', fontsize=16)

    secax1 = ax1.secondary_xaxis(
        'bottom',
        functions=(count_to_pct, pct_to_count)
    )
    secax1.spines['bottom'].set_position(('outward', 20))
    # secax1.set_xlabel('percentage of randomly selected features', fontsize=14)
    secax1.xaxis.set_major_formatter(lambda x1, _: f"{x1:.1f}%")

    # Second axis

    ax2.errorbar(x2, y2, yerr=y2_std, fmt='-o', capsize=5, color='red')
    
    # ax2.set_xlim(x2_low, x2_high)
    # ax2.set_xticks(x2)
    # ax2.tick_params(axis='x', rotation=45, labelsize=11)

    ax2.set_title(f'Range {x2_low_orig}% - {x2_high_orig-x2_step_orig}% with step size {x2_step_orig}', fontsize=12)

    # remove primary x ticks and labels
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    # optionally remove the primary x-axis spine
    ax2.spines['bottom'].set_visible(False)

    secax2 = ax2.secondary_xaxis(
        'bottom',
        functions=(count_to_pct, pct_to_count)
    )
    secax2.spines['bottom'].set_position(('outward', 20))
    # secax2.set_xlabel('percentage of randomly selected features', fontsize=14)
    secax2.xaxis.set_major_formatter(lambda x2, _: f"{x2:.1f}%")

    ref_percents = [1, 2, 5, 10]
    for p in ref_percents:
        x_ref = (p / 100) * num_features
        ax2.axvline(
            x=x_ref,
            color='grey',
            linestyle='--',
            linewidth=1.0,
            alpha=0.8,
            zorder=1
        )
    fig.supxlabel('Percentage of randomly selected features', fontsize=16)
    

    # # Third axis
    # ax3.errorbar(x3, y3, yerr=y3_std, fmt='-o', capsize=5, color='green')
    
    # ax3.set_xlim(min(x3), max(x3))

    # ax3.set_title(f'Range {min(x3)}-{max(x3)} with step size {x3.iloc[1] - x3.iloc[0]}', fontsize=12)
    # ax3_x_ticks = list(range(min(x3), max(x3)+1, x3_step))
    # ax3.set_xticks(ax3_x_ticks)
    # ax3.tick_params(axis='x', rotation=45, labelsize=11)

    # Hide the spines between the two axes
    ax1.spines['right'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ax3.spines['left'].set_visible(False)

    ax1.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    ax2.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    # ax3.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} \n with all features ({num_features}): {reference_value}')

    ax1.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02}')
    ax2.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    # ax3.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    
    ax1.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    ax2.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    # ax3.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    
    if plot_acc:
        if config['annotate_paper_results']:
            x_from_paper = config['x_from_paper']
            y_from_paper = config['y_from_paper']
            Paper = config['paper']
            print(f"Annotating paper results: {Paper}, x: {x_from_paper}, y: {y_from_paper}")
            ax1.annotate(f'Paper: {Paper} \nSelected Features: {x_from_paper} \nAccuracy: {y_from_paper:.2f}', 
                        xy=(x_from_paper, y_from_paper), 
                        #  xytext=(x_from_paper+10, y_from_paper+0.08),
                        xytext=(0.6, 0.3),
                        textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
            ax1.scatter(x_from_paper, y_from_paper, color='red', marker='x', s=100, label=f'{x_from_paper} (acc={y_from_paper:.2f})')

    if config['opt_model']:
        model_opt = "O"
    else:
        model_opt = "D"
    
    if config['run_var_perc_random_subset_of_all_features']:
        use_active_suffix = 0
    else:
        use_active_suffix = 1

    if train_test_sep:
        fig.suptitle(f'{metric} vs Random Features \n Train Dataset: {dataset} ({num_samples} samples, {num_classes} classes, {num_features} {active_suffix if (use_active_suffix and active_suffix) else "all"} features) Test Dataset: {test_dataset}')   
    else:
        fig.suptitle(f'{metric} vs Random Features; {dataset} Dataset ({num_samples} samples, {num_classes} classes, {num_features} {active_suffix if (use_active_suffix and active_suffix) else "all"} features)')   

    ax2.legend(loc='lower center', fontsize=11)


    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)  

    fname = f'{dataset}_{model_name}_({model_opt})_{metric}_{set_selection_method.replace(" ","_")}_perc_{active_suffix if (use_active_suffix and active_suffix) else "all"}_features.pdf'
    file_path = os.path.join(save_fig_path, fname)
    
    print(f"\n\nSaving {metric} figure to {file_path}")
    plt.savefig(file_path, dpi=600)
    # plt.show()

def cv_spline_scores(x, y, s_values, k=3, cv_folds=5):
    scores = []
    for s in s_values:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config['random_state'])
        fold_scores = []
        for train_idx, test_idx in kf.split(x):
            spline = UnivariateSpline(x[train_idx], y[train_idx], s=s, k=k)
            y_pred = spline(x[test_idx])
            fold_scores.append(mean_squared_error(y[test_idx], y_pred))
        scores.append(np.mean(fold_scores))
    return np.array(scores)

def get_area_above_curve(result_csv_path, x_from_paper, y_from_paper, debug=1):

    """
    Computes the area above the curve (AAC) for a given model and dataset.
    
    Parameters:
        df (pandas DataFrame): A dataframe containing features and a target (in the last column)
        target (str): The name of the target column in the dataframe
        model_name (str): The name of the model to be used
        dataset (str): The name of the dataset
        debug (bool): Whether to print debug information
        opt_model (int): Whether to use an optimized model or not
        train_test_sep (int): If 1, uses a separate test set provided in `test_df`
        test_df (pandas DataFrame): Separate test set dataframe, required if `train_test_sep` is set to 1

    Returns:
        float: The area above the curve (AAC) for the given model and dataset
    """   
    plot_data = pd.read_csv(result_csv_path)
    plt.clf() 

    x_data = plot_data['Random Gene Set Size']
    
    if config['plot_acc']:
        metric = "Accuracy"
        y_data = plot_data['Accuracy Mean']
    elif config['plot_auc']:
        metric = "AUC"
        y_data = plot_data['AUC Mean']
    else:
        raise ValueError("Please set config['plot_acc'] or config['plot_auc'] to True.") 

    assert_fraction(y_from_paper)
    
    if debug:
        print(f"No. of datapoints to approximate the spline fit : {len(x_data)}")

    # search range
    s_values = np.logspace(-2, 3, 50)
    scores = cv_spline_scores(x_data, y_data, s_values)
    best_s = s_values[np.argmin(scores)]

    if debug:
        print(f"Best smoothing factor (s): {best_s:.2g}")
        # fit final spline with best_s
        spline = UnivariateSpline(x_data, y_data, s=best_s)

        # generate points for plotting
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_fit = spline(x_fit)

        # plot
        plt.figure(figsize=(8, 5))
        
        plt.plot(x_data, y_data, 'o', label='data')
        plt.plot(x_fit, y_fit, label=f'spline fit (s={best_s:.2g})')
        
        plt.xlabel('number of features')
        plt.ylabel(f"{metric}  value")
        
        plt.axhline(y=y_from_paper, color='r', linestyle='--', label=f'{metric} from paper: {y_from_paper:.3f}')
        plt.axvline(x=x_from_paper, color='b', linestyle='--', label=f'Features from paper: {x_from_paper}')

        plt.title(f'Spline fit to {metric} data for {config["dataset"]} with {config["model_name"]} model')
        
        plt.xlim(x_data.min(), x_data.max())
        plt.ylim(0, 1.01)
        
        plt.xticks(np.arange(0, x_data.max() + 1, step=200))
        plt.yticks(np.arange(0, 1.01, step=0.1))
        
        plt.legend()
        
        # plt.show()

    # interpolate f(t)
    f = UnivariateSpline(x_data, y_data, s=best_s)

    # define function to find root of f(t) - y
    def diff(t):    
        return f(t) - y_from_paper

    # pick a bracket (must ensure f(t) crosses y in this interval)
    # bracket = [x_from_paper + 1e-3, x_data.iloc[-1]]
    bracket = [0, x_data.iloc[-1]]

    try:
        # find root
        print(f"\n\nFinding root in the interval {bracket}")
        res = root_scalar(diff, bracket=bracket, method='brentq')
    except ValueError as e:
        print(f"Error: {e}")
        res = None

    if res is not None:
        x_prime = res.root if res.converged else None
        print(f"Similar result can be achieved with {x_prime} features.")
    else:
        x_prime = None
        print("No root found.")
        
    if debug:
        print(f"x: {x_from_paper}, y: {y_from_paper}, x': {x_prime}")

        # compute shaded area between f(t) and y from x to x'

    if x_prime is not None:
        area, _ = quad(lambda t: y_from_paper - f(t), x_from_paper, x_prime)
        print(f"Area above the curve with the point ({x_from_paper}, {y_from_paper:.2f}) from x = {x_from_paper} to x' = {x_prime:.2f} ==> {area:.5f}")
    else:
        print("curve never crosses y again after x: area = 0")
        area = 0

    plt.plot(x_data, f(x_data), label='best spline fit')
    plt.plot(x_from_paper, y_from_paper, 'ro')
    plt.xlim(x_data.min(), x_data.max())
    plt.ylim(0, 1.01)

    try:
        plt.plot(x_prime, f(x_prime), 'go')
        plt.annotate(f" Min. Features needed \n to match: {x_prime:.0f}", xy=(x_prime, f(x_prime)), xytext=(x_prime+800, f(x_prime)-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='bottom')
        plt.annotate(f"{metric} from a \n published study: ({x_from_paper:.0f}, {y_from_paper:.4f})", xy=(x_from_paper, y_from_paper), xytext=(x_from_paper+300, y_from_paper+0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='bottom')

        plt.xlim(x_data.min(), x_data.max())
        plt.ylim(0, 1.01)

        plt.xticks(np.arange(0, x_data.max() + 1, step=200))
        plt.yticks(np.arange(0, 1.01, step=0.1))
        
        plt.xlabel('No of genes randomly selected for model training')
        plt.ylabel(f'Mean {metric} over 20 runs')

        plt.axhline(y=y_from_paper, color='r', linestyle='--')
        plt.axvline(x=x_from_paper, color='b', linestyle='--')

        # shade area between curve and y_ref from x_known to x_root
        x_shade = x_data[(x_data >= x_from_paper) & (x_data <= x_prime)]
        y_shade = f(x_shade)
        x_poly = np.concatenate(([x_from_paper], x_shade, [x_prime]))
        y_poly = np.concatenate(([y_from_paper], y_shade, [y_from_paper]))
        plt.fill(x_poly, y_poly, color='skyblue', alpha=0.4, label=f'Approx. Area: {area:.2f}')
    except:
        print("Random set can not match the performance of the published study.")

    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    fname = f"{config['dataset']}_{config['model_name']}_({'O' if config['opt_model'] else 'D'})_{metric}_area_plot.pdf"

    file_path = os.path.join(save_fig_path, fname)

    print(f"\n\nSaving area above the curve plot to {file_path}")
    plt.savefig(file_path, dpi=600)

    plt.legend(loc='best')
    # plt.show()

    return area

def test_random_features_model_accuracy(df, batch_size=20, debug=config['debug'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    '''
        Parameters:
                df (pandas.DataFrame): Original unmodified dataframe containing features and a target (in the last column)
                batch_size (int): Number of features to randomly select in each batch
                debug (bool): Whether to print debug information
                train_test_sep (bool): Whether to use a separate test set for evaluation. 
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                batch_numbers (list): A list of batch numbers
                accuracy_values (list): A list of accuracy values corresponding to each batch
                auc_values (list): A list of AUC values corresponding to each batch (if applicable).
    '''
    target=config['target']
    remove_cols=config['remove_cols']
    stop_at = config['num_runs']   # Number of runs per random feature set

    # Initialize list to store accuracy values
    accuracy_values = []
    auc_values = []
    batch_numbers = []

    # accuracy_cis = []
    # auc_cis = []

    # Initialize batch counter
    batch_number = 1

    y = df[target]
    n_classes = len(np.unique(y))

    if debug:
        print(f"Number of classes in target variable '{target}': {n_classes}")
        print(f"Type of the target variable 'n_classes': {type(n_classes)}")
        

    df_rand_feat = df.drop(columns=[target]).copy()  # Drop the target column to get features only

    while len(df_rand_feat.columns) >= batch_size:
        # Randomly select columns
        selected_columns = np.random.choice(df_rand_feat.columns, size=batch_size, replace=False)

        # Extract the selected features
        X_batch = df_rand_feat[selected_columns]

        if not(train_test_sep):
            
            df_random_features = pd.concat([X_batch, y], axis=1)
            accuracy, roc_auc_ovo_test, roc_auc_ovr, pr_auc = check_model_acc_full_feature(df = df_random_features, debug=0)

        elif train_test_sep:

            train_df_random_features = pd.concat([X_batch, y], axis=1)
            test_df_random_features = pd.concat([test_df[selected_columns], test_df[target]], axis=1)

            accuracy, roc_auc_ovo_test, roc_auc_ovr, pr_auc = check_model_acc_full_feature(df = train_df_random_features, 
                                         train_test_sep = 1, 
                                         test_df = test_df_random_features,
                                         debug=0
            )
        
        batch_numbers.append(batch_number)
        auc_values.append(roc_auc_ovr) 
        accuracy_values.append(accuracy)
        # pr_auc_values.append(pr_auc)

        if debug: 
            try:
                print(f"Batch {batch_number} - Accuracy: {accuracy:.4f}")
                print(f"AUC: {roc_auc_ovr:.4f}")
            except:
                pass

        # Remove the selected columns from the DataFrame
        if remove_cols:
            if debug:
                print("Removing columns")
            df_rand_feat = df_rand_feat.drop(columns=selected_columns).copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Increment the batch counter
        batch_number += 1

        if batch_number > stop_at:   # Stop after a certain number of runs
            break 

    return batch_numbers, accuracy_values, auc_values


def evaluate_model_with_perc_random_features(data_final, perc_sel, use_active_suffix = 1, test_df = pd.DataFrame()):
        '''
        Evaluates the model's performance using randomly selected features.

        Parameters:
                data_final (pandas.DataFrame): The final dataset containing features and target variable.
                perc_sel (float): Percentage of features to select randomly in each run.
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                None
        '''

        def spinner(message):
            """A simple spinner to indicate progress."""
            while not done:
                for char in "|/-\\":
                    print(f"\r{message} {char}", end="", flush=True)
                    time.sleep(0.1)
            print("\r" + " " * (len(message) + 2) + "\r", end="", flush=True)

        # Determine the percentage of features to select randomly
        # perc_sel = config['perc_random_subset_selected_features_subsets'][1]
        
        perc_sel = perc_sel / 100  # percentage of features to select randomly in each run
        batch_size = int(perc_sel * (data_final.shape[1]-1))  # exclude target column   


        # perc_sel = fs_utils.config['per_random_subset_size'][0] / 100  # percentage of features to select randomly in each run

        print(f"\n\nStarting evaluation of model's performance with {perc_sel*100:.2f}% features randomly selected from {data_final.shape[1]-1} features", flush=True)

        # Start the timer for checking model accuracy with selected features
        start = time.perf_counter()

        done = False
        if config['show_progress_bar']:
            t = threading.Thread(target=spinner,  args=(f"Evaluating model on {perc_sel*100:.2f}% features...",))
            t.start()

        if config['train_test_sep']:
            batch_numbers, accuracy_values, auc_values, pr_auc_values, b_acc_values, f1_values, prec_values, rec_values, mcc_values, cm_norm_values, cm_counts_values, random_subset_features_list = random_features_model_accuracy(
                                        df = data_final,
                                        batch_size = batch_size,  # excluding target column,
                                        train_test_sep = 1, 
                                        test_df = test_df
                                        )
                                        
        else:
            if debug:
                print("Using same dataset for train and test.")
                print(f"data_final.shape: {data_final.shape}")
                print(f"batch_size: {batch_size}", flush=True)

            batch_numbers, accuracy_values, auc_values, pr_auc_values, b_acc_values, f1_values, prec_values, rec_values, mcc_values, cm_norm_values, cm_counts_values, random_subset_features_list = random_features_model_accuracy(
                                        df = data_final,
                                        batch_size = batch_size  # excluding target column
                                        )

        if config['show_progress_bar']:
            done = True
            t.join()

        # End the timer for checking model accuracy with selected features
        end = time.perf_counter()

        if config['num_runs'] == len(batch_numbers):
            print(f"All {config['num_runs']} runs completed successfully.")
        else:
            print(f"Only {len(batch_numbers)} out of {config['num_runs']} runs completed.")

        if config['save_cm_subset_selected_features']:
            mean_sd_cm_separate_plots = False

            # print("\nGenerating and saving confusion matrix plots...", flush=True)

            cm_save_fig_path = os.path.join(
                    save_fig_path, f"cm_{perc_sel}_features"
                    )
            if not os.path.exists(cm_save_fig_path):
                os.makedirs(cm_save_fig_path)       

            class_labels = sorted(data_final[config['target']].unique())

            cm_norm_values   = np.array(cm_norm_values)
            cm_counts_values = np.array(cm_counts_values)

            assert cm_counts_values.shape == cm_norm_values.shape
            # assert cm_counts_values.ndim == 3

            print(f'cm_counts_values.shape: {cm_counts_values.shape}')
            print(f'cm_norm_values.shape: {cm_norm_values.shape}')
            print(f'cm_counts_values.ndim: {cm_counts_values.ndim}')
            print(f'cm_norm_values.ndim: {cm_norm_values.ndim}')

            num_runs, n_classes, _ = cm_norm_values.shape

            if config['save_individual_cm_plots']:
                for k in range(num_runs):

                    annot = np.empty((n_classes, n_classes), dtype=object)

                    for i in range(n_classes):
                        for j in range(n_classes):
                            annot[i, j] = (
                                f"{cm_counts_values[k, i, j]}\n"
                                f"({cm_norm_values[k, i, j]*100:.1f}%)"
                            )
                    if n_classes <= 10:
                        annot = annot
                    else:
                        annot = None

                    plt.figure(figsize=(7, 6))
                    sns.heatmap(
                        cm_counts_values[k],
                        annot=annot,
                        fmt="",
                        # cmap="Blues",
                        xticklabels=class_labels,
                        yticklabels=class_labels,
                        cbar_kws={"label": "sample count"}
                    )

                    plt.xlabel("predicted label")
                    plt.ylabel("true label")
                    # plt.title(f"confusion matrix (run {k+1})")

                    plt.title(
                        f"{int(perc_sel*100)}% random features | "
                        f"{config['dataset']} | "
                        f"{config['model_name']} | "
                        f"run {k+1}"
                    )
                    
                    plt.savefig(
                        os.path.join(
                            cm_save_fig_path,
                            f"{config['dataset']}_{config['model_name']}_run{k+1}_random_subset_{int(perc_sel*100)}pct_confusion_matrix_{active_suffix if (use_active_suffix and active_suffix) else 'all'}.pdf"
                        ),
                        dpi=600
                    )
                    plt.close()

                    print(f"Saved {num_runs} confusion matrices to {save_fig_path}")   

            cm_array = np.array(cm_norm_values)

            cm_mean = cm_array.mean(axis=0)
            cm_std  = cm_array.std(axis=0)

            if mean_sd_cm_separate_plots:
                print(f'cm_array.shape: {cm_array.shape}')
                print(f'cm_mean.shape: {cm_mean.shape}')
                print(f'cm_std.shape: {cm_std.shape}')

                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

                if n_classes <= 10:
                    annot_bool = True
                else:
                    annot_bool = False

                sns.heatmap(
                    cm_mean,
                    annot=annot_bool,
                    fmt=".2f",
                    # cmap="Greens",
                    ax=axes[0],
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    cbar_kws={"label": "mean recall"}
                )
                axes[0].set_title("mean")

                sns.heatmap(
                    cm_std,
                    annot=annot_bool,
                    fmt=".2f",
                    # cmap="Oranges",
                    ax=axes[1],
                    xticklabels=class_labels,
                    yticklabels=False,
                    cbar_kws={"label": "std across runs"}
                )
                axes[1].set_title("std")

                for ax in axes:
                    ax.set_xlabel("predicted label")
                axes[0].set_ylabel("true label")

                plt.suptitle(f"confusion matrix (mean ± std) over {num_runs} runs")

            else:
            # build annotation strings: mean ± std
            #  annot = np.full(cm_mean.shape, "", dtype=object)
            # for i in range(cm_mean.shape[0]):
                # annot[i, i] = f"{cm_mean[i,i]:.2f}\n± {cm_std[i,i]:.2f}"
                annot = np.empty_like(cm_mean, dtype=object)
                for i in range(cm_mean.shape[0]):
                    for j in range(cm_mean.shape[1]):
                        annot[i, j] = f"{cm_mean[i,j]:.4f}\n ± {cm_std[i,j]:.2f}"

                plt.figure(figsize=(7, 6))

                if n_classes <= 10:
                    annot = annot
                else:
                    annot = None
                    # cm_norm_values.to_csv(os.path.join(cm_save_fig_path, f"{config['dataset']}_{config['model_name']}_random_subset_{int(perc_sel*100)}pct_{active_suffix if active_suffix else ''}_cm_values.csv"))
                    # print(f"Saved cm_norm_values to {cm_save_fig_path}")

                sns.heatmap(
                    cm_mean,
                    annot=annot,
                    fmt="",
                    # square=True,
                    cmap = plt.cm.viridis,
                    vmin=0, vmax=1,
                    linewidths=0.6,
                    linecolor="white",
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    annot_kws={"size": 9, "weight": "bold"},
                    cbar_kws={"label": "mean recall"}
                )

                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                
                if config['train_test_sep']:
                    plt.title(f"Train: {config['dataset']} ({data_final.shape}, classes:{n_classes}), {int(perc_sel*100)}pct {active_suffix if active_suffix is not None else ''} \n Test: {config['test_dataset']} {test_df.shape} \n Confusion Matrix (Mean ± Std over 20 runs)")
                else:
                    plt.title(f"{config['dataset']} ({data_final.shape}, classes:{n_classes}), {int(perc_sel*100)}pct {active_suffix if active_suffix is not None else ''} \n Confusion Matrix (Mean ± Std over 20 runs)")

            # plt.title(
            #     f"{int(perc_sel*100)}% random features | "
            #     f"{config['dataset']} | "
            #     f"{config['model_name']} | "
            #     f"{num_runs} runs",
            #     fontsize=11
            # )
            # plt.suptitle(f"Confusion Matrix Mean and Std across {num_runs} Runs for {int(perc_sel*100)}% Randomly Selected Features for {config['dataset']} with {config['model_name']} Model")
            if use_active_suffix:
                plt.savefig(os.path.join(cm_save_fig_path, f"{config['dataset']}_{config['model_name']}_random_subset_{int(perc_sel*100)}pct_all_features_cm_mean_std.pdf"), dpi=600)
            else:
                plt.savefig(os.path.join(cm_save_fig_path, f"{config['dataset']}_{config['model_name']}_random_subset_{int(perc_sel*100)}pct_{active_suffix if active_suffix is not None else ''}_cm_mean_std.pdf"), dpi=600)
            plt.close()
            print(f"Saved confusion matrices to {cm_save_fig_path}")

        if config['save_random_subset_selected_features']:
            if perc_sel != 1:
                random_features_file = os.path.join(save_fig_path, f"{config['dataset']}_{config['model_name']}_{config['num_runs']}runs_random_subset_{int(perc_sel*100)}pct_selected_features.csv")
                random_features_df = pd.DataFrame(random_subset_features_list)
                random_features_df.to_csv(random_features_file)
                print(f"Saved random subsets of selected features to {random_features_file}")

        pd.DataFrame({
            'Batch Number': batch_numbers,
            'Accuracy': accuracy_values,
            'AUC': auc_values,
            'PR AUC': pr_auc_values,
            'Balanced Accuracy': b_acc_values,
            'Macro F1 Score': f1_values,
            'Macro Precision': prec_values,
            'Macro Recall': rec_values,
            'MCC': mcc_values
        }).to_csv(os.path.join(save_fig_path, f"{config['dataset']}_{config['model_name']}_{config['num_runs']}runs_random_features_{int(perc_sel*100)}pct_{active_suffix if active_suffix is not None else 'all_features'}_extended_results.csv"), index=False)   

        print(f"\n\nExtended results are saved in a CSV file at {save_fig_path}")

        print(f"\n\nThe experiment was done with {int(perc_sel*(data_final.shape[1]-1))} ({perc_sel*100:.3f}%) {active_suffix} features randomly selected over {len(batch_numbers)} runs")
        print(f"Accuracy: {np.mean(accuracy_values):.4f} +/- {np.std(accuracy_values):.4f}")
        print(f"AUC     : {np.mean(auc_values):.4f} +/- {np.std(auc_values):.4f}")
        print(f"PR AUC  : {np.mean(pr_auc_values):.4f} +/- {np.std(pr_auc_values):.4f}")
        print(f"Balanced Accuracy: {np.mean(b_acc_values):.4f} +/- {np.std(b_acc_values):.4f}")
        print(f"Macro F1 Score: {np.mean(f1_values):.4f} +/- {np.std(f1_values):.4f}")
        print(f"Macro Precision: {np.mean(prec_values):.4f} +/- {np.std(prec_values):.4f}")
        print(f"Macro Recall  : {np.mean(rec_values):.4f} +/- {np.std(rec_values):.4f}")   
        print(f"MCC     : {np.mean(mcc_values):.4f} +/- {np.std(mcc_values):.4f}")

        print("-"*25)
        print(f'\n\nExecution time for Model {config["model_name"]} and dataset {config["dataset"]} with {int(perc_sel*(data_final.shape[1]-1))} ({perc_sel*100:.1f}%) features over {len(batch_numbers)} runs: {end - start:.2f} seconds')


def evaluate_model_with_random_features(data_final, subset_size, test_df = pd.DataFrame()):
        '''
        Evaluates the model's performance using randomly selected features.

        Parameters:
                data_final (pandas.DataFrame): The final dataset containing features and target variable.
                subset_size (int): Number of features to select randomly in each run.
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                None
        '''

        def spinner(message):
            """A simple spinner to indicate progress."""
            while not done:
                for char in "|/-\\":
                    print(f"\r{message} {char}", end="", flush=True)
                    time.sleep(0.1)
            print("\r" + " " * (len(message) + 2) + "\r", end="", flush=True)


        print(f"\n\nStarting evaluation of model's performance with {subset_size} features randomly selected from {data_final.shape[1]-1} {active_suffix} features", flush=True)

        # Start the timer for checking model accuracy with selected features
        start = time.perf_counter()

        done = False
        if config['show_progress_bar']:
            t = threading.Thread(target=spinner,  args=(f"Evaluating model on {subset_size} features...",))
            t.start()

        if config['train_test_sep']:
            
            batch_numbers, accuracy_values, auc_values, pr_auc_values, b_acc_values, f1_values, prec_values, rec_values, mcc_values, cm_norm_values, cm_counts_values, random_subset_features_list = random_features_model_accuracy(
                                        df = data_final,
                                        batch_size = int(subset_size),
                                        debug = 0,
                                        train_test_sep = 1, 
                                        test_df = test_df
                                        )
                                        
        else:
            batch_numbers, accuracy_values, auc_values, pr_auc_values, b_acc_values, f1_values, prec_values, rec_values, mcc_values, cm_norm_values, cm_counts_values, random_subset_features_list = random_features_model_accuracy(
                                        df = data_final,
                                        batch_size = int(subset_size),
                                        debug = 0,
                                        )

        done = True
        if config['show_progress_bar']:
            t.join()

        # End the timer for checking model accuracy with selected features
        end = time.perf_counter()

        if config['num_runs'] == len(batch_numbers):
            print(f"All {config['num_runs']} runs completed successfully.")
        else:
            print(f"Only {len(batch_numbers)} out of {config['num_runs']} runs completed.")

        if config['save_random_subset_selected_features']:
            if subset_size != (data_final.shape[1]-1):
                random_features_file = os.path.join(save_fig_path, f"{config['dataset']}_{config['model_name']}_{config['num_runs']}runs_{active_suffix}_{subset_size}_random_subset_selected_features.csv")
                random_features_df = pd.DataFrame(random_subset_features_list)
                random_features_df.to_csv(random_features_file)
                print(f"Saved random subsets of selected features to {random_features_file}")

        pd.DataFrame({
                'Batch Number': batch_numbers,
                'Accuracy': accuracy_values,
                'AUC': auc_values,
                'PR AUC': pr_auc_values,
                'Balanced Accuracy': b_acc_values,
                'Macro F1 Score': f1_values,
                'Macro Precision': prec_values,
                'Macro Recall': rec_values,
                'MCC': mcc_values
        }).to_csv(os.path.join(save_fig_path, f"{config['dataset']}_{config['model_name']}_{config['num_runs']}runs_{active_suffix}_{subset_size}_random_features_extended_results.csv"), index=False)   

        print(f"\n\nThe experiment was done with {subset_size} {active_suffix} features randomly selected over {len(batch_numbers)} runs")
        print(f"Accuracy: {np.mean(accuracy_values):.4f} +/- {np.std(accuracy_values):.4f}")
        print(f"AUC     : {np.mean(auc_values):.4f} +/- {np.std(auc_values):.4f}")
        print(f"PR AUC  : {np.mean(pr_auc_values):.4f} +/- {np.std(pr_auc_values):.4f}")
        print(f"Balanced Accuracy: {np.mean(b_acc_values):.4f} +/- {np.std(b_acc_values):.4f}")
        print(f"Macro F1 Score: {np.mean(f1_values):.4f} +/- {np.std(f1_values):.4f}")
        print(f"Macro Precision: {np.mean(prec_values):.4f} +/- {np.std(prec_values):.4f}")
        print(f"Macro Recall  : {np.mean(rec_values):.4f} +/- {np.std(rec_values):.4f}")   
        print(f"MCC     : {np.mean(mcc_values):.4f} +/- {np.std(mcc_values):.4f}")

        print("\n\n")
        print("-"*25)
        print(f'\n\nExecution time for Model {config["model_name"]} and dataset {config["dataset"]} with {subset_size} {active_suffix} features over {len(batch_numbers)} runs: {end - start:.4f} seconds')
