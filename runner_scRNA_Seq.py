import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import fs_utils



import threading, itertools, os, sys, time

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split 
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from pathlib import Path

target = fs_utils.config['target']

script_start_time = time.time()

paths = {
    "filepath": fs_utils.config["filepath"],
    "hvg_file_path": fs_utils.config["hvg_file_path"],
    "hk_genes_file_path": fs_utils.config["hk_genes_file_path"],
    "union_cluster_genes_file_path": fs_utils.config["union_cluster_genes_file_path"],
}

missing = []

for name, p in paths.items():
    p = Path(p)
    if not p.exists():
        missing.append((name, p))

if missing:
    for name, p in missing:
        print(f"missing: {name} → {p}")
    raise FileNotFoundError("one or more required files are missing")
else:
    print("all required files exist 👍")

done = False
acc_full_feature, auc_full_feature = None, None

# if fs_utils.config['use_feature_ticks_ranges'] == fs_utils.config['use_feature_ticks_perc_ranges']:
    # raise ValueError("exactly one of use_feature_ticks_ranges or use_feature_ticks_perc_ranges must be true")

plot_fn = (
    fs_utils.plot_random_features_acc_auc
    if fs_utils.config['use_feature_ticks_ranges']
    else fs_utils.plot_perc_random_features_acc_auc
)


def spinner(msg="Running..."):
    is_tty = False
    try:
        tty = open('/dev/tty', 'w')
        is_tty = True
    except Exception:
        tty = sys.stdout
        is_tty = os.isatty(sys.stdout.fileno())

    # If no terminal is available (e.g. detached tmux), skip spinner entirely
    if not is_tty:
        return

    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        tty.write(f'\r{msg} {c}')
        tty.flush()
        time.sleep(0.1)

    # clear line when done
    tty.write('\r' + ' ' * (len(msg) + 4) + '\r')
    tty.flush()
    if tty is not sys.stdout:
        tty.close()

filepath = fs_utils.config["filepath"]

print(f"Loading dataset: {fs_utils.config['dataset']} from {filepath}")

# if fs_utils.config['dataset'] == 'TCGA' or fs_utils.config['dataset'] == 'Data_Bischoff2021_Lung' or fs_utils.config['dataset'] == 'Lung Single-cell RNA-Seq' \
#     or fs_utils.config['dataset'] == 'Data_Bischoff2021_Lung_cells' or fs_utils.config['dataset'] == 'Qian2020_Breast_annotated_with_cell_types' or \
#           fs_utils.config['dataset'] == 'Data_Wu2021_Breast' or fs_utils.config['dataset'] == 'Wu2021_Breast_annotated_with_cluster_numbers' or \
#             fs_utils.config['dataset'] == 'Wu2021_HVG1000_leiden0_02' or fs_utils.config['dataset'] == 'Wu2021_HVG1000_leiden0_5' or \
#                 fs_utils.config['dataset'] == 'Wu2021_HVG1000_leiden2_0' or fs_utils.config['dataset'] == 'Wu2021_with_HVG2000_leiden0_02_cluster_numbers' or \
#                     fs_utils.config['dataset'] == 'Wu2021_with_HVG2000_leiden0_5_cluster_numbers' \
#             or fs_utils.config['dataset'] == 'Wu2021_with_HVG2000_leiden2_0_cluster_numbers' or fs_utils.config['dataset'] == 'Wu2021_with_HVG3000_leiden0_02_cluster_numbers' \
#     or fs_utils.config['dataset'] == 'Wu2021_with_HVG3000_leiden0_5_cluster_numbers' or fs_utils.config['dataset'] == 'Wu2021_with_HVG3000_leiden2_0_cluster_numbers' \
#         or fs_utils.config['dataset'] == 'Wu2021_with_HVG5000_leiden2_0_cluster_numbers' or fs_utils.config['dataset'] == 'pbmc10k_1000hvgs_leiden0_02' \
#             or fs_utils.config['dataset'] == 'pbmc10k_1000hvgs_leiden0_5' or fs_utils.config['dataset'] == 'pbmc10k_1000hvgs_leiden2_0' \
#                 or fs_utils.config['dataset'] =='Qian2020_1000hvgs_leiden0_02' or fs_utils.config['dataset'] == 'Qian2020_1000hvgs_leiden0_5' \
#                     or fs_utils.config['dataset'] == 'Qian2020_1000hvgs_leiden2_0' or fs_utils.config['dataset'] == 'Bischoff2021_1000hvgs_leiden0_02' \
#                         or fs_utils.config['dataset'] == 'Bischoff2021_1000hvgs_leiden0_5' or fs_utils.config['dataset'] == 'Bischoff2021_1000hvgs_leiden2_0':
if  any(ds in fs_utils.config['dataset'] for ds in ['Wu', 'Qian', 'Bischoff', 'pbmc10k_celltypist_labels']):   
    data_final = pd.read_parquet(filepath)
    print(f'\n\nlast few columns before renaming the target column: {data_final.columns[-5:]}\n\n')

    try:
        data_final.rename(columns={'cell_type': 'type'}, inplace=True)
    except KeyError as e:
        pass

elif 'sergio' in fs_utils.config['dataset'].lower():
    X = pd.read_csv(filepath, index_col=0)

    y = pd.read_csv(f'../Data/Sergio/meta_{fs_utils.config['dataset']}.csv', index_col=0)

    data_final = X.copy()
    data_final[fs_utils.config["target"]] = y['Label'].values
    
elif 'h5ad' in filepath:
    import anndata

    adata = anndata.read_h5ad(filepath)

    X = adata.to_df()
    X = X.loc[:, ~X.columns.duplicated(keep='first')]

    y = adata.obs['cell_type']

    data_final = X.copy()
    data_final[fs_utils.config["target"]] = y.values

elif any(ds in fs_utils.config['dataset'] for ds in ['sc_10x', 'sc_celseq2', 'sc_dropseq', 'crc_GSE81861']):

    if fs_utils.config['dataset'] == 'crc_GSE81861':
        X = pd.read_csv(filepath, index_col=0)
        y = pd.read_csv(f'{filepath.replace(".count", ".metadata")}', index_col=0)['labels']    
    else:
        X = pd.read_csv(filepath, index_col=0).T
        y = pd.read_csv(f'{filepath.replace(".count", ".metadata")}', index_col=0)['cell_line_demuxlet']

    data_final = X.copy()
    data_final[fs_utils.config["target"]] = y.values

else:
    data_final = pd.read_csv(filepath, index_col=0)

# try renaming cluster column to 'type'
try:
    data_final.rename(columns={'cluster_num': 'type'}, inplace=True)
except KeyError as e:
    print("\n\ncluster_num column not found.")
try:
    data_final.rename(columns={'leiden_0.02': 'type'}, inplace=True)
    #print(f'\n\nlast few columns after rename: {data_final.columns[-5:]}\n\n')
except KeyError as e:
    print("\n\nleiden_0.02 column not found.")
    raise e
try:
    data_final.rename(columns={'leiden_0.5': 'type'}, inplace=True)
    #print(f'\n\nlast few columns after rename: {data_final.columns[-5:]}\n\n')
except KeyError as e:   
    print("\n\nleiden_0.5 column not found.")
    raise e
try:
    data_final.rename(columns={'leiden_2.0': 'type'}, inplace=True)
    #print(f'\n\nlast few columns after rename: {data_final.columns[-5:]}\n\n')
except KeyError as e:
    print("\n\nleiden_2.0 column not found.")
    raise e
try:
    data_final.rename(columns={'cell_line_demuxlet': 'type'}, inplace=True)
    #print(f'\n\nlast few columns after rename: {data_final.columns[-5:]}\n\n')
except KeyError as e:
    print("\n\ncell_line_demuxlet column not found.")
    raise e

try:
    data_final.rename(columns={'celltypist_cell_label_coarse': 'type'}, inplace=True)
    #print(f'\n\nlast few columns after rename: {data_final.columns[-5:]}\n\n')
except KeyError as e:
    print("\n\celltypist_cell_label_coarse column not found.")
    raise e


print(f"Initial dataset shape: {data_final.shape}", flush=True)
print(f'\n\nlast few columns after renaming the target column: {data_final.columns[-5:]}\n\n')

if dataset := fs_utils.config['dataset'] in ['Data_Bischoff2021_Lung', 'Data_Bischoff2021_Lung_cells', 'Lung Single-cell RNA-Seq', 'Data_Wu2021_Breast']:
    try:
        data_final.drop('sample', axis=1, inplace=True)
    except KeyError as e:
        pass

    try:
        data_final.rename(columns={'cell_type': 'type'}, inplace=True)
    except KeyError as e:
        pass

try:
    data_final.rename(columns={'label': 'type'}, inplace=True)
except KeyError as e:
        pass


if data_final['type'].isnull().sum() > 0:
    data_final.dropna(subset=['type'], inplace=True)

print(f"{data_final['type'].isnull().sum()} rows with null type values removed.")
print(f"Final dataset shape used in this set of experiments: {data_final.shape}", flush=True)


# cols = data_final.columns

# bad_cols = cols[
#     cols.isna()
#     | (cols.astype(str).str.strip() == "")
#     | (cols.astype(str).str.lower() == "nan")
# ]

# print(f"Bad columns found: {bad_cols.to_list()}", flush=True)

# print(f'data_final.columns[data_final.columns.isna()]: {data_final.columns[data_final.columns.isna()]}')

# print(f'data_final.loc[:, data_final.columns.isna()].isna().all(): {data_final.loc[:, data_final.columns.isna()].isna().all()}')

data_final = data_final.loc[:, ~data_final.columns.isna()]

assert not data_final.columns.isna().any()
assert not (data_final.columns.astype(str).str.strip() == "").any()

y_train = data_final[target]
# encoding labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

data_final[target] = y_train

if fs_utils.config['train_test_sep']:
    print("As there is a separate test dataset, we need to load it now...", flush=True)
    
    if fs_utils.config['test_dataset'] == 'Arcene_test':
        # Arcene test dataset has no header
        # so we need to load it with header=None
        # and then rename the last column to 'type'
        # as it is the label column
        # and the first 10000 columns are features
        # and the last column is the label
        print("Loading Arcene test dataset...")
        test_df = pd.read_csv(fs_utils.config['test_filepath'], header=None)
        test_df.rename(columns={10000: 'type'}, inplace=True)
        
    elif fs_utils.config['test_dataset'] == 'Gisette_validation' or fs_utils.config['test_dataset'] == 'Gisette':
        print("Loading Gisette test dataset...")
        X_test = pd.read_csv(fs_utils.config['test_filepath'], sep=r'\s+', engine='python', header=None)
        if fs_utils.config['test_dataset'] == 'Gisette_validation':
            y_test = pd.read_csv('../Data/NIPS_2003/gisette/GISETTE/gisette_valid.labels', header=None)
        else:
            y_test = pd.read_csv('../Data/NIPS_2003/gisette/GISETTE/gisette_train.labels', header=None)

        X_test.columns = [f'feat_{i}' for i in range(X_test.shape[1])]
        y_test.columns = ['type']

        test_df = pd.concat([X_test, y_test], axis=1)

    elif fs_utils.config['test_dataset'] == 'Madelon_Validation':
        print("Loading Madelon test dataset...")

        X_test = pd.read_csv(fs_utils.config['test_filepath'], sep=r'\s+', engine='python', header=None)
        y_test = pd.read_csv('../Data/NIPS_2003/Madelon/madelon/madelon_valid.labels',header=None)

        X_test.columns = [f'feat_{i}' for i in range(X_test.shape[1])]
        y_test.columns = ['type']

        test_df = pd.concat([X_test, y_test], axis=1)

    elif any(ds in fs_utils.config['dataset'] for ds in ['Wu', 'Qian', 'Bischoff']): 
        print(f"Loading test dataset {fs_utils.config['test_dataset']}...")
        test_df = pd.read_parquet(fs_utils.config['test_filepath'])
        try:
            test_df.drop('sample', axis=1, inplace=True)
        except KeyError as e:
            pass

        try:
            test_df.rename(columns={'cell_type': 'type'}, inplace=True)
        except KeyError as e:
            pass
        
        print(f'Test set has the shape: {test_df.shape}\n\n', flush=True)

    elif any(ds in fs_utils.config['test_dataset'] for ds in ['sc_10x', 'sc_celseq2', 'sc_dropseq']):
        X = pd.read_csv(fs_utils.config['test_filepath'], index_col=0).T
        
        y = pd.read_csv(f'{fs_utils.config['test_filepath'].replace(".count", ".metadata")}', index_col=0)['cell_line_demuxlet']

        test_df = X.copy()
        test_df[fs_utils.config["target"]] = y.values

    else:
        test_df = pd.read_csv(fs_utils.config['test_filepath'])
        

    try:
        test_df.set_index('samples', inplace=True)
    except KeyError as e:
        pass

    try:
        test_df.rename(columns={'label': 'type'}, inplace=True)
    except KeyError as e:
        pass


    if test_df[target].isnull().sum() > 0:
        print(f"{test_df[target].isnull().sum()} rows with null type values to be removed.")
        test_df.dropna(subset=[target], inplace=True)

    print(f"Test dataset shape used in this set of experiments: {test_df.shape}", flush=True)

    print("train unique:")
    print(sorted(data_final[target].unique()))

    print("\ntest unique:")
    print(sorted(test_df[target].unique()))

    # Ensure both train and test have same set of classes 
    y_train = data_final[target]
    y_test  = test_df[target]

    common_classes = set(y_train) & set(y_test)
    print(f'common classes: {common_classes}')
    assert common_classes != 0 

    data_final = data_final[data_final[target].isin(common_classes)]
    test_df    = test_df[test_df[target].isin(common_classes)]

    # y_train = data_final[target]
    y_test  = test_df[target]

    # encoding labels
    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # data_final[target] = y_train

    # Encode test labels using the label encoder already fitted on train 
    assert set(test_df[target]).issubset(set(le.classes_))
    
    test_df[target] = le.transform(test_df[target])
    
    print(f'classes in train dataset: {set(data_final[target])}')
    print(f'classes in test dataset: {set(test_df[target])}')
          
    # assert set(data_final[target]) == set(test_df[target])

    print(f'Test set has the shape: {test_df.shape}\n\n', flush=True)

    vc_test = test_df[target].value_counts()
    class_wise_dist = pd.DataFrame({
    "count": vc_test,
    "percentage": 100 * vc_test / vc_test.sum()
    })
    class_wise_dist.index = le.inverse_transform(class_wise_dist.index)
    print(class_wise_dist.index[class_wise_dist.index.isna()])

    assert not class_wise_dist.index.isna().any()
    class_wise_dist["percentage"] = class_wise_dist["percentage"].round(2)

    class_wise_dist = class_wise_dist.reset_index().rename(columns={"index": "class"})
    class_wise_dist["encoded_label"] = class_wise_dist["class"].map(label_mapping)
    class_wise_dist = class_wise_dist[["class", "encoded_label", "count", "percentage"]]

    print(f"\n\nClass-wise distribution for the test dataset {fs_utils.config['test_dataset']}:", flush=True)
    print(class_wise_dist)

    if fs_utils.config['use_only_hvgs']:
        class_wise_dist.to_csv(f'{fs_utils.save_fig_path}/{fs_utils.config["test_dataset"]}_class_wise_dist.csv')
        print(f"\n\nSaved class-wise distribtution to {fs_utils.save_fig_path}")

vc = data_final[target].value_counts()
class_wise_dist = pd.DataFrame({
    "count": vc,
    "percentage": 100 * vc / vc.sum()
    # "encoded_label": class_wise_dist.index
})
class_wise_dist.index = le.inverse_transform(class_wise_dist.index)
assert not class_wise_dist.index.isna().any()
class_wise_dist["percentage"] = class_wise_dist["percentage"].round(2)


class_wise_dist = class_wise_dist.reset_index().rename(columns={"index": "class"})
class_wise_dist["encoded_label"] = class_wise_dist["class"].map(label_mapping)
class_wise_dist = class_wise_dist[["class", "encoded_label", "count", "percentage"]]

print(f"\n\nClass-wise distribution for the full dataset {fs_utils.config['dataset']}:", flush=True)
print(class_wise_dist)
print(label_mapping)

if fs_utils.config['use_only_hvgs']:
    class_wise_dist.to_csv(f'{fs_utils.save_fig_path}/{fs_utils.config["dataset"]}_class_wise_dist.csv')
    print(f"\n\nSaved class-wise distribtution to {fs_utils.save_fig_path}")

print(f"\n\nFull Dataset {fs_utils.config['dataset']} at a glance:")
print("-"*25)
print("Number of samples: "+ str(data_final.shape[0]))
print("Number of features: "+str(data_final.shape[1]-1))
print("Number of classes: ", len(vc))
print("-"*25)
print("Class-wise distribution: ", (data_final[target].value_counts()))
print("-"*25)
print(data_final.shape)
print("-"*25, flush=True)



if fs_utils.config['train_test_sep']:
    print(f"\n\nTest dataset {fs_utils.config['test_dataset']} at a glance:")
    print("-"*25)
    print("Number of samples: "+ str(test_df.shape[0]))
    print("Number of features: "+str(test_df.shape[1]-1))
    print("Number of classes: ", len(vc_test))
    print("-"*25)
    print("Class-wise distribution: ", (test_df[fs_utils.config['target']].value_counts()))
    print("-"*25)
    print(test_df.shape)
    print("-"*25)

    # Ensure train and test df have same set of classes 
    train_classes = set(data_final[fs_utils.config['target']])
    test_classes  = set(test_df[fs_utils.config['target']])

    print("classes only in train:", train_classes - test_classes)
    print("classes only in test:",  test_classes - train_classes)

    # mask = y_test.isin(train_classes)
    # test_df = test_df[mask]
    # y_test  = y_test[mask]

    # Ensure train and test df have same set of columns 
    train_cols = set(data_final.columns)
    test_cols  = set(test_df.columns)

    matched = train_cols & test_cols
    train_only = train_cols - test_cols
    test_only  = test_cols - train_cols

    print(f"matched columns: {len(matched)}")
    print(f"features only in train:   {len(train_only)}")
    print(f"features only in test:    {len(test_only)}")

    print(f"features % overlap (wrt train): {100 * len(matched) / len(train_cols):.2f}%")
    print(f"features % overlap (wrt test):  {100 * len(matched) / len(test_cols):.2f}%")

    common_cols = data_final.columns.intersection(test_df.columns)

    data_final = data_final.loc[:, common_cols]
    test_df    = test_df.loc[:, common_cols]

    assert set(data_final.columns) == set(test_df.columns), "Train and test datasets should have the same set of columns"

try:
    le
except NameError:
    le = LabelEncoder()
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    data_final[target] = y_train

col = data_final.pop(fs_utils.config['target'])
data_final[fs_utils.config['target']] = col

if fs_utils.config['get_sparsity_summary']:
    print("\n\n Calculating sparsity summary...")

    summary_sparsity = fs_utils.sparsity_summary_df(data_final, fs_utils.config['target'])

    summary_sparsity.to_csv(f'./Plots/{fs_utils.config["dataset"]}/{fs_utils.config["dataset"]}_sparsity_summary.csv')

    print(f'{summary_sparsity} for dataset {fs_utils.config["dataset"]} saved successfully.')

if fs_utils.config['run_topk_fs_evaluation']:
    print("\n\nStarting evaluation of top-k feature selection methods...", flush=True)

    # Start the timer for checking model accuracy with all features
    start = time.perf_counter()

    done = False
    if fs_utils.config['show_progress_bar']:
        t = threading.Thread(target=spinner,  args=("Evaluating model on top-k feature set...",))
        t.start()

    summary = fs_utils.evaluate_topk_methods(data_final, label_col=fs_utils.config['target'], k_list=fs_utils.config['k_list'], outer_folds=5, n_jobs=4)
    print(summary[["method","k","mean_acc", "median_acc","std_acc", "mean_auc", "median_auc", "std_auc", "n_selected_median"]])

    if fs_utils.config['show_progress_bar']:
        done = True
        t.join()

    # End the timer for checking model accuracy with all features
    end = time.perf_counter()

    print("\n\nFinished evaluation of top-k feature selection methods...", flush=True)

    print(f'Execution time for Top-k feature selection methods evaluation for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]}: {end - start:.4f} seconds', flush=True)

if fs_utils.config['train_test_sep']:
    test_col = test_df.pop(fs_utils.config['target'])
    test_df[fs_utils.config['target']] = test_col

# if fs_utils.config['run_with_all_features'] or fs_utils.config['run_var_random_subsets_of_all_features'] or fs_utils.config['run_var_perc_random_subset_of_all_features'] :  # Verify the logic of this line
if fs_utils.config['run_with_all_features']:
    print("\n\nStarting evaluation of model's performance with all features", flush=True)

    # Start the timer for checking model accuracy with all features
    start = time.perf_counter()

    done = False
    if fs_utils.config['show_progress_bar']:
        t = threading.Thread(target=spinner,  args=("Evaluating model on full feature set...",))
        t.start()

    if fs_utils.config['train_test_sep']:   
        # If train-test separation is enabled, use the test_df for evaluation
        # accuracy, roc_auc_ovo, roc_auc_ovr, pr_auc
        acc_full_feature, roc_auc_ovo_full_feature, auc_full_feature, pr_auc_full_feature = fs_utils.check_model_acc_full_feature(df = data_final, 
                                                                    train_test_sep = 1, 
                                                                    test_df = test_df,
                                                                    debug=1
                                                                    )
    elif fs_utils.config['train_test_sep'] == False:
        acc_full_feature, roc_auc_ovo_full_feature, auc_full_feature, pr_auc_full_feature = fs_utils.check_model_acc_full_feature(df = data_final, 
                                                                                debug=0)
        

    if fs_utils.config['show_progress_bar']:
        done = True
        t.join()

    # End the timer for checking model accuracy with all features
    end = time.perf_counter()

    # Print the results

    print(f"\nAccuracy with all features: {acc_full_feature:.4f}")
    print(f"AUC with all features: {auc_full_feature:.4f}")
    print(f'Execution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with {data_final.shape[1]-1} features: {end - start:.4f} seconds', flush=True)

if fs_utils.config['run_sel_perc_random_subset_of_all_feature']:

    for perc_sel in fs_utils.config['sel_per_random_subset_size']:
        fs_utils.evaluate_model_with_perc_random_features(data_final=data_final, use_active_suffix=0, perc_sel=perc_sel, test_df = test_df if fs_utils.config['train_test_sep'] else pd.DataFrame())

if fs_utils.config['run_var_random_subsets_of_all_features']:

    num_samples = data_final.shape[0]
    num_features = data_final.shape[1]-1
    num_classes = len(data_final[fs_utils.config['target']].value_counts())

    print(f"Checking various random subsets of all features({num_features})...")

    start = time.perf_counter()

    result_csv_file = fs_utils.get_result_random_sets(data_final, use_active_suffix=0)

    end = time.perf_counter()

    print(f'\n\nExecution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with various random subsets:{end - start:.4f} seconds')

    result_df = pd.read_csv(result_csv_file)


    # plot_fn(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=1, plot_auc=0, debug=0)

    # plot_fn(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=0, plot_auc=1, debug=0)

if fs_utils.config['run_var_perc_random_subset_of_all_features']:

    print("Checking various percentage random subsets of all features...")

    start = time.perf_counter()

    result_csv_file = fs_utils.get_result_random_sets(data_final, use_active_suffix=0)

    end = time.perf_counter()

    print(f'\n\nExecution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with various random subsets:{end - start:.4f} seconds')

    result_df = pd.read_csv(result_csv_file)

    num_samples = data_final.shape[0]
    num_features = data_final.shape[1]-1
    num_classes = len(data_final[fs_utils.config['target']].value_counts())

    fs_utils.plot_perc_random_features_acc_auc(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=1, plot_auc=0)

    fs_utils.plot_perc_random_features_acc_auc(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=0, plot_auc=1)


# Select genes based on the specified criteria

use_only_fs_list = fs_utils.config['use_only_fs_list']
use_only_hvgs = fs_utils.config['use_only_hvgs']
use_only_non_hvgs = fs_utils.config['use_only_non_hvgs']
use_only_mt_genes = fs_utils.config['use_only_mt_genes']
use_only_hk_genes = fs_utils.config['use_only_hk_genes']
use_union_of_cluster_genes = fs_utils.config['use_union_of_cluster_genes']
use_uc_plus = fs_utils.config['use_uc_plus']

all_genes = data_final.columns[:-1]  # Exclude target column

genes_to_use = None
gene_group = None

print("**"*25)

if use_only_fs_list:
    print("Using only FS selected features")
    gene_group = "FS"
    genes_to_use = pd.read_csv(fs_utils.config['fs_list_file_path'])
    genes_to_use = genes_to_use['DUBStepR'].dropna().tolist()

elif use_only_hvgs:
    print("Using only HVGs")
    gene_group = "HVG"
    genes_to_use = pd.read_csv(fs_utils.config['hvg_file_path'])
    genes_to_use = genes_to_use.gene.to_list()

elif use_only_non_hvgs:
    print("Using only non-HVGs")
    gene_group = "Non-HVG"
    genes_to_exclude = pd.read_csv(fs_utils.config['hvg_file_path'])
    genes_to_exclude = genes_to_exclude.gene.to_list()
    genes_to_use = all_genes.difference(genes_to_exclude)

elif use_only_mt_genes:
    print("Using only MT genes")
    gene_group = "MT"
    genes_to_use = data_final.columns[data_final.columns.str.startswith('MT-')].to_list()

elif use_only_hk_genes:
    print("Using only HK genes")
    gene_group = "HK"
    hk_df = pd.read_csv(fs_utils.config['hk_genes_file_path'], sep=';')
    
    # For all datasets, make sure that the gene types are matching with HK gene types - HK file has Ensembl, HUGO, Refseq and CCDS.ID - choose accordingly
    
    if 'ENS' in all_genes[0]:
        hk_ensg_df = pd.read_csv('../Data/Housekeeping_GenesHuman_ENSTIDs.csv')
        hk_genes = hk_ensg_df['ENST_ID'].dropna().unique()
    else:
        hk_genes = hk_df['Gene.name'].dropna().unique()
    
    print(f"Total HK genes from the file: {len(hk_genes)}")
    print(f"Total genes in the dataset: {len(all_genes)}")
    print(f'\n\nlast few genes in the dataset: {all_genes[-5:]}\n\n')
    print(f'\n\nlast few HK genes: {hk_genes[-5:]}\n\n')

    genes_to_use = sorted(set(all_genes.str.upper()).intersection(set(hk_genes)))

    # pd.DataFrame(genes_to_use).to_csv(fs_utils.save_fig_path+'hk_genes.csv', index=False)

elif use_union_of_cluster_genes:
    print("Using only union of cluster genes")
    gene_group = "UC"
    cluster_genes = pd.read_csv(fs_utils.config['union_cluster_genes_file_path'])
    genes_to_use = list(set(cluster_genes.values.ravel()))
    # fs_utils.plot_umap(genes_to_use, data_final, test_df if fs_utils.config['train_test_sep'] else None)

elif use_uc_plus:
    rng = np.random.default_rng(42)
    total_target = 250
    print("Using union of cluster genes + HVG")
    gene_group = "UC+HVG"
    cluster_genes = pd.read_csv(fs_utils.config['union_cluster_genes_file_path'])
    fs_genes = pd.read_csv(fs_utils.config['hvg_file_path'])
    # fs_genes = fs_genes.DUBStepR.to_list()
    cluster_genes_set = set(cluster_genes.values.ravel())
    fs_genes_set = set(fs_genes)

    n_random_needed = total_target - len(cluster_genes_set)
     
    # remove cluster genes from candidate pool
    candidate_pool = list(fs_genes_set - cluster_genes_set)

    assert n_random_needed <= len(candidate_pool) 
    
    random_genes = rng.choice(list(candidate_pool), size=n_random_needed, replace=False)
    genes_to_use = sorted(cluster_genes_set.union(set(random_genes)))

    assert len(genes_to_use) == total_target

else:
    # genes_to_use = all_genes
    print("No specific gene set specified.")

gene_meta = pd.read_parquet('gene_metadata.parquet')

# remove version suffix just in case (safe even if none exist)
gene_meta["ensembl_id"] = gene_meta["ensembl_id"].str.split(".").str[0]

# remove duplicate symbols
gene_meta = gene_meta.drop_duplicates(subset=["gene_symbol"])

# build dictionary
symbol_to_ensg = dict(
    zip(gene_meta["gene_symbol"], gene_meta["ensembl_id"])
)

ensg_to_symbol = dict(
    zip(gene_meta["ensembl_id"], gene_meta["gene_symbol"])
)

def detect_gene_type(cols):
    cols = pd.Index(cols).astype(str)
    ens_fraction = cols.str.startswith("ENSG").mean()
    
    print(f"Fraction of ENSEMBL IDs: {ens_fraction:.2f}")
    if ens_fraction > 0.8:
        return "ensembl"
    elif ens_fraction < 0.2:
        return "hugo"
    else:
        return "mixed"

if use_uc_plus or use_only_fs_list or use_only_hvgs or use_only_non_hvgs or use_only_mt_genes or use_only_hk_genes or use_union_of_cluster_genes:

    print(f'selected genes columns: {genes_to_use[:10]}')
    type1 = detect_gene_type(genes_to_use) 

    print(f'dataset columns: {data_final.columns[:10]}')
    type2 = detect_gene_type(data_final.columns)

    print(f"Selected gene list type: {type1}"
                f"\nDataset columns type: {type2}")

    if type1 != type2:
        print(f"Gene ID type mismatch detected between the selected gene list and the dataset columns.")
        print(f"Selected gene list type: {type1}"
                f"\nDataset columns type: {type2}")
        
        if type1 == 'hugo' and type2 == 'ensembl':
            print('Converting selected genes types from HUGO to ENSEMBLE')
            genes_to_use = pd.Index(genes_to_use).map(symbol_to_ensg)
        elif type1 == 'ensembl' and type2 == 'hugo':
            print('Converting selected genes types from ENSEMBLE to HUGO')
            genes_to_use = pd.Index(genes_to_use).map(ensg_to_symbol)

        # Check again 
        type1 = detect_gene_type(genes_to_use) 
        type2 = detect_gene_type(data_final.columns)

        if type1 != type2:
            print(f"Gene ID type mismatch detected between the selected gene list and the dataset columns.")
            print(f"Selected gene list type: {type1}"
                    f"\nDataset columns type: {type2}")    
            raise ValueError(f"Gene ID type mismatch: {type1} vs {type2}")
    
    y = data_final[fs_utils.config['target']]
    try:
        X = data_final[genes_to_use]
    except KeyError:
        missing = set(genes_to_use) - set(data_final.columns)

        print(f'Number of genes from the selected list(size: {len(genes_to_use)}) not found in the dataframe(size: {len(data_final.columns)-1}) = {len(missing)}')
        print(f'Not all {fs_utils.active_suffix if fs_utils.active_suffix else ""} genes were found, taking intersection...')
        common_genes = data_final.columns.intersection(genes_to_use)
        # common_genes.to_series().to_csv(f"{fs_utils.save_fig_path}/common_genes.txt", index=False, header=False)
        print(f'Found {len(common_genes)} common genes...')
        
        if common_genes.empty:
            print(f"No common genes found between the selected gene list and the dataset columns.")
            print(f"Total genes in the dataset: {len(all_genes)}")
            print(f'last few genes in the dataset: {data_final.columns[-5:]}')
            print(f'\n\nlast few selected genes: {genes_to_use[-5:]}')

            raise ValueError("No common genes found between the selected gene list and the dataset columns.")
        
        X = data_final[common_genes]

        print(f'some selected genes: {common_genes[:10]}')

    print("**"*25)
    print(f'Using {X.shape[1]} {gene_group} features for the analysis')    
    print("**"*25)
    filtered_genes_df = X.copy()
    filtered_genes_df[y.name] = y.values

    # if use_union_of_cluster_genes:
        # fs_utils.plot_umap(genes_to_use, data_final, test_df if fs_utils.config['train_test_sep'] else None)


# # Check NULL model

if fs_utils.config['run_null_model'] and fs_utils.config['use_only_hvgs']:

    print("Running NULL model evaluation with permuted labels...", flush=True)

    accs = []
    bal_accs = []
    cm_norm_values = []
        
    if genes_to_use is not None:

        print(f'Using selected genes ({len(filtered_genes_df.columns) - 1}) for NULL model evaluation.', flush=True)
        y = filtered_genes_df[fs_utils.config["target"]]  
        X = filtered_genes_df.drop(columns=[fs_utils.config['target']])

    elif genes_to_use is None:
        print(f"Using all genes ({data_final.shape[1] - 1}) for NULL model evaluation.", flush=True)
        y = data_final[fs_utils.config['target']]
        X = data_final.drop(columns=[fs_utils.config['target']])

    class_labels = np.unique(y)

    rng = np.random.default_rng(42)

    clf = RandomForestClassifier(random_state=42)

    num_runs = fs_utils.config['num_runs']

    run = 0
    for i in rng.choice(100, num_runs, replace=False):

        print(f"Permutation iteration with random_state={i}...run {run+1}/{num_runs}", flush=True)

        y_permuted = shuffle(y, random_state=i)

        # Xtr, Xte, ytr, yte = train_test_split(X, y_permuted, test_size=0.2, random_state=i, stratify=y_permuted)

        if fs_utils.config['train_test_sep'] != 1:
            Xtr, Xte, ytr, yte = train_test_split(X, y_permuted, test_size=0.2, random_state=i, stratify=y_permuted)

        elif fs_utils.config['train_test_sep'] == 1:
        
            print(f'Null Model: Training on permuted TRAIN labels in {fs_utils.config['dataset']}, evaluated on REAL test set {fs_utils.config['test_dataset']}')
            test_df = test_df[filtered_genes_df.columns]

            Xtr = X
            ytr = y_permuted

            yte = test_df[target]
            Xte = test_df.drop(columns=[target])


        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        cm_norm = confusion_matrix( 
                yte,
                ypred,
                labels=class_labels,
                normalize="true"        # row-wise %
        )

        # print("Confusion Matrix (normalized):")
        # print(cm_norm)

        cm_norm_values.append(cm_norm)

        acc = accuracy_score(yte, ypred)
        bal_acc = balanced_accuracy_score(yte, ypred)
        
        accs.append(acc)
        bal_accs.append(bal_acc)

        print(f"for random state {i} run {run+1}/{num_runs} - accuracy: {accs[-1]:.2f}, Balanced accuracy: {bal_accs[-1]:.2f}", flush=True)
        run += 1
        
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
            annot[i, j] = f"{cm_mean[i,j]:.4f}\n ± {cm_std[i,j]:.2f}"

    num_runs, n_classes, _ = cm_array.shape
    if n_classes <= 10:
        annot = annot
    else:
        annot = None

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

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if fs_utils.config['train_test_sep']:
        plt.title(f"Train on permuted labels in {fs_utils.config['dataset']} Test on {fs_utils.config['test_dataset']} \n Null Model Confusion Matrix (mean ± std over {num_runs} runs)")
    else:    
        plt.title(f"Dataset: {fs_utils.config['dataset']} \n Null Model Confusion Matrix (mean ± std over {num_runs} runs)")

    plt.savefig(os.path.join(fs_utils.save_fig_path, f"{fs_utils.config['dataset']}_{fs_utils.config['model_name']}_{num_runs}_null_model_confusion_matrix_mean_std.pdf"), dpi=600)
    plt.close()
    print(f"Saved confusion matrix (mean ± std over {num_runs} runs) to {fs_utils.save_fig_path}")
    
    # accuracy stats
    acc_means = np.mean(accs)
    acc_stds = np.std(accs)

    balanced_acc_means = np.mean(bal_accs)
    balanced_acc_stds = np.std(bal_accs)

    print("null accuracy summary:")
    print(f"mean: {acc_means:.2f}")
    print(f"std : {acc_means:.2f}")
    
    print("\n\n")
    print("========================================")
    print("null balanced accuracy summary:")
    print(f"mean: {balanced_acc_means:.2f}")
    print(f"std : {balanced_acc_means:.2f}")
    print("========================================")

    print("Finished NULL model evaluation.")
    print('Dataset used:', fs_utils.config['dataset'])
    print('Model used:', clf.__class__.__name__)
    print('Number of features used:', X.shape[1])

if fs_utils.config['run_random_train_test_split']:
    if fs_utils.config['train_test_sep']:
        print("As there is a separate test dataset, skipping random train-test split evaluation...")
        if genes_to_use is not None:
            print(f'Using {len(genes_to_use)} {fs_utils.active_suffix if fs_utils.active_suffix else "selected"} genes for random train-test split evaluation.', flush=True)
            accuracy, roc_auc_ovo, roc_auc_ovr, pr_auc = fs_utils.check_model_acc_full_feature(df = filtered_genes_df, train_test_sep=1, test_df=test_df[filtered_genes_df.columns])
        else:
            print(f"Using all genes for random train-test split evaluation.", flush=True)
            accuracy, roc_auc_ovo, roc_auc_ovr, pr_auc = fs_utils.check_model_acc_full_feature(df = data_final, train_test_sep=1, test_df=test_df[data_final.columns])
        print("\n\n")
        print("=="*25)
        print(f"Using 100 pct {fs_utils.config['dataset']} as train and {fs_utils.config['test_dataset']} as the test set for evaluation...")
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC AUC OVO: {roc_auc_ovo:.4f}")
        print(f"ROC AUC OVR: {roc_auc_ovr:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print("=="*25)
        
    else:
        if genes_to_use is not None:
            print(f'Using {len(genes_to_use)} {fs_utils.active_suffix if fs_utils.active_suffix else "selected"} genes for random train-test split evaluation.', flush=True)
            fs_utils.run_random_train_test_split(filtered_genes_df)
        else:
            print(f"Using all genes for random train-test split evaluation.", flush=True)
            fs_utils.run_random_train_test_split(data_final)

# Evaluate model with the all selected features

if fs_utils.config['run_all_selected_features'] and (use_uc_plus or use_only_fs_list or use_only_hvgs or use_only_non_hvgs or use_only_mt_genes or use_only_hk_genes or use_union_of_cluster_genes):
    print(f'{len(genes_to_use)} {gene_group} genes to be used for model evaluation.')

    # Start the timer for checking model accuracy with selected features

    start = time.perf_counter()

    done = False
    if fs_utils.config['show_progress_bar']:
        t = threading.Thread(target=spinner,  args=(f"Evaluating model on {filtered_genes_df.shape[1]-1} selected features...",))
        t.start()

    acc_full_feature, roc_auc_ovo_full_feature, auc_full_feature, pr_auc_full_feature = fs_utils.check_model_acc_full_feature(df = filtered_genes_df, debug=0)

    if fs_utils.config['show_progress_bar']:
        done = True
        t.join()

    # End the timer for checking model accuracy with selected features
    end = time.perf_counter()

    # Print the results

    print(f"\nAccuracy with selected features: {acc_full_feature:.4f}")
    print(f"ROC AUC OVO with selected features: {roc_auc_ovo_full_feature:.4f}")
    print(f"ROC AUC OVR with selected features: {auc_full_feature:.4f}")
    print(f"PR AUC with selected features: {pr_auc_full_feature:.4f}")
    print(f'Execution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with {filtered_genes_df.shape[1]-1} selected features: {end - start:.4f} seconds')

#Check with pre defined random subsets of the selected features

if fs_utils.config['run_random_subsets_selected_features'] and (use_uc_plus or use_only_fs_list or use_only_hvgs or use_only_non_hvgs or use_only_hk_genes or use_union_of_cluster_genes):

    print("Checking various random subsets of the selected features...")

    start = time.perf_counter()

    if fs_utils.config['train_test_sep']:
        assert test_df is not None, "Test dataframe should not be None when train_test_sep is True"
        assert fs_utils.config['test_dataset'] is not None, "Test dataset name should not be None when train_test_sep is True"
        assert fs_utils.config['test_filepath'] is not None, "Test dataset filepath should not be None when train_test_sep is True"
        # assert filtered_genes_df.columns.equals(test_df.columns), "Train and test dataframes should have the same columns when train_test_sep is True"

        result_csv_file = fs_utils.get_result_random_sets(filtered_genes_df, train_test_sep=1, test_df=test_df[filtered_genes_df.columns])
    else:
        result_csv_file = fs_utils.get_result_random_sets(filtered_genes_df)

    end = time.perf_counter()

    print(f'\n\nExecution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with various random subsets:{end - start:.4f} seconds')

    result_df = pd.read_csv(result_csv_file)

    num_samples = filtered_genes_df.shape[0]
    num_features = filtered_genes_df.shape[1]-1
    num_classes = len(filtered_genes_df[fs_utils.config['target']].value_counts())

    if acc_full_feature is None or auc_full_feature is None:
        if fs_utils.config['train_test_sep']:
            assert test_df is not None, "Test dataframe should not be None when train_test_sep is True"
            acc_full_feature, roc_auc_ovo_full_feature, auc_full_feature, pr_auc_full_feature = fs_utils.check_model_acc_full_feature(df = filtered_genes_df, train_test_sep=1, test_df=test_df[filtered_genes_df.columns], debug=1)
        else:
            acc_full_feature, roc_auc_ovo_full_feature, auc_full_feature, pr_auc_full_feature = fs_utils.check_model_acc_full_feature(df = filtered_genes_df, debug=0)

        print(f'=========with all selected features({num_features})')
        print(f"Model accuracy : {acc_full_feature:.4f}")
        print(f"Model ROC AUC OVO : {roc_auc_ovo_full_feature:.4f}")
        print(f"Model ROC AUC OVR : {auc_full_feature:.4f}")
        print(f"Model PR AUC : {pr_auc_full_feature:.4f}", flush=True)

    # plot_fn(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=1, plot_auc=0, debug=fs_utils.config['debug'])

    # plot_fn(num_samples, num_features, num_classes, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=0, plot_auc=1, debug=fs_utils.config['debug'])

#Check with various percentage random subsets of the selected features

if fs_utils.config['run_perc_random_subset_selected_features']:
    print("\n\nChecking various percentage random subsets of the selected features...")

    for perc_sel in fs_utils.config['perc_random_subset_selected_features_subsets']:
        print(f"\nEvaluating percentage random subset: {perc_sel}")
        print(f'Dataset: {fs_utils.config["dataset"]}, Model: {fs_utils.config["model_name"]}, shape: {filtered_genes_df.shape}', flush=True)
        print("-"*50)
        if fs_utils.config['train_test_sep']:
            fs_utils.evaluate_model_with_perc_random_features(data_final=filtered_genes_df, perc_sel=perc_sel, test_df=test_df[filtered_genes_df.columns])
        else:
            fs_utils.evaluate_model_with_perc_random_features(data_final=filtered_genes_df, perc_sel=perc_sel)

# Check with various random subsets of the selected features
if fs_utils.config['run_random_subset_selected_features']:
    print("Checking specified random subset sizes of the selected features...")
    
    for subset_size in fs_utils.config['random_subset_selected_features_subsets']:
        print(f"\nEvaluating random subsets of size: {subset_size}")
        print(f'Dataset: {fs_utils.config["dataset"]}, Model: {fs_utils.config["model_name"]}, shape: {filtered_genes_df.shape}')
        print("-"*50)

        fs_utils.evaluate_model_with_random_features(data_final=filtered_genes_df, subset_size = subset_size)

script_end_time = time.time()
total_script_time = script_end_time - script_start_time
print(f"\nTotal script execution time: {total_script_time:.4f} seconds", flush=True)
sys.exit(0)





















































if fs_utils.config['opt_model']:
    result_csv_path = f"./Plots/{fs_utils.config['dataset']}/{fs_utils.config['model_name']}/{fs_utils.config['dataset']}_{fs_utils.config['model_name']}_(O)_random_features_acc_auc_results.csv"
else:
    result_csv_path = f"./Plots/{fs_utils.config['dataset']}/{fs_utils.config['model_name']}/{fs_utils.config['dataset']}_{fs_utils.config['model_name']}_(D)_random_features_acc_auc_results.csv"


x_from_paper, y_from_paper = fs_utils.config['x_from_paper'], fs_utils.config['y_from_paper']
print(f"Selected features from paper: {x_from_paper}, Accuracy: {y_from_paper:.2f}")

fs_utils.get_area_above_curve(result_csv_path, x_from_paper, y_from_paper, debug=0)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


y = data_final[fs_utils.config["target"]]  
X = data_final.drop(columns=[fs_utils.config['target']])

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Define wrapper for logistic regression on feature subsets
class SubsetFeatureModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, feature_idx):
        self.base_estimator = base_estimator
        self.feature_idx = feature_idx

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X.iloc[:, self.feature_idx], y)
        return self

    def predict(self, X):
        return self.model_.predict(X.iloc[:, self.feature_idx])

    def predict_proba(self, X):
        return self.model_.predict_proba(X.iloc[:, self.feature_idx])


# for _ in range(5):
# Parameters
k = 55  # number of features per model
for n_models in [5, 7, 9, 11, 13]:  # ensemble sizes
    # Create random feature subsets
    rng = np.random.default_rng(42)
    # Create random feature subsets
    feature_indices_list = [rng.choice(X.shape[1], size=k, replace=False) for _ in range(n_models)]

    # Prepare estimators
    estimators = []
    for i, idx in enumerate(feature_indices_list):
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        # clf = RandomForestClassifier(random_state=42)
        estimators.append((f'model_{i}', SubsetFeatureModel(clf, idx)))

    # Voting classifier ensemble
    ensemble = VotingClassifier(estimators=estimators, voting='soft')

    # Cross-validation performance
    scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
    # scores.mean(), scores.std()

    print(f'with {n_models} RF models and {k} features each:')
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# ASSUME you already have:
# X : Feature matrix (NumPy array or DataFrame)
# y : Labels (1D array or Series)
# -------------------------------

# Step 1: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Define Base Classifiers
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Step 3: Define Ensembles
voting_ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)

stacking_ensemble = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True
)

# Step 4: Fit Models
voting_ensemble.fit(X_train, y_train)
stacking_ensemble.fit(X_train, y_train)

# Step 5: Predict
y_pred_vote = voting_ensemble.predict(X_test)
y_pred_stack = stacking_ensemble.predict(X_test)

# Step 6: Evaluate
vote_acc = accuracy_score(y_test, y_pred_vote)
stack_acc = accuracy_score(y_test, y_pred_stack)

vote_report = classification_report(y_test, y_pred_vote, output_dict=True)
stack_report = classification_report(y_test, y_pred_stack, output_dict=True)

# Step 7: Create Summary Table
results_df = pd.DataFrame({
    "Model": ["Voting Ensemble", "Stacking Ensemble"],
    "Accuracy": [vote_acc, stack_acc],
    "Precision (macro avg)": [vote_report['macro avg']['precision'], stack_report['macro avg']['precision']],
    "Recall (macro avg)": [vote_report['macro avg']['recall'], stack_report['macro avg']['recall']],
    "F1 Score (macro avg)": [vote_report['macro avg']['f1-score'], stack_report['macro avg']['f1-score']]
})

print("\nEnsemble Performance Summary:")
print(results_df)

print(np.intersect1d(X_train, X_test).shape)


train_rows = {tuple(row) for row in X_train}
test_rows = {tuple(row) for row in X_test}
overlap = train_rows & test_rows
print(f"Overlapping samples: {len(overlap)}")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV

# --- Random Feature Voting Ensemble ---
def random_feature_voting_ensemble(X, y, base_models, n_features=35, test_size=0.2, seed=42):
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
    
    predictions = []
    for model in base_models:
        feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
        clf = clone(model)
        clf.fit(X_train.iloc[:, feat_idx], y_train)
        preds = clf.predict_proba(X_test.iloc[:, feat_idx])
        predictions.append(preds)

    # Soft voting: average probabilities
    avg_preds = np.mean(predictions, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)

    acc = accuracy_score(y_test, final_preds)
    return acc, final_preds

# --- Random Feature Stacking Ensemble ---
def random_feature_stacking_ensemble(X, y, base_models, meta_model=None, n_features=35, test_size=0.2, seed=42):
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
    
    meta_features_train = []
    meta_features_test = []

    for model in base_models:
        feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
        clf = clone(model)
        clf.fit(X_train.iloc[:, feat_idx], y_train)

        meta_features_train.append(clf.predict_proba(X_train.iloc[:, feat_idx]))
        meta_features_test.append(clf.predict_proba(X_test.iloc[:, feat_idx]))

    # Stack predictions as new features
    X_meta_train = np.hstack(meta_features_train)
    X_meta_test = np.hstack(meta_features_test)

    # Define meta-model
    if meta_model is None:
        meta_model = LogisticRegressionCV(max_iter=1000)

    meta_model.fit(X_meta_train, y_train)
    final_preds = meta_model.predict(X_meta_test)

    acc = accuracy_score(y_test, final_preds)
    return acc, final_preds


int(np.round(X.shape[1]* 0.01))


def random_feature_ensemble():
    # Define base models
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=1)

    base_models = [lr, rf, xgb] * 3  # 9 total base models

    num_features = int(np.round(X.shape[1]* 0.01))
    # Run voting ensemble
    acc_voting, preds_voting = random_feature_voting_ensemble(X, y, base_models, n_features=num_features)
    print(f"Voting Ensemble Accuracy: {acc_voting:.4f}")

    # Run stacking ensemble
    acc_stacking, preds_stacking = random_feature_stacking_ensemble(X, y, base_models, n_features=num_features)
    print(f"Stacking Ensemble Accuracy: {acc_stacking:.4f}")

# # CV ensemble


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np

def cross_validated_random_feature_ensemble(X, y, base_models, n_features=35, n_splits=5, seed=42, mode='voting'):
    """
    Cross-validated evaluation of random-feature ensemble (voting or stacking).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        base_models (list): List of scikit-learn base models.
        n_features (int): Number of features for each model.
        n_splits (int): Number of CV folds.
        seed (int): Random seed.
        mode (str): 'voting' or 'stacking'.

    Returns:
        Tuple: (mean accuracy, std accuracy, list of fold accuracies)
    """
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        predictions = []

        for model in base_models:
            feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
            clf = clone(model)
            clf.fit(X_train.iloc[:, feat_idx], y_train)

            if mode == 'voting':
                predictions.append(clf.predict_proba(X_test.iloc[:, feat_idx]))
            elif mode == 'stacking':
                predictions.append((clf.predict_proba(X_train.iloc[:, feat_idx]), clf.predict_proba(X_test.iloc[:, feat_idx])))

        if mode == 'voting':
            avg_preds = np.mean(predictions, axis=0)
            final_preds = np.argmax(avg_preds, axis=1)

        elif mode == 'stacking':
            meta_train = np.hstack([p[0] for p in predictions])
            meta_test = np.hstack([p[1] for p in predictions])

            meta_model = LogisticRegressionCV(max_iter=1000)
            meta_model.fit(meta_train, y_train)
            final_preds = meta_model.predict(meta_test)

        acc = accuracy_score(y_test, final_preds)
        accs.append(acc)

    return accs

# # Ensemble with plots for ROC and PR 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegressionCV

def cross_validated_random_feature_ensemble_with_plots(X, y, base_models, n_features=35, n_splits=5, seed=42, mode='voting'):
    """
    CV evaluation of random-feature ensemble (voting or stacking) with metrics and plots.

    Returns:
        dict: {
            'fold_accuracies': list,
            'mean_accuracy': float,
            'std_accuracy': float,
            'auc': float,
            'average_precision': float,
            'confusion_matrix': np.ndarray
        }
    """
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    all_y_true = []
    all_y_probs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        predictions = []

        for model in base_models:
            feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
            clf = clone(model)
            clf.fit(X_train.iloc[:, feat_idx], y_train)

            if mode == 'voting':
                predictions.append(clf.predict_proba(X_test.iloc[:, feat_idx]))
            elif mode == 'stacking':
                predictions.append((clf.predict_proba(X_train.iloc[:, feat_idx]),
                                    clf.predict_proba(X_test.iloc[:, feat_idx])))

        if mode == 'voting':
            avg_preds = np.mean(predictions, axis=0)
            final_preds = np.argmax(avg_preds, axis=1)
            final_probs = avg_preds

        elif mode == 'stacking':
            meta_train = np.hstack([p[0] for p in predictions])
            meta_test = np.hstack([p[1] for p in predictions])
            meta_model = LogisticRegressionCV(max_iter=1000)
            meta_model.fit(meta_train, y_train)
            final_preds = meta_model.predict(meta_test)
            final_probs = meta_model.predict_proba(meta_test)

        accs.append(accuracy_score(y_test, final_preds))
        all_y_true.append(y_test)
        all_y_probs.append(final_probs)

    # concat all folds
    y_true = np.concatenate(all_y_true)
    y_probs = np.concatenate(all_y_probs)

    if y_probs.shape[1] == 2:
        # binary case, same as before
        y_prob_pos = y_probs[:,1]
        auc = roc_auc_score(y_true, y_prob_pos)
        ap = average_precision_score(y_true, y_prob_pos)
        # plots
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        plt.figure(); plt.plot(fpr, tpr); plt.show()
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
        plt.figure(); plt.plot(recall, precision); plt.show()
    else:
        # multiclass
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        ap = average_precision_score(y_true, y_probs, average='macro')
        plot_multiclass_curves(y_true, y_probs)

    # confusion matrix
    y_pred = np.argmax(y_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    # # ROC curve (binary only)
    # if y_prob_pos is not None:
    #     fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    #     plt.figure()
    #     plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.2f})')
    #     plt.plot([0,1],[0,1],'k--')
    #     plt.xlabel('FPR')
    #     plt.ylabel('TPR')
    #     plt.title('ROC Curve')
    #     plt.legend()
    #     plt.show()

    #     precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
    #     plt.figure()
    #     plt.plot(recall, precision, label=f'PR (AP={ap:.2f})')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision-Recall Curve')
    #     plt.legend()
    #     plt.show()

    return {
        'fold_accuracies': accs,
        'mean_accuracy': np.mean(accs),
        'std_accuracy': np.std(accs),
        'auc': auc,
        'average_precision': ap,
        'confusion_matrix': cm
    }


from sklearn.preprocessing import label_binarize

def plot_multiclass_curves(y_true, y_probs, class_names=None):
    """
    y_true: (n_samples,) integer labels
    y_probs: (n_samples, n_classes) predicted probabilities
    class_names: optional list of class names
    """
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'class {i}' + (f' ({class_names[i]})' if class_names else '') + f' AUC={auc_i:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Multiclass ROC Curve')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap_i = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'class {i}' + (f' ({class_names[i]})' if class_names else '') + f' AP={ap_i:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curve')
    plt.legend()
    plt.show()


# Define your base models
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=1)
base_models = [lr, rf, xgb] *3 # total 9 base models

results = cross_validated_random_feature_ensemble_with_plots(X, y, base_models, n_features=13)
print(results['mean_accuracy'], results['auc'])
print(results['confusion_matrix'])

results = cross_validated_random_feature_ensemble_with_plots(X, y, base_models, n_features=13, mode='stacking')
print(results['mean_accuracy'], results['auc'])
print(results['confusion_matrix'])


results = cross_validated_random_feature_ensemble_with_plots(X, y, base_models, n_features=13, mode='stacking')
print(results['mean_accuracy'], results['auc'])
print(results['confusion_matrix'])

# # ensemble with plots and classification reports


import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             classification_report)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

def plot_multiclass_curves(y_true, y_probs, class_names=None):
    """plot one-vs-rest ROC and PR curves for multiclass."""
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))


    unique_labels = np.unique(y_true)
    class_names = [str(lbl) for lbl in unique_labels]

    # ROC
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        name = class_names[i] if class_names else str(i)
        plt.plot(fpr, tpr, label=f'class {name} AUC={auc_i:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('Multiclass ROC (OvR)')
    plt.legend(); plt.show()

    # PR
    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap_i = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        name = class_names[i] if class_names else str(i)
        plt.plot(recall, precision, label=f'class {name} AP={ap_i:.2f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall (OvR)')
    plt.legend(); plt.show()


def cross_validated_random_feature_ensemble_new(
    X, y, base_models, n_features=35, n_splits=5, seed=42, mode='voting', class_names=None
):
    """
    CV evaluation of random-feature ensemble with metrics + ROC/PR plots.

    Returns:
        dict with:
          - fold accuracies
          - mean/std accuracy
          - macro/micro AUC & AP
          - confusion matrix
          - classification report (per-class precision/recall/f1/support)
    """
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, all_y_true, all_y_probs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        predictions = []

        for model in base_models:
            feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
            clf = clone(model).fit(X_train.iloc[:, feat_idx], y_train)

            if mode == 'voting':
                predictions.append(clf.predict_proba(X_test.iloc[:, feat_idx]))
            elif mode == 'stacking':
                predictions.append((clf.predict_proba(X_train.iloc[:, feat_idx]),
                                    clf.predict_proba(X_test.iloc[:, feat_idx])))

        if mode == 'voting':
            avg_preds = np.mean(predictions, axis=0)
            final_preds = np.argmax(avg_preds, axis=1)
            final_probs = avg_preds
        else:  # stacking
            meta_train = np.hstack([p[0] for p in predictions])
            meta_test = np.hstack([p[1] for p in predictions])
            meta_model = LogisticRegressionCV(max_iter=1000).fit(meta_train, y_train)
            final_preds = meta_model.predict(meta_test)
            final_probs = meta_model.predict_proba(meta_test)

        accs.append(accuracy_score(y_test, final_preds))
        all_y_true.append(y_test); all_y_probs.append(final_probs)

    # concat across folds
    y_true = np.concatenate(all_y_true)
    y_probs = np.concatenate(all_y_probs)
    y_pred = np.argmax(y_probs, axis=1)

    # metrics
    if y_probs.shape[1] == 2:
        y_prob_pos = y_probs[:,1]
        auc_macro = roc_auc_score(y_true, y_prob_pos)
        ap_macro = average_precision_score(y_true, y_prob_pos)
        auc_micro = auc_macro  # same in binary
        ap_micro = ap_macro

        # plots
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        plt.figure(); plt.plot(fpr, tpr, label=f'ROC AUC={auc_macro:.2f}')
        plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('ROC Curve'); plt.legend(); plt.show()

        prec, rec, _ = precision_recall_curve(y_true, y_prob_pos)
        plt.figure(); plt.plot(rec, prec, label=f'AP={ap_macro:.2f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision-Recall Curve'); plt.legend(); plt.show()

    else:
        auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        auc_micro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='micro')
        ap_macro = average_precision_score(y_true, y_probs, average='macro')
        ap_micro = average_precision_score(y_true, y_probs, average='micro')
        plot_multiclass_curves(y_true, y_probs, class_names=class_names)

    cm = confusion_matrix(y_true, y_pred)

    labels = np.unique(y_true)   # actual labels present
    class_names = [str(l) for l in labels]

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True
)
    # per-class precision/recall/f1
    # report = classification_report(
        # y_true, y_pred, target_names=class_names if class_names else None, output_dict=True
    # )

    return {
        'fold_accuracies': accs,
        'mean_accuracy': np.mean(accs),
        'std_accuracy': np.std(accs),
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'ap_macro': ap_macro,
        'ap_micro': ap_micro,
        'confusion_matrix': cm,
        'classification_report': report
    }


results = cross_validated_random_feature_ensemble_new(
    X, y, base_models, n_features=13, class_names=['A','B','C']
)

print("mean acc:", results['mean_accuracy'])
print("macro auc:", results['auc_macro'], "micro auc:", results['auc_micro'])
print("macro ap:", results['ap_macro'], "micro ap:", results['ap_micro'])
print("confusion matrix:\n", results['confusion_matrix'])
print("classification report:\n", results['classification_report'])

# # Earlier code to define X, y, base_models, etc. goes here


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define your base models
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=1)
base_models = [lr, rf, xgb] *3 # total 9 base models

# n_features = int(np.round(X.shape[1]* 0.01))  # number of features for each model

# n_features = [300, 400, 500, 600, 700, 800, 900, 1000]  # different feature sizes for each model
# n_features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
n_features = [13]

seed = 42  # random seed for reproducibility

results = []

for n_feat in n_features:
    print(f"Running ensemble with {n_feat} features per model and random seed {seed}...")

    # Voting ensemble
    start_time = time.perf_counter()
    print("Voting ensemble...")
    accs = cross_validated_random_feature_ensemble(X, y, base_models, n_features=n_feat, seed=seed, mode='voting')
    end_time = time.perf_counter()
    
    voting_ensemble_elapsed_time = end_time - start_time
    print(f"Voting ensemble completed in {voting_ensemble_elapsed_time:.2f} seconds")


    # Stacking ensemble
    print("Stacking ensemble...")
    start_time = time.perf_counter()
    stacking_accs = cross_validated_random_feature_ensemble(X, y, base_models, n_features=n_feat, seed=seed, mode='stacking')
    end_time = time.perf_counter()
    stacking_ensemble_elapsed_time = end_time - start_time
    print(f"Stacking ensemble completed in {stacking_ensemble_elapsed_time:.2f} seconds")

    voting_mean = np.mean(accs)
    voting_median = np.median(accs)
    voting_std = np.std(accs)

    stacking_mean = np.mean(stacking_accs)
    stacking_median = np.median(stacking_accs)
    stacking_std = np.std(stacking_accs)

    print(f"Voting Ensemble - Accuracy Mean: {voting_mean:.4f}, Median: {voting_median:.4f}, Std: {voting_std:.4f}, Time taken: {voting_ensemble_elapsed_time:.2f} seconds")
    print(f"Stacking Ensemble - Accuracy Mean: {stacking_mean:.4f}, Median: {stacking_median:.4f}, Std: {stacking_std:.4f}, Time taken: {stacking_ensemble_elapsed_time:.2f} seconds")
    print("-"*50)

     # Store results
    results.append({
        "n_features": n_feat,
        "seed": seed,
        "mode": "voting",
        "mean_accuracy": voting_mean,
        "median_accuracy": voting_median,
        "std_accuracy": voting_std,
        "time_seconds": voting_ensemble_elapsed_time
    })
    results.append({
        "n_features": n_feat,
        "seed": seed,
        "mode": "stacking",
        "mean_accuracy": stacking_mean,
        "median_accuracy": stacking_median,
        "std_accuracy": stacking_std,
        "time_seconds": stacking_ensemble_elapsed_time
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV

ensemble_result_file_name = f"{fs_utils.save_fig_path}/ensemble_results.csv"

df_results.to_csv(ensemble_result_file_name, index=False)
print(f'✅ Results saved to {ensemble_result_file_name}')    

# Save to CSV

# ensemble_result_file_name = f"{fs_utils.save_fig_path}/ensemble_results_only_MT.csv"

# df_results.to_csv(ensemble_result_file_name, index=False)
# print(f'✅ Results saved to {ensemble_result_file_name}')    

print(f'for dataset {fs_utils.config["dataset"]} ({data_final.shape[0]} samples, {data_final.shape[1]-1} features)')
print(f'{y.value_counts()}')




