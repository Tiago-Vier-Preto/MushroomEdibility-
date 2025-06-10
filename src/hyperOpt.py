import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from utils import load_data
from preProcessingData import pre_process_data

COLUMNS_TO_NORMALIZE = ['stem-width', 'stem-height', 'cap-diameter']
N_CV_SPLITS = 5
N_OPTUNA_TRIALS = 50
RANDOM_STATE = 42

def normalize_data_for_fold(X_train_fold, X_val_fold, columns_to_normalize):
    X_train_fold_norm = X_train_fold.copy()
    X_val_fold_norm = X_val_fold.copy()

    for col in columns_to_normalize:
        if col in X_train_fold_norm.columns:
            if pd.api.types.is_numeric_dtype(X_train_fold_norm[col]):
                scaler = MinMaxScaler()
                X_train_fold_norm[col] = scaler.fit_transform(X_train_fold_norm[[col]])
                if col in X_val_fold_norm.columns:
                    X_val_fold_norm[col] = scaler.transform(X_val_fold_norm[[col]])
    return X_train_fold_norm, X_val_fold_norm

def objective_knn(trial, X, y, current_columns_to_normalize):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p_val = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2

    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p_val)
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

def objective_mlp(trial, X, y, current_columns_to_normalize):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = [trial.suggest_int(f'n_units_l{i+1}', 10, 200, log=True) for i in range(n_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    learning_rate_mode = trial.suggest_categorical('learning_rate_mode', ['constant', 'invscaling', 'adaptive']) if solver == 'sgd' else 'constant'

    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)

        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation, solver=solver,
            alpha=alpha, learning_rate_init=learning_rate_init, learning_rate=learning_rate_mode,
            max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE
        )
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

def objective_dt(trial, X, y, current_columns_to_normalize):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 3, 50)

    min_samples_split_float = trial.suggest_float('min_samples_split_float', 0.001, 0.2, log=True)
    min_samples_leaf_float = trial.suggest_float('min_samples_leaf_float', 0.001, 0.1, log=True)

    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.04)

    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split_float,
            min_samples_leaf=min_samples_leaf_float,
            ccp_alpha=ccp_alpha,
            random_state=RANDOM_STATE
        )
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

if __name__ == "__main__":
    print("Hyperparameter optimization start:\n")

    raw_data = load_data("data.csv")
    data_after_user_preprocessing = pre_process_data(raw_data.copy())

    target_column_name = 'class'
    X_from_user_preprocessing = data_after_user_preprocessing.drop(columns=[target_column_name])
    y_from_user_preprocessing = data_after_user_preprocessing[target_column_name]

    print(f"\nOriginal y (from user processed data) dtype: {y_from_user_preprocessing.dtype}")
    if y_from_user_preprocessing.dtype == 'object' or not pd.api.types.is_numeric_dtype(y_from_user_preprocessing):
        print(f"Original y unique values (first 5): {y_from_user_preprocessing.unique()[:5]}")
        print("Target variable y is not numeric. Applying LabelEncoder...")
        le = LabelEncoder()
        y_full_encoded = le.fit_transform(y_from_user_preprocessing.astype(str))
        y_full = pd.Series(y_full_encoded, name=target_column_name, index=y_from_user_preprocessing.index)
        print(f"Encoded y_full dtype: {y_full.dtype}")
        print(f"Encoded y_full unique values (first 5): {y_full.unique()[:5]}")
        print(f"LabelEncoder classes (mapping): {le.classes_}")
    else:
        y_full = y_from_user_preprocessing 
        print("Target variable y (from user processed data) is already numeric.")

    print(f"\nX_full shape after user's pre_process_data and dropping target: {X_from_user_preprocessing.shape}")

    X_final_features = X_from_user_preprocessing.copy() 

    print(f"\nEnsuring specific COLUMNS_TO_NORMALIZE ({COLUMNS_TO_NORMALIZE}) are clean and numeric:")
    for col in COLUMNS_TO_NORMALIZE:
        if col in X_final_features.columns:

            if not pd.api.types.is_numeric_dtype(X_final_features[col]):
                print(f"  Column '{col}' is not numeric (dtype: {X_final_features[col].dtype}). Attempting conversion to numeric...")
                X_final_features[col] = pd.to_numeric(X_final_features[col], errors='coerce')

            if X_final_features[col].isnull().any():
                print(f"  Column '{col}' contains NaNs. Imputing...")
                median_val = X_final_features[col].median()
                if pd.isna(median_val): 
                    print(f"    Median for '{col}' is NaN (column might be all NaNs). Imputing with 0.")
                    X_final_features[col] = X_final_features[col].fillna(0)
                else:
                    print(f"    Imputing NaNs in '{col}' with its median ({median_val:.2f}).")
                    X_final_features[col] = X_final_features[col].fillna(median_val)
        else:
            print(f"  Warning: Column '{col}' specified in COLUMNS_TO_NORMALIZE not found in X features after user preprocessing.")

    effective_columns_to_normalize = []
    for col_name in COLUMNS_TO_NORMALIZE:
        if col_name in X_final_features.columns and pd.api.types.is_numeric_dtype(X_final_features[col_name]):
            effective_columns_to_normalize.append(col_name)
    print(f"Effective columns for MinMax normalization within folds: {effective_columns_to_normalize}")

    if X_final_features.isnull().any().any():
        nan_cols_report = X_final_features.columns[X_final_features.isnull().any()].tolist()
        print(f"\nCRITICAL ERROR: NaNs found in final X features after all preprocessing steps! Columns with NaNs: {nan_cols_report}")
        for ncol in nan_cols_report:
            print(f"    NaN count in '{ncol}': {X_final_features[ncol].isnull().sum()}")
        exit()

    non_numeric_final_report = [col for col in X_final_features.columns if not pd.api.types.is_numeric_dtype(X_final_features[col])]
    if non_numeric_final_report:
        print(f"\nCRITICAL ERROR: These columns in final X features are still NOT numeric: {non_numeric_final_report}")
        for col_report in non_numeric_final_report:
            print(f"   Column '{col_report}' dtype: {X_final_features[col_report].dtype}")
        exit()

    print("Preprocessing of X features (hyperopt.py specific part) complete.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_final_features, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )

    print(f"\nData split into: ")
    print(f"  X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape} (y_train_val dtype: {y_train_val.dtype})")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape} (y_test dtype: {y_test.dtype})\n")

    common_optimize_args = {
        "n_trials": N_OPTUNA_TRIALS,
        "show_progress_bar": True
    }

    print(f"--- Optimizing KNN ---")
    study_knn = optuna.create_study(direction='maximize', study_name='knn_optimization')
    study_knn.optimize(lambda trial: objective_knn(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest KNN Hyperparameters:", study_knn.best_params)
    print(f"Best KNN Cross-Validated Precision: {study_knn.best_value:.4f}\n")

    print(f"--- Optimizing Decision Tree ---")
    study_dt = optuna.create_study(direction='maximize', study_name='dt_optimization')
    study_dt.optimize(lambda trial: objective_dt(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest Decision Tree Hyperparameters:", study_dt.best_params)
    print(f"Best Decision Tree Cross-Validated Precision: {study_dt.best_value:.4f}\n")

    print(f"--- Optimizing MLP ---")
    study_mlp = optuna.create_study(direction='maximize', study_name='mlp_optimization')
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest MLP Hyperparameters:", study_mlp.best_params)
    print(f"Best MLP Cross-Validated Precision: {study_mlp.best_value:.4f}\n")

    print("Hyperparameter optimization complete.")