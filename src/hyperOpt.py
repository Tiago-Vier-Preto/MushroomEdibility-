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

N_CV_SPLITS = 5
N_OPTUNA_TRIALS = 50
RANDOM_STATE = 42

def normalize_data_for_fold(x_train_fold, x_val_fold, columns_to_normalize):
    x_train_fold_norm = x_train_fold.copy()
    x_val_fold_norm = x_val_fold.copy()
    for col in columns_to_normalize:
        if col in x_train_fold_norm.columns:
            scaler = MinMaxScaler()
            x_train_fold_norm[col] = scaler.fit_transform(x_train_fold_norm[[col]])
            if col in x_val_fold_norm.columns:
                x_val_fold_norm[col] = scaler.transform(x_val_fold_norm[[col]])
    return x_train_fold_norm, x_val_fold_norm

def objective_knn(trial, x, y, columns_to_normalize):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p_val = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2

    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        x_train_fold_norm, x_val_fold_norm = normalize_data_for_fold(x_train_fold, x_val_fold, columns_to_normalize)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p_val)
        model.fit(x_train_fold_norm, y_train_fold)
        preds = model.predict(x_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

def objective_mlp(trial, x, y, columns_to_normalize):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = [trial.suggest_int(f'n_units_l{i+1}', 10, 200, log=True) for i in range(n_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    learning_rate_mode = trial.suggest_categorical('learning_rate_mode', ['constant', 'invscaling', 'adaptive']) if solver == 'sgd' else 'constant'
    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        x_train_fold_norm, x_val_fold_norm = normalize_data_for_fold(x_train_fold, x_val_fold, columns_to_normalize)

        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation, solver=solver,
            alpha=alpha, learning_rate_init=learning_rate_init, learning_rate=learning_rate_mode,
            max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE
        )
        model.fit(x_train_fold_norm, y_train_fold)
        preds = model.predict(x_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

def objective_dt(trial, x, y, current_columns_to_normalize):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth_val = trial.suggest_int('max_depth', 5, 50)
    min_samples_split_val = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf_val = trial.suggest_int('min_samples_leaf', 1, 10)
    ccp_alpha_val = trial.suggest_float('ccp_alpha', 0.0, 0.01) 
    precision_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        x_train_fold_norm, x_val_fold_norm = normalize_data_for_fold(x_train_fold, x_val_fold, current_columns_to_normalize)
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth_val,
            min_samples_split=min_samples_split_val,
            min_samples_leaf=min_samples_leaf_val,
            ccp_alpha=ccp_alpha_val,
            random_state=RANDOM_STATE
        )
        model.fit(x_train_fold_norm, y_train_fold)
        preds = model.predict(x_val_fold_norm)
        precision_scores_fold.append(precision_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(precision_scores_fold)

if __name__ == "__main__":
    print("Preprocessing:\n")
    raw_data = load_data("data.csv")
    data = pre_process_data(raw_data.copy())

    x_data = data.drop(columns=['class'])
    y_data = data['class']

    if y_data.dtype == 'object' or not pd.api.types.is_numeric_dtype(y_data):
        le = LabelEncoder()
        y_full_encoded = le.fit_transform(y_data.astype(str))
        y_full = pd.Series(y_full_encoded, name='class', index=y_data.index)
    else:
        y_full = y_data

    normalize_columns = ['stem-width', 'stem-height', 'cap-diameter']
    print("Preprocessing complete.")

    print("Hyperparameter optimization start:\n")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_data, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )

    optimize_args = {
        "n_trials": N_OPTUNA_TRIALS,
        "show_progress_bar": True
    }

    print(f"----- Optimizing KNN -----\n")
    study_knn = optuna.create_study(direction='maximize', study_name='knn_optimization')
    study_knn.optimize(lambda trial: objective_knn(trial, x_train_val, y_train_val, normalize_columns), **optimize_args)
    print("\nBest KNN Hyperparameters:", study_knn.best_params)
    print(f"Best KNN Cross-Validated Precision: {study_knn.best_value:.4f}\n")

    print(f"----- Optimizing Decision Tree -----\n")
    study_dt = optuna.create_study(direction='maximize', study_name='dt_optimization')
    study_dt.optimize(lambda trial: objective_dt(trial, x_train_val, y_train_val, normalize_columns), **optimize_args)
    print("\nBest Decision Tree Hyperparameters:", study_dt.best_params)
    print(f"Best Decision Tree Cross-Validated Precision: {study_dt.best_value:.4f}\n")

    print(f"----- Optimizing MLP -----\n")
    study_mlp = optuna.create_study(direction='maximize', study_name='mlp_optimization')
    study_mlp.optimize(lambda trial: objective_mlp(trial, x_train_val, y_train_val, normalize_columns), **optimize_args)
    print("\nBest MLP Hyperparameters:", study_mlp.best_params)
    print(f"Best MLP Cross-Validated Precision: {study_mlp.best_value:.4f}\n")

    print("Hyperparameter optimization complete.")