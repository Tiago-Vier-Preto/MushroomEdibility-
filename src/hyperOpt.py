import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
# Assuming utils.py is in the same directory (src) or Python path is configured
from utils import load_data

# Suppress ConvergenceWarning for MLPClassifier if it doesn't converge in some trials
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress the fillna inplace warning for now, though addressed

# --- Global Configuration ---
COLUMNS_TO_NORMALIZE = ['stem-width', 'stem-height', 'cap-diameter'] # Numeric columns for MinMax scaling
N_CV_SPLITS = 5
N_OPTUNA_TRIALS = 50 # Adjust as needed
RANDOM_STATE = 42

# --- Normalization Function (remains the same) ---
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
            else:
                print(f"Warning (in normalize_data_for_fold): Column '{col}' is not numeric and will not be scaled.")
        else:
            print(f"Warning (in normalize_data_for_fold): Column '{col}' not found in training data fold for normalization.")
    return X_train_fold_norm, X_val_fold_norm

# --- Objective Functions (KNN, DT, MLP - largely same, ensure they use the effective columns list) ---
def objective_knn(trial, X, y, current_columns_to_normalize):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p_val = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2

    f1_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p_val)
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        f1_scores_fold.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(f1_scores_fold)

def objective_dt(trial, X, y, current_columns_to_normalize):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 1.0)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.01, 0.5)
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.035)

    f1_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)
        
        model = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, random_state=RANDOM_STATE
        )
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        f1_scores_fold.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(f1_scores_fold)

def objective_mlp(trial, X, y, current_columns_to_normalize):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = [trial.suggest_int(f'n_units_l{i+1}', 10, 200, log=True) for i in range(n_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    learning_rate_mode = trial.suggest_categorical('learning_rate_mode', ['constant', 'invscaling', 'adaptive']) if solver == 'sgd' else 'constant'

    f1_scores_fold = []
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
        f1_scores_fold.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(f1_scores_fold)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hyperparameter optimization process...\n")

    data = load_data("data.csv")
    target_column_name = 'class' # Ensure this matches your target column
    if target_column_name not in data.columns:
        print(f"CRITICAL ERROR: Target column '{target_column_name}' not found.")
        exit()
        
    X_full = data.drop(columns=[target_column_name])
    y_full = data[target_column_name]
    print(f"Original X_full shape: {X_full.shape}")

    # --- Preprocessing X_full to ensure all features are numeric and no NaNs ---
    print("\nPreprocessing X_full...")
    X_processed = X_full.copy()

    # Identify columns intended to be numeric vs categorical from the start
    # For this example, we assume COLUMNS_TO_NORMALIZE are definitely numeric.
    # Other columns, if object type, will be treated as categorical.
    
    intended_numeric_cols = COLUMNS_TO_NORMALIZE.copy() # Start with these
    # You might want to add other column names here if they are numeric but not scaled

    for col in X_processed.columns:
        if col in intended_numeric_cols:
            print(f"  Processing intended numeric column: {col}")
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
            if X_processed[col].isnull().any(): # Check if NaNs exist
                median_val = X_processed[col].median()
                if pd.isna(median_val): # True if column became all NaNs or was already all NaNs
                    print(f"    Warning: Median is NaN for column '{col}'. Imputing its NaNs with 0.")
                    X_processed[col] = X_processed[col].fillna(0) # Impute with 0 if median is NaN
                else:
                    print(f"    Imputing NaNs in '{col}' with its median ({median_val:.2f}).")
                    X_processed[col] = X_processed[col].fillna(median_val)
        elif X_processed[col].dtype == 'object':
            print(f"  Processing column '{col}' as categorical (was object type).")
            # Fill NaNs with a placeholder string BEFORE converting to string, then convert all to string
            X_processed[col] = X_processed[col].fillna('missing_value').astype(str)


    # One-hot encode all columns that are now 'object' or 'category' type
    # This will include columns processed above if they ended up as strings.
    categorical_cols_for_ohe = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols_for_ohe:
        print(f"  Applying one-hot encoding to: {categorical_cols_for_ohe}")
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols_for_ohe, prefix=categorical_cols_for_ohe, dummy_na=False) # NaNs were filled
        print(f"  X_processed shape after one-hot encoding: {X_processed.shape}")
    
    # Validate and update the list of columns to normalize based on X_processed
    effective_columns_to_normalize = []
    print(f"  Validating COLUMNS_TO_NORMALIZE ({COLUMNS_TO_NORMALIZE}) against processed columns:")
    for col_name in COLUMNS_TO_NORMALIZE: # Iterate original list
        if col_name in X_processed.columns: # Check if it still exists (it should, if it was numeric)
            if pd.api.types.is_numeric_dtype(X_processed[col_name]):
                effective_columns_to_normalize.append(col_name)
            else: # Should not happen if logic above is correct for intended_numeric_cols
                print(f"    Warning: Column '{col_name}' (for normalization) is NOT numeric in X_processed. Skipping.")
        else: # If an intended numeric col for scaling got OHE'd (shouldn't happen with this logic)
            print(f"    Warning: Column '{col_name}' (for normalization) not found in X_processed. Skipping.")
    print(f"  Effective columns for MinMax normalization: {effective_columns_to_normalize}")

    # Final check for any NaNs remaining
    if X_processed.isnull().any().any():
        print("\nCRITICAL ERROR: NaNs found in X_processed after all preprocessing steps!")
        nan_cols_report = X_processed.columns[X_processed.isnull().any()].tolist()
        print(f"  Columns with NaNs: {nan_cols_report}")
        for ncol in nan_cols_report:
            print(f"    NaN count in '{ncol}': {X_processed[ncol].isnull().sum()}")
        exit()
    else:
        print("  No NaNs found in final X_processed.")

    # Final check for non-numeric columns
    non_numeric_final_report = [col for col in X_processed.columns if not pd.api.types.is_numeric_dtype(X_processed[col])]
    if non_numeric_final_report:
        print(f"\nCRITICAL ERROR: These columns in X_processed are still NOT numeric: {non_numeric_final_report}")
        exit()
    
    print("Preprocessing complete. All features in X_processed are numeric and NaN-free.")
    X_full = X_processed # Use the fully processed features

    # 2. Initial Holdout
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )
    
    print(f"\nData split into: ")
    print(f"  X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")

    # --- Run Optuna Studies ---
    common_optimize_args = {
        "n_trials": N_OPTUNA_TRIALS,
        "show_progress_bar": True
    }

    print(f"--- Optimizing KNN ---")
    study_knn = optuna.create_study(direction='maximize', study_name='knn_optimization')
    study_knn.optimize(lambda trial: objective_knn(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest KNN Hyperparameters:", study_knn.best_params)
    print(f"Best KNN Cross-Validated F1-score: {study_knn.best_value:.4f}\n")

    print(f"--- Optimizing Decision Tree ---")
    study_dt = optuna.create_study(direction='maximize', study_name='dt_optimization')
    study_dt.optimize(lambda trial: objective_dt(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest Decision Tree Hyperparameters:", study_dt.best_params)
    print(f"Best Decision Tree Cross-Validated F1-score: {study_dt.best_value:.4f}\n")

    print(f"--- Optimizing MLP ---")
    study_mlp = optuna.create_study(direction='maximize', study_name='mlp_optimization')
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train_val, y_train_val, effective_columns_to_normalize), **common_optimize_args)
    print("\nBest MLP Hyperparameters:", study_mlp.best_params)
    print(f"Best MLP Cross-Validated F1-score: {study_mlp.best_value:.4f}\n")

    print("Hyperparameter optimization complete.")