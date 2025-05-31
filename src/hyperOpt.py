import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # Import LabelEncoder
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
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global Configuration ---
COLUMNS_TO_NORMALIZE = ['stem-width', 'stem-height', 'cap-diameter']
N_CV_SPLITS = 5
N_OPTUNA_TRIALS = 50
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

# MLP Objective Function
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
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx] # y_train_fold and y_val_fold will be numeric if y is numeric
        
        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)
        
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation, solver=solver,
            alpha=alpha, learning_rate_init=learning_rate_init, learning_rate=learning_rate_mode,
            max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE
        )
        model.fit(X_train_fold_norm, y_train_fold) # MLP will use numeric y_train_fold
        preds = model.predict(X_val_fold_norm)
        f1_scores_fold.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(f1_scores_fold)

# Decision Tree Objective Function
def objective_dt(trial, X, y, current_columns_to_normalize):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 3, 50) # Allow deeper trees, None is also an option if you add it conditionally

    # --- ADJUSTED SEARCH SPACES ---
    # Option 1: Use floats for proportions, but much smaller ranges
    min_samples_split_float = trial.suggest_float('min_samples_split_float', 0.001, 0.2, log=True)
    min_samples_leaf_float = trial.suggest_float('min_samples_leaf_float', 0.001, 0.1, log=True)

    # Option 2: Use integers for absolute counts (often more intuitive)
    # You can choose one option or even let Optuna choose which type to use:
    # For example, to use integers:
    # min_samples_split_int = trial.suggest_int('min_samples_split_int', 2, 100, log=True) # e.g., from 2 to 100 samples
    # min_samples_leaf_int = trial.suggest_int('min_samples_leaf_int', 1, 50, log=True)   # e.g., from 1 to 50 samples
    # For this example, I'll stick to revising the float versions as it's a smaller change from your original.

    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.04) # Max ccp_alpha can also be tuned

    f1_scores_fold = []
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        X_train_fold_norm, X_val_fold_norm = normalize_data_for_fold(X_train_fold, X_val_fold, current_columns_to_normalize)
        
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split_float, # Use the adjusted parameter
            min_samples_leaf=min_samples_leaf_float,   # Use the adjusted parameter
            ccp_alpha=ccp_alpha,
            random_state=RANDOM_STATE
        )
        model.fit(X_train_fold_norm, y_train_fold)
        preds = model.predict(X_val_fold_norm)
        f1_scores_fold.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0))
    return np.mean(f1_scores_fold)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hyperparameter optimization process...\n")

    data = load_data("data.csv")
    target_column_name = 'class' 
    if target_column_name not in data.columns:
        print(f"CRITICAL ERROR: Target column '{target_column_name}' not found.")
        exit()
        
    X_full = data.drop(columns=[target_column_name])
    y_full = data[target_column_name] # This is y_full before encoding
    
    print(f"Original X_full shape: {X_full.shape}")
    print(f"Original y_full dtype: {y_full.dtype}")
    if y_full.dtype == 'object' or not pd.api.types.is_numeric_dtype(y_full):
        print(f"Original y_full unique values (first 5): {y_full.unique()[:5]}")
        print("Target variable y_full is not numeric. Applying LabelEncoder...")
        le = LabelEncoder()
        # Ensure y_full is 1D for LabelEncoder
        y_full_encoded = le.fit_transform(y_full.astype(str)) # Convert to string first if mixed types, then encode
        y_full = pd.Series(y_full_encoded, name=target_column_name, index=y_full.index) # Preserve index
        print(f"Encoded y_full dtype: {y_full.dtype}")
        print(f"Encoded y_full unique values (first 5): {y_full.unique()[:5]}")
        print(f"LabelEncoder classes (mapping): {le.classes_}")
    else:
        print("Target variable y_full is already numeric.")


    # --- Preprocessing X_full (Feature Engineering) ---
    print("\nPreprocessing X_full features...")
    X_processed = X_full.copy()
    intended_numeric_cols = COLUMNS_TO_NORMALIZE.copy()

    for col in X_processed.columns:
        if col in intended_numeric_cols:
            # print(f"  Processing intended numeric column: {col}") # Verbose, can be enabled
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
            if X_processed[col].isnull().any():
                median_val = X_processed[col].median()
                if pd.isna(median_val):
                    # print(f"    Warning: Median is NaN for column '{col}'. Imputing its NaNs with 0.") # Verbose
                    X_processed[col] = X_processed[col].fillna(0)
                else:
                    # print(f"    Imputing NaNs in '{col}' with its median ({median_val:.2f}).") # Verbose
                    X_processed[col] = X_processed[col].fillna(median_val)
        elif X_processed[col].dtype == 'object':
            # print(f"  Processing column '{col}' as categorical (was object type).") # Verbose
            X_processed[col] = X_processed[col].fillna('missing_value').astype(str)

    categorical_cols_for_ohe = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols_for_ohe:
        # print(f"  Applying one-hot encoding to: {categorical_cols_for_ohe}") # Verbose
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols_for_ohe, prefix=categorical_cols_for_ohe, dummy_na=False)
        # print(f"  X_processed shape after one-hot encoding: {X_processed.shape}") # Verbose
    
    effective_columns_to_normalize = []
    # print(f"  Validating COLUMNS_TO_NORMALIZE ({COLUMNS_TO_NORMALIZE}) against processed columns:") # Verbose
    for col_name in COLUMNS_TO_NORMALIZE:
        if col_name in X_processed.columns:
            if pd.api.types.is_numeric_dtype(X_processed[col_name]):
                effective_columns_to_normalize.append(col_name)
            # else: # Verbose
                # print(f"    Warning: Column '{col_name}' (for normalization) is NOT numeric in X_processed. Skipping.")
        # else: # Verbose
            # print(f"    Warning: Column '{col_name}' (for normalization) not found in X_processed. Skipping.")
    # print(f"  Effective columns for MinMax normalization: {effective_columns_to_normalize}") # Verbose

    if X_processed.isnull().any().any():
        print("\nCRITICAL ERROR: NaNs found in X_processed after all preprocessing steps!")
        # ... (NaN reporting logic from before) ...
        exit()
    # else: # Verbose
        # print("  No NaNs found in final X_processed.")

    non_numeric_final_report = [col for col in X_processed.columns if not pd.api.types.is_numeric_dtype(X_processed[col])]
    if non_numeric_final_report:
        print(f"\nCRITICAL ERROR: These columns in X_processed are still NOT numeric: {non_numeric_final_report}")
        exit()
    
    print("Preprocessing of X features complete.")
    X_full = X_processed

    # 2. Initial Holdout
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full # y_full is now numeric
    )
    
    print(f"\nData split into: ")
    print(f"  X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape} (y_train_val dtype: {y_train_val.dtype})")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape} (y_test dtype: {y_test.dtype})\n")

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