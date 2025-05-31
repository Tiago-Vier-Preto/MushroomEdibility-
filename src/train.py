from src.models import KnnClassifier, naive_bayes_classifier, decision_tree_classifier, logistic_regression_classifier, mlp_classifier
from sklearn.model_selection import KFold
import numpy as np

def train_model(X, y):

    metrics = {
        "KNN": [],
        "Naive Bayes": [],
        "Decision Tree": [],
        "Logistic Regression": [],
        "MLP": []
    }

    # 5-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test = normalize_specific_columns(X_train, X_test, ['stem-width', 'stem-height', 'cap-diameter'])
    
        metrics["KNN"].append(KnnClassifier(X_train, y_train, X_test, y_test))
        metrics["Naive Bayes"].append(naive_bayes_classifier(X_train, y_train, X_test, y_test))
        metrics["Decision Tree"].append(decision_tree_classifier(X_train, y_train, X_test, y_test))
        metrics["Logistic Regression"].append(logistic_regression_classifier(X_train, y_train, X_test, y_test))
        metrics["MLP"].append(mlp_classifier(X_train, y_train, X_test, y_test))
    
    avg_metrics = {}
    for model, results in metrics.items():
        avg_metrics[model] = np.mean(results, axis=0)
    
    return avg_metrics


def normalize_specific_columns(train_data, test_data, columns):
    train_data = train_data.copy()
    test_data = test_data.copy()

    for column in columns:
        if column in train_data.columns:
            min_val = train_data[column].min()
            max_val = train_data[column].max()
            if max_val != min_val:
                train_data[column] = (train_data[column] - min_val) / (max_val - min_val)
                test_data[column] = (test_data[column] - min_val) / (max_val - min_val)
            else:
                # Avoid division by zero when all values are the same
                train_data[column] = 0.0
                test_data[column] = 0.0
    return train_data, test_data